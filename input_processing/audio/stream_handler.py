# risk_rag_system/input_processing/audio/stream_handler.py

from typing import BinaryIO, Dict, Any, Optional, Generator
import wave
import numpy as np
from pydantic import BaseModel
import pyaudio
import time
from loguru import logger
from pathlib import Path
from typing import AsyncIterator

class AudioConfig(BaseModel):
    """Configuration for audio processing"""
    channels: int = 1
    sample_rate: int = 16000
    chunk_size: int = 1024
    format: int = pyaudio.paFloat32
    silence_threshold: float = 0.03
    silence_duration: float = 1.0  # seconds
    max_duration: int = 300  # seconds

class AudioMetadata(BaseModel):
    """Metadata for audio streams"""
    duration: float
    sample_rate: int
    channels: int
    format: str
    file_size: int
    encoding: str

class StreamHandler:
    """Handles audio stream processing and management"""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._audio = pyaudio.PyAudio()
        self._stream = None
        self._temp_dir = Path("./temp/audio")
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized StreamHandler")

    async def start_stream(self) -> None:
        """Start audio input stream"""
        try:
            self._stream = self._audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._callback
            )
            logger.info("Started audio stream")
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            raise

    def _callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: Dict[str, Any],
        status: int
    ) -> tuple:
        """Callback for audio stream processing"""
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Check for silence
            if self._is_silence(audio_data):
                return (in_data, pyaudio.paContinue)
            
            # Process active audio
            processed_data = self._process_audio(audio_data)
            return (processed_data.tobytes(), pyaudio.paContinue)
            
        except Exception as e:
            logger.error(f"Error in stream callback: {e}")
            return (in_data, pyaudio.paAbort)

    def _is_silence(self, audio_data: np.ndarray) -> bool:
        """Check if audio chunk is silence"""
        return np.max(np.abs(audio_data)) < self.config.silence_threshold

    def _process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio chunk"""
        # Apply preprocessing
        # 1. Normalize
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # 2. Remove DC offset
        audio_data = audio_data - np.mean(audio_data)
        
        # 3. Apply basic noise reduction
        noise_floor = np.mean(np.abs(audio_data)) * 2
        audio_data[np.abs(audio_data) < noise_floor] = 0
        
        return audio_data

    async def save_stream(
        self,
        filepath: Path,
        duration: Optional[float] = None
    ) -> Path:
        """Save audio stream to file"""
        try:
            frames = []
            start_time = time.time()
            max_duration = duration or self.config.max_duration
            
            while True:
                if self._stream.is_active():
                    data = self._stream.read(self.config.chunk_size)
                    frames.append(data)
                    
                    if time.time() - start_time > max_duration:
                        break
                else:
                    break

            # Save to WAV file
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(self._audio.get_sample_size(self.config.format))
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(b''.join(frames))

            logger.info(f"Saved audio stream to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving audio stream: {e}")
            raise

    async def load_file(self, filepath: Path) -> tuple[np.ndarray, AudioMetadata]:
        """Load audio file and return data with metadata"""
        try:
            with wave.open(str(filepath), 'rb') as wf:
                # Get file metadata
                metadata = AudioMetadata(
                    duration=wf.getnframes() / wf.getframerate(),
                    sample_rate=wf.getframerate(),
                    channels=wf.getnchannels(),
                    format=f"WAV {wf.getsampwidth() * 8}bit",
                    file_size=filepath.stat().st_size,
                    encoding=f"PCM {wf.getsampwidth() * 8}bit"
                )
                
                # Read audio data
                audio_data = np.frombuffer(
                    wf.readframes(wf.getnframes()),
                    dtype=np.float32
                )
                
                return audio_data, metadata

        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise

    async def stream_chunks(
        self,
        chunk_duration: float = 0.5
    ) -> AsyncIterator[np.ndarray]:
        """Stream audio in chunks"""
        if not self._stream:
            raise RuntimeError("Stream not started")
            
        chunk_samples = int(chunk_duration * self.config.sample_rate)
        buffer = np.array([], dtype=np.float32)
        
        while True:
            if len(buffer) >= chunk_samples:
                # Yield chunk and keep remainder
                yield buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]
            else:
                # Read more data
                if self._stream.is_active():
                    data = self._stream.read(self.config.chunk_size)
                    new_data = np.frombuffer(data, dtype=np.float32)
                    buffer = np.concatenate([buffer, new_data])
                else:
                    break

    def cleanup(self) -> None:
        """Cleanup resources"""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._audio:
            self._audio.terminate()
        logger.info("Cleaned up StreamHandler resources")

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the handler"""
        return {
            "config": self.config.dict(),
            "stream_active": self._stream is not None and self._stream.is_active(),
            "temp_dir": str(self._temp_dir)
        }