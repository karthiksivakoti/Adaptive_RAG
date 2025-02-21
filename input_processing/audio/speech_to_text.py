# risk_rag_system/input_processing/audio/speech_to_text.py

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pydantic import BaseModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from loguru import logger
from pathlib import Path

class TranscriptionConfig(BaseModel):
    """Configuration for speech-to-text processing"""
    model_name: str = "openai/whisper-large-v3"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    chunk_length: int = 30  # seconds
    batch_size: int = 8
    language: Optional[str] = None
    task: str = "transcribe"  # or "translate"
    return_timestamps: bool = True

class TranscriptionResult(BaseModel):
    """Structure for transcription results"""
    text: str
    segments: List[Dict[str, Any]]
    language: str
    confidence: float
    word_timestamps: Optional[List[Dict[str, Any]]] = None

class SpeechToText:
    """Handles speech-to-text processing"""
    
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self._initialize_model()
        logger.info(f"Initialized SpeechToText with model: {self.config.model_name}")

    def _initialize_model(self) -> None:
        """Initialize the speech-to-text model"""
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name
            )
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            self.model.to(self.config.device)
            logger.info("Successfully loaded speech-to-text model")
        except Exception as e:
            logger.error(f"Error initializing speech-to-text model: {e}")
            raise

    async def transcribe_stream(
        self,
        audio_chunks: List[np.ndarray],
        sample_rate: int
    ) -> TranscriptionResult:
        """Transcribe streaming audio chunks"""
        try:
            # Process audio chunks
            features = self._prepare_audio(audio_chunks, sample_rate)
            
            # Generate transcription
            transcription = await self._generate_transcription(features)
            
            # Post-process results
            result = self._process_transcription(transcription)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio stream: {e}")
            raise

    async def transcribe_file(
        self,
        filepath: Path,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe audio file"""
        try:
            # Load and process audio
            audio_input = self.processor.load_audio(str(filepath))
            features = self.processor(
                audio_input,
                sampling_rate=self.processor.sampling_rate,
                return_tensors="pt"
            ).to(self.config.device)
            
            # Override language if specified
            if language:
                self.config.language = language
            
            # Generate transcription
            transcription = await self._generate_transcription(features)
            
            # Process results
            result = self._process_transcription(transcription)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio file: {e}")
            raise

    def _prepare_audio(
        self,
        audio_chunks: List[np.ndarray],
        sample_rate: int
    ) -> Dict[str, torch.Tensor]:
        """Prepare audio chunks for model input"""
        # Concatenate chunks
        audio = np.concatenate(audio_chunks)
        
        # Resample if needed
        if sample_rate != self.processor.sampling_rate:
            audio = self._resample_audio(
                audio,
                sample_rate,
                self.processor.sampling_rate
            )
            
        # Process audio
        features = self.processor(
            audio,
            sampling_rate=self.processor.sampling_rate,
            return_tensors="pt"
        ).to(self.config.device)
        
        return features

    def _resample_audio(
        self,
        audio: np.ndarray,
        orig_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_rate == target_rate:
            return audio
            
        # Calculate resampling ratio
        ratio = target_rate / orig_rate
        
        # Calculate new length
        new_length = int(len(audio) * ratio)
        
        # Resample using linear interpolation
        resampled = np.interp(
            np.linspace(0, len(audio) - 1, new_length),
            np.arange(len(audio)),
            audio
        )
        
        return resampled

    async def _generate_transcription(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Generate transcription from processed audio"""
        with torch.no_grad():
            # Generate transcription
            outputs = self.model.generate(
                input_features=features.input_features,
                language=self.config.language,
                task=self.config.task,
                return_timestamps=self.config.return_timestamps
            )
            
            # Decode outputs
            transcription = self.processor.decode(
                outputs[0],
                output_word_offsets=True
            )
            
            return transcription

    def _process_transcription(
        self,
        transcription: Dict[str, Any]
    ) -> TranscriptionResult:
        """Process and structure transcription results"""
        # Extract text and metadata
        text = transcription.text
        segments = transcription.get("chunks", [])
        
        # Calculate confidence scores
        if hasattr(transcription, "token_probs"):
            confidence = float(np.mean(transcription.token_probs))
        else:
            confidence = self._estimate_confidence(segments)
            
        # Process word timestamps if available
        word_timestamps = None
        if transcription.get("word_offsets"):
            word_timestamps = [
                {
                    "word": word,
                    "start": start,
                    "end": end
                }
                for word, (start, end) in zip(
                    transcription.words,
                    transcription.word_offsets
                )
            ]

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=transcription.language,
            confidence=confidence,
            word_timestamps=word_timestamps
        )

    def _estimate_confidence(self, segments: List[Dict[str, Any]]) -> float:
        """Estimate confidence score from segments"""
        if not segments:
            return 0.0
            
        # Use segment probabilities if available
        if all("probability" in seg for seg in segments):
            return float(np.mean([seg["probability"] for seg in segments]))
            
        # Fallback estimation based on segment consistency
        consistency_scores = []
        
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]
            
            # Check timing consistency
            timing_score = 1.0
            if current.get("end") and next_seg.get("start"):
                gap = next_seg["start"] - current["end"]
                if gap > 1.0:  # Gap larger than 1 second
                    timing_score *= 0.8
                    
            # Check text coherence
            text_score = 1.0
            if len(current.get("text", "").split()) < 2:
                text_score *= 0.9  # Penalize very short segments
                
            consistency_scores.append((timing_score + text_score) / 2)
            
        return float(np.mean(consistency_scores)) if consistency_scores else 0.5

    def cleanup(self) -> None:
        """Cleanup model resources"""
        try:
            del self.model
            del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleaned up SpeechToText resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the processor"""
        return {
            "config": self.config.dict(),
            "model_loaded": hasattr(self, "model"),
            "device": self.config.device,
            "language": self.config.language
        }