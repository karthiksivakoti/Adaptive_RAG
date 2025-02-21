# risk_rag_system/llm/openai_handler.py

from typing import Dict, Any, Optional, List, Union, AsyncGenerator
import asyncio
import aiohttp
from pydantic import BaseModel
from loguru import logger
import json
import numpy as np
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken

class OpenAIConfig(BaseModel):
    """Configuration for OpenAI API"""
    api_key: str
    model: str = "gpt-4-turbo-preview"
    api_base: str = "https://api.openai.com/v1"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 30.0
    max_retries: int = 3
    batch_size: int = 10

class OpenAIHandler:
    """Handler for OpenAI API interactions"""
    
    def __init__(
        self,
        config: Optional[OpenAIConfig] = None,
        session: Optional[aiohttp.ClientSession] = None
    ):
        self.config = config or OpenAIConfig()
        self.session = session
        self._setup_session_headers()
        self.tokenizer = tiktoken.encoding_for_model(self.config.model)
        logger.info(f"Initialized OpenAIHandler with model: {self.config.model}")

    def _setup_session_headers(self) -> None:
        """Setup API request headers"""
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """Generate completion using OpenAI API"""
        try:
            await self._ensure_session()
            
            # Prepare request data
            data = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty
            }
            
            if stop_sequences:
                data["stop"] = stop_sequences
            
            if functions:
                data["functions"] = functions
                if function_call:
                    data["function_call"] = function_call
            
            # Make API request
            async with self.session.post(
                f"{self.config.api_base}/chat/completions",
                json=data
            ) as response:
                if response.status != 200:
                    error_data = await response.text()
                    raise Exception(f"API error: {error_data}")
                
                result = await response.json()
            
            # Extract response
            message = result["choices"][0]["message"]
            
            if functions and "function_call" in message:
                return {
                    "function": message["function_call"]["name"],
                    "arguments": json.loads(message["function_call"]["arguments"])
                }
            
            return message["content"]
            
        except Exception as e:
            logger.error(f"Error in OpenAI completion: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> np.ndarray:
        """Get embeddings using OpenAI API"""
        try:
            await self._ensure_session()
            embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                
                data = {
                    "model": model,
                    "input": batch
                }
                
                async with self.session.post(
                    f"{self.config.api_base}/embeddings",
                    json=data
                ) as response:
                    if response.status != 200:
                        error_data = await response.text()
                        raise Exception(f"API error: {error_data}")
                    
                    result = await response.json()
                
                # Extract embeddings
                batch_embeddings = [item["embedding"] for item in result["data"]]
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_logprobs(
        self,
        text: str,
        topk: int = 5,
        model: Optional[str] = None
    ) -> Dict[str, List[Dict[str, float]]]:
        """Get token logprobs using OpenAI API"""
        try:
            await self._ensure_session()
            
            data = {
                "model": model or self.config.model,
                "prompt": text,
                "max_tokens": 0,
                "echo": True,
                "logprobs": topk
            }
            
            async with self.session.post(
                f"{self.config.api_base}/completions",
                json=data
            ) as response:
                if response.status != 200:
                    error_data = await response.text()
                    raise Exception(f"API error: {error_data}")
                
                result = await response.json()
            
            # Extract logprobs
            logprobs = result["choices"][0]["logprobs"]
            return {
                "token_logprobs": [
                    dict(zip(top_tokens, map(float, top_logprobs)))
                    for top_tokens, top_logprobs in zip(
                        logprobs["top_tokens"],
                        logprobs["top_logprobs"]
                    )
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting logprobs: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def create_moderation(
        self,
        text: str,
        model: str = "text-moderation-latest"
    ) -> Dict[str, Any]:
        """Create content moderation using OpenAI API"""
        try:
            await self._ensure_session()
            
            data = {
                "input": text,
                "model": model
            }
            
            async with self.session.post(
                f"{self.config.api_base}/moderations",
                json=data
            ) as response:
                if response.status != 200:
                    error_data = await response.text()
                    raise Exception(f"API error: {error_data}")
                
                result = await response.json()
            
            return result["results"][0]
            
        except Exception as e:
            logger.error(f"Error in content moderation: {e}")
            raise

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat completions"""
        try:
            await self._ensure_session()
            
            data = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature or self.config.temperature,
                "stream": True
            }
            
            async with self.session.post(
                f"{self.config.api_base}/chat/completions",
                json=data
            ) as response:
                if response.status != 200:
                    error_data = await response.text()
                    raise Exception(f"API error: {error_data}")
                
                # Process streaming response
                async for line in response.content:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.startswith(b"data: "):
                        line = line[6:]  # Remove "data: " prefix
                    if line == b"[DONE]":
                        break
                        
                    try:
                        chunk = json.loads(line)
                        if len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue
                    
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count exact number of tokens in text"""
        return len(self.tokenizer.encode(text))

    def estimate_cost(self, tokens: int, model: Optional[str] = None) -> float:
        """Estimate API cost for tokens"""
        model = model or self.config.model
        # Pricing per 1K tokens (as of Feb 2024)
        pricing = {
            "gpt-4-turbo-preview": (0.01, 0.03),  # Input, Output
            "gpt-4": (0.03, 0.06),
            "gpt-3.5-turbo": (0.0005, 0.0015),
            "text-embedding-3-small": (0.00002, 0.00002),
            "text-embedding-3-large": (0.00013, 0.00013)
        }
        
        if model not in pricing:
            raise ValueError(f"Unknown model for pricing: {model}")
            
        input_cost, output_cost = pricing[model]
        # Assuming half input, half output for estimation
        return (tokens / 1000) * ((input_cost + output_cost) / 2)

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

    def get_state(self) -> Dict[str, Any]:
        """Get current handler state"""
        return {
            "model": self.config.model,
            "api_base": self.config.api_base,
            "max_tokens": self.config.max_tokens,
            "has_session": self.session is not None
        }