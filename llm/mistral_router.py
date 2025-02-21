# risk_rag_system/llm/mistral_router.py

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
import json
import asyncio
import aiohttp
from functools import lru_cache

class RouterConfig(BaseModel):
    """Configuration for Mistral router"""
    model_name: str = "microsoft/phi-2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_tokens: int = 2048
    temperature: float = 0.7
    fallback_url: Optional[str] = None
    api_key: Optional[str] = None
    cache_size: int = 1000

class MistralRouter:
    """Routes requests between local Mistral model and API fallback"""
    
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self._initialize_model()
        self.session = None
        logger.info(f"Initialized MistralRouter with model: {self.config.model_name}")

    def _initialize_model(self) -> None:
        """Initialize the local Mistral model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map="auto"
            )
            self.model.eval()
            logger.info("Successfully loaded local Mistral model")
        except Exception as e:
            logger.error(f"Error initializing local model: {e}")
            self.model = None
            self.tokenizer = None

    @lru_cache(maxsize=1000)
    async def get_completion(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Get completion from model"""
        try:
            # Try local model first
            if self.model and self.tokenizer:
                return await self._local_completion(
                    prompt,
                    max_tokens or self.config.max_tokens,
                    temperature or self.config.temperature
                )
            
            # Fallback to API
            if self.config.fallback_url:
                return await self._api_completion(
                    prompt,
                    max_tokens or self.config.max_tokens,
                    temperature or self.config.temperature
                )
                
            raise ValueError("No available model or API fallback")
            
        except Exception as e:
            logger.error(f"Error getting completion: {e}")
            raise

    async def get_rating(self, prompt: str) -> float:
        """Get numerical rating from model"""
        try:
            # Add explicit rating instruction
            rating_prompt = f"""
            {prompt}
            
            Provide a single numerical rating between 0.0 and 1.0.
            Just return the number, no explanation or other text.
            """
            
            response = await self.get_completion(rating_prompt)
            
            # Extract and validate numerical rating
            try:
                rating = float(response.strip())
                if 0.0 <= rating <= 1.0:
                    return rating
                return 0.0
            except ValueError:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting rating: {e}")
            return 0.0

    async def get_structured_output(
        self,
        prompt: str,
        output_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get structured JSON output from model"""
        try:
            # Add schema instruction if provided
            if output_schema:
                schema_str = json.dumps(output_schema, indent=2)
                structured_prompt = f"""
                {prompt}
                
                Provide output in the following JSON schema:
                {schema_str}
                
                Return only the JSON, no other text.
                """
            else:
                structured_prompt = f"""
                {prompt}
                
                Provide output as a JSON object.
                Return only the JSON, no other text.
                """
            
            response = await self.get_completion(structured_prompt)
            
            # Parse and validate JSON
            try:
                output = json.loads(response)
                if output_schema:
                    # Basic schema validation
                    self._validate_schema(output, output_schema)
                return output
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON output from model")
                
        except Exception as e:
            logger.error(f"Error getting structured output: {e}")
            raise

    async def _local_completion(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate completion using local model"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_tokens
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Extract response after prompt
        return response[len(prompt):].strip()

    async def _api_completion(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate completion using API fallback"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with self.session.post(
            self.config.fallback_url,
            headers=headers,
            json=data
        ) as response:
            if response.status != 200:
                raise ValueError(f"API request failed: {response.status}")
                
            result = await response.json()
            return result["choices"][0]["text"].strip()

    def _validate_schema(
        self,
        output: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> None:
        """Basic JSON schema validation"""
        for key, value_type in schema.items():
            if key not in output:
                raise ValueError(f"Missing required key: {key}")
                
            if isinstance(value_type, dict):
                if not isinstance(output[key], dict):
                    raise ValueError(f"Invalid type for {key}")
                self._validate_schema(output[key], value_type)
            elif isinstance(value_type, list):
                if not isinstance(output[key], list):
                    raise ValueError(f"Invalid type for {key}")
            else:
                if not isinstance(output[key], value_type):
                    raise ValueError(f"Invalid type for {key}")

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        if self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("Cleaned up MistralRouter resources")

    def get_state(self) -> Dict[str, Any]:
        """Get current router state"""
        return {
            "model_name": self.config.model_name,
            "device": self.config.device,
            "has_local_model": self.model is not None,
            "has_api_fallback": self.config.fallback_url is not None,
            "cache_size": self.config.cache_size
        }