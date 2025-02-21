# risk_rag_system/llm/mixtral_handler.py

from typing import Dict, Any, Optional, List, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
import json
import asyncio
from pydantic import BaseModel
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class MixtralConfig(BaseModel):
    """Configuration for Mixtral model"""
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    load_in_4bit: bool = True  # Use 4-bit quantization
    use_flash_attention: bool = True
    batch_size: int = 1
    max_batch_tokens: int = 4096

class MixtralHandler:
    """Handler for Mixtral model inference"""
    
    def __init__(self, config: Optional[MixtralConfig] = None):
        self.config = config or MixtralConfig()
        self._initialize_model()
        self.thread_pool = ThreadPoolExecutor(max_workers=2)  # For CPU-bound tasks
        logger.info(f"Initialized MixtralHandler with model: {self.config.model_name}")

    def _initialize_model(self) -> None:
        """Initialize the model and tokenizer"""
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                padding_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize model with optimizations
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16 if self.config.device == "cuda" else torch.float32,
            }
            
            if self.config.load_in_4bit:
                model_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                })
            
            if self.config.use_flash_attention and self.config.device == "cuda":
                model_kwargs["use_flash_attention_2"] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Enable model evaluation mode
            self.model.eval()
            
            logger.info("Successfully loaded Mixtral model and tokenizer")
            
        except Exception as e:
            logger.error(f"Error initializing Mixtral model: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate text completion"""
        try:
            # Prepare inputs
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_tokens
            ).to(self.config.device)
            
            # Prepare generation config
            gen_kwargs = {
                "max_new_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
                "repetition_penalty": self.config.repetition_penalty,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            if stop_sequences:
                gen_kwargs["stopping_criteria"] = self._create_stopping_criteria(
                    stop_sequences
                )
            
            # Run generation in thread pool
            outputs = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self._generate,
                inputs,
                gen_kwargs
            )
            
            # Decode and clean response
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Remove prompt from response
            response = response[len(prompt):].strip()
            
            # Apply stop sequences if any
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in response:
                        response = response[:response.index(stop_seq)]
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise

    def _generate(
        self,
        inputs: Dict[str, torch.Tensor],
        gen_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """Run model generation"""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        return outputs

    async def get_embeddings(
        self,
        texts: List[str],
        pooling: str = "mean"
    ) -> torch.Tensor:
        """Get text embeddings"""
        try:
            embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_tokens
                ).to(self.config.device)
                
                # Get model outputs
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get hidden states from last layer
                hidden_states = outputs.hidden_states[-1]
                
                # Apply pooling
                if pooling == "mean":
                    # Mean pooling
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    batch_embeddings = torch.sum(
                        hidden_states * attention_mask, dim=1
                    ) / torch.sum(attention_mask, dim=1)
                elif pooling == "cls":
                    # CLS token pooling
                    batch_embeddings = hidden_states[:, 0]
                else:
                    raise ValueError(f"Unknown pooling type: {pooling}")
                
                embeddings.append(batch_embeddings.cpu())
            
            # Concatenate all batches
            all_embeddings = torch.cat(embeddings, dim=0)
            
            # Normalize embeddings
            return torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def get_logprobs(
        self,
        text: str,
        topk: int = 5
    ) -> Dict[str, List[Dict[str, float]]]:
        """Get token logprobs for text"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_tokens
            ).to(self.config.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits[0]  # Remove batch dimension
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get top-k tokens and probabilities for each position
            topk_probs, topk_indices = torch.topk(probs, k=topk, dim=-1)
            
            # Convert to list of dicts
            result = []
            for pos in range(len(text)):
                pos_probs = {}
                for i in range(topk):
                    token = self.tokenizer.decode([topk_indices[pos, i]])
                    prob = float(topk_probs[pos, i])
                    pos_probs[token] = prob
                result.append(pos_probs)
            
            return {"token_logprobs": result}
            
        except Exception as e:
            logger.error(f"Error getting logprobs: {e}")
            raise

    def _create_stopping_criteria(self, stop_sequences: List[str]) -> Any:
        """Create stopping criteria for generation"""
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class StopOnSequences(StoppingCriteria):
            def __init__(self, stops: List[List[int]], tokenizer):
                super().__init__()
                self.stops = stops
                self.tokenizer = tokenizer
            
            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
                **kwargs
            ) -> bool:
                for stop in self.stops:
                    if input_ids[0][-len(stop):].tolist() == stop:
                        return True
                return False
        
        # Convert stop sequences to token ids
        stop_ids = [
            self.tokenizer.encode(seq, add_special_tokens=False)
            for seq in stop_sequences
        ]
        
        return StoppingCriteriaList([StopOnSequences(stop_ids, self.tokenizer)])

    async def cleanup(self) -> None:
        """Cleanup resources"""
        self.thread_pool.shutdown()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_state(self) -> Dict[str, Any]:
        """Get current handler state"""
        return {
            "model_name": self.config.model_name,
            "device": self.config.device,
            "max_tokens": self.config.max_tokens,
            "loaded_in_4bit": self.config.load_in_4bit,
            "using_flash_attention": self.config.use_flash_attention
        }