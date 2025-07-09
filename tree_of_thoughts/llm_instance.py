"""
LLM Instance manager for Tree of Thoughts framework
"""

from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMInstance:
    """Shared LLM instance for both generation and evaluation"""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_length: int = 2048
    ):
        """
        Initialize the LLM instance
        
        Args:
            model_name: Hugging Face model identifier
            device: Device placement ("auto", "cuda", "cpu")
            torch_dtype: Data type for model weights
            load_in_8bit: Whether to use 8-bit quantization
            load_in_4bit: Whether to use 4-bit quantization
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side="left"
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure model loading parameters
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device,
            "low_cpu_mem_usage": True
        }
        
        # Add quantization if requested
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch_dtype
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Create pipeline for easy text generation
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=max_length,
            device_map=device
        )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text completions
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0=deterministic, 1=random)
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or take argmax
            num_return_sequences: Number of completions to generate
            
        Returns:
            List of generated text completions
        """
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract only the generated text
        results = []
        for output in outputs:
            generated_text = output['generated_text']
            # Remove the prompt from the beginning
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            results.append(generated_text)
        
        return results