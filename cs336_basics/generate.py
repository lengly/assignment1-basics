#!/usr/bin/env python3
"""
Text generation module for CS336 Transformer Language Model.

This module provides functionality to generate text completions from a trained
language model with various sampling strategies including temperature scaling
and top-p (nucleus) sampling.
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Union, Dict, Any
import warnings

import torch
import torch.nn.functional as F
import numpy as np

from .model import TransformerLM
from .tokenizer import Tokenizer
from .utils import load_checkpoint


class TextGenerator:
    """Text generation class with various sampling strategies."""
    
    def __init__(self, model: TransformerLM, tokenizer: Tokenizer, device: str = "auto"):
        """
        Initialize the text generator.
        
        Args:
            model: The trained TransformerLM model
            tokenizer: The tokenizer for encoding/decoding text
            device: Device to run generation on ("auto", "cuda", "cpu")
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Get special tokens
        self.endoftext_token = self._find_endoftext_token()
        
        logging.info(f"Text generator initialized on device: {self.device}")
        logging.info(f"End-of-text token ID: {self.endoftext_token}")
    
    def _find_endoftext_token(self) -> Optional[int]:
        """Find the end-of-text token ID in the vocabulary."""
        # Common end-of-text token variations
        endoftext_variations = [
            "<|endoftext|>",
            "</s>",
            "<eos>",
            "<end>",
            "EOS",
            "END"
        ]
        
        for token in endoftext_variations:
            try:
                # Try to encode the token
                token_ids = self.tokenizer.encode(token)
                if len(token_ids) == 1:
                    return token_ids[0]
            except:
                continue
        
        # If no special token found, return None
        logging.warning("No end-of-text token found in vocabulary")
        return None
    
    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw logits from the model
            temperature: Temperature value (0.0 < temperature <= 2.0)
            
        Returns:
            Temperature-scaled logits
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        if temperature == 1.0:
            return logits
        
        return logits / temperature
    
    def _apply_top_p_sampling(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Apply top-p (nucleus) sampling to logits.
        
        Args:
            logits: Logits from the model
            top_p: Cumulative probability threshold (0.0 < top_p <= 1.0)
            
        Returns:
            Filtered logits with top-p sampling applied
        """
        if top_p <= 0 or top_p > 1:
            raise ValueError("top_p must be between 0 and 1")
        
        if top_p == 1.0:
            return logits
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Find the cutoff index
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Create a mask for the original logits
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        
        # Apply the mask by setting filtered logits to -inf
        filtered_logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return filtered_logits
    
    def _sample_next_token(self, logits: torch.Tensor, temperature: float = 1.0, 
                          top_p: float = 1.0, top_k: Optional[int] = None) -> int:
        """
        Sample the next token using the specified sampling strategy.
        
        Args:
            logits: Raw logits from the model
            temperature: Temperature for sampling (default: 1.0)
            top_p: Top-p sampling threshold (default: 1.0)
            top_k: Top-k sampling threshold (optional)
            
        Returns:
            Sampled token ID
        """
        # Apply temperature scaling
        if temperature != 1.0:
            logits = self._apply_temperature(logits, temperature)
        
        # Apply top-k filtering (if specified)
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Apply top-p sampling
        if top_p < 1.0:
            logits = self._apply_top_p_sampling(logits, top_p)
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token.item()
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 1.0,
                top_p: float = 1.0, top_k: Optional[int] = None, 
                stop_at_eot: bool = True, seed: Optional[int] = None) -> str:
        """
        Generate text completion for a given prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (0.0 < temperature <= 2.0)
            top_p: Top-p sampling threshold (0.0 < top_p <= 1.0)
            top_k: Top-k sampling threshold (optional)
            stop_at_eot: Whether to stop at end-of-text token
            seed: Random seed for reproducibility
            
        Returns:
            Generated text completion
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Encode the prompt
        try:
            input_ids = self.tokenizer.encode(prompt)
        except Exception as e:
            logging.error(f"Error encoding prompt: {e}")
            return prompt
        
        if len(input_ids) == 0:
            logging.warning("Empty prompt after tokenization")
            return prompt
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Generate tokens
        generated_tokens = []
        context_length = self.model.token_embeddings.weight.size(1)  # Get from model
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Get model predictions
                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :]  # Get logits for the last position
                
                # Sample next token
                next_token = self._sample_next_token(
                    next_token_logits, temperature, top_p, top_k
                )
                
                # Check for end-of-text token
                if stop_at_eot and self.endoftext_token is not None and next_token == self.endoftext_token:
                    break
                
                # Add token to generated sequence
                generated_tokens.append(next_token)
                
                # Update input tensor for next iteration
                input_tensor = torch.cat([
                    input_tensor, 
                    torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                ], dim=1)
                
                # Truncate context if it gets too long
                if input_tensor.size(1) > context_length:
                    input_tensor = input_tensor[:, -context_length:]
        
        # Decode the generated tokens
        try:
            generated_text = self.tokenizer.decode(generated_tokens)
        except Exception as e:
            logging.error(f"Error decoding generated tokens: {e}")
            generated_text = ""
        
        return generated_text
    
    def generate_multiple(self, prompt: str, num_samples: int = 3, **kwargs) -> List[str]:
        """
        Generate multiple completions for the same prompt.
        
        Args:
            prompt: Input text prompt
            num_samples: Number of samples to generate
            **kwargs: Additional arguments for generate()
            
        Returns:
            List of generated completions
        """
        completions = []
        for i in range(num_samples):
            # Use different seed for each sample
            seed = kwargs.pop('seed', None)
            if seed is not None:
                kwargs['seed'] = seed + i
            
            completion = self.generate(prompt, **kwargs)
            completions.append(completion)
        
        return completions


def load_generator_from_checkpoint(checkpoint_path: str, tokenizer_vocab_path: str, 
                                 tokenizer_merges_path: str, config_path: Optional[str] = None,
                                 device: str = "auto") -> TextGenerator:
    """
    Load a text generator from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        tokenizer_vocab_path: Path to the tokenizer vocabulary file
        tokenizer_merges_path: Path to the tokenizer merges file
        config_path: Path to the configuration JSON file (optional)
        device: Device to load the model on
        
    Returns:
        Initialized TextGenerator instance
    """
    # Load tokenizer
    tokenizer = Tokenizer.from_files(tokenizer_vocab_path, tokenizer_merges_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model configuration
    if config_path:
        # Load configuration from file
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract model parameters from config
        vocab_size = config.get('vocab_size', 50257)
        context_length = config.get('context_length', 1024)
        num_layers = config.get('num_layers', 12)
        d_model = config.get('d_model', 768)
        num_heads = config.get('num_heads', 12)
        d_ff = config.get('d_ff', 3072)
        rope_theta = config.get('rope_theta', 10000.0)
        
        logging.info(f"Loaded model configuration from: {config_path}")
    else:
        # Fallback to inferring from checkpoint (original behavior)
        model_state = checkpoint['model_state_dict']
        
        # Infer model configuration from state dict
        vocab_size = model_state['token_embeddings.weight'].size(0)
        d_model = model_state['token_embeddings.weight'].size(1)
        
        # Count layers
        num_layers = 0
        for key in model_state.keys():
            if key.startswith('layers.') and key.endswith('.attn.q_proj.weight'):
                num_layers += 1
        
        # Get other parameters (use defaults if not available)
        context_length = 1024  # Default
        num_heads = 12  # Default
        d_ff = 3072  # Default
        rope_theta = 10000.0  # Default
        
        logging.info("No config file provided, inferring model configuration from checkpoint")
    
    # Create model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create generator
    generator = TextGenerator(model, tokenizer, device)
    
    logging.info(f"Loaded generator from checkpoint: {checkpoint_path}")
    logging.info(f"Model configuration: vocab_size={vocab_size}, d_model={d_model}, num_layers={num_layers}")
    logging.info(f"Model configuration: context_length={context_length}, num_heads={num_heads}, d_ff={d_ff}")
    
    return generator


def main():
    """Command-line interface for text generation."""
    parser = argparse.ArgumentParser(description="Generate text with CS336 Transformer")
    
    # Model and tokenizer paths
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab-path", type=str, required=True, help="Path to tokenizer vocabulary")
    parser.add_argument("--merges-path", type=str, required=True, help="Path to tokenizer merges")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling threshold")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling threshold")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-stop-eot", action="store_true", help="Don't stop at end-of-text token")
    
    # Output options
    parser.add_argument("--output-file", type=str, help="Output file for generated text")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load generator
        generator = load_generator_from_checkpoint(
            args.checkpoint, args.vocab_path, args.merges_path, args.config, args.device
        )
        
        # Generate text
        if args.num_samples == 1:
            # Single generation
            completion = generator.generate(
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                stop_at_eot=not args.no_stop_eot,
                seed=args.seed
            )
            
            full_text = args.prompt + completion
            print(f"Generated text:\n{full_text}")
            
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(full_text)
        else:
            # Multiple generations
            completions = generator.generate_multiple(
                prompt=args.prompt,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                stop_at_eot=not args.no_stop_eot,
                seed=args.seed
            )
            
            print(f"Generated {args.num_samples} completions:")
            for i, completion in enumerate(completions, 1):
                full_text = args.prompt + completion
                print(f"\n--- Sample {i} ---")
                print(full_text)
            
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for i, completion in enumerate(completions, 1):
                        full_text = args.prompt + completion
                        f.write(f"--- Sample {i} ---\n")
                        f.write(full_text)
                        f.write("\n\n")
    
    except Exception as e:
        logging.error(f"Error during generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 