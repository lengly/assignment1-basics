#!/usr/bin/env python3
"""
Training script for CS336 Transformer Language Model.

This script provides a complete training pipeline with:
- Configurable hyperparameters
- Memory-efficient data loading with np.memmap
- Checkpoint serialization
- Logging with Weights & Biases support
- Training and validation loops
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wandb

# Mixed precision training imports
from torch.amp import autocast
from torch.cuda.amp import GradScaler
# torch.set_float32_matmul_precision('high')

from .model import TransformerLM
from .tokenizer import Tokenizer
from .utils import (
    stable_cross_entropy,
    perplexity_from_loss,
    get_batch,
    save_checkpoint,
    load_checkpoint,
    AdamW,
    gradient_clipping,
    cosine_cycle_schedule
)


class MemmapDataset(Dataset):
    """Memory-efficient dataset using numpy memmap for large datasets."""
    
    def __init__(self, data_path: str, context_length: int):
        """
        Initialize dataset with memmap loading.
        
        Args:
            data_path: Path to the tokenized data file (.npy)
            context_length: Length of context window for language modeling
        """
        self.data_path = data_path
        self.context_length = context_length
        
        # Load data using memmap for memory efficiency
        self.data = np.load(data_path, mmap_mode='r')
        self.length = (len(self.data) - context_length) // context_length
        
        logging.info(f"Loaded dataset from {data_path}")
        logging.info(f"Dataset size: {len(self.data)} tokens")
        logging.info(f"Number of training sequences: {self.length}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Get input sequence and target sequence
        idx = idx * self.context_length
        input_seq = self.data[idx:idx + self.context_length]
        target_seq = self.data[idx + 1:idx + self.context_length + 1]
        
        # Copy the data to ensure it's writable before converting to tensor
        input_seq = input_seq.copy()
        target_seq = target_seq.copy()
        
        return torch.from_numpy(input_seq).long(), torch.from_numpy(target_seq).long()


class TrainingConfig:
    """Configuration class for training hyperparameters."""
    
    def __init__(self, **kwargs):
        # Model hyperparameters
        self.vocab_size = kwargs.get('vocab_size', 50257)
        self.context_length = kwargs.get('context_length', 1024)
        self.num_layers = kwargs.get('num_layers', 12)
        self.d_model = kwargs.get('d_model', 768)
        self.num_heads = kwargs.get('num_heads', 12)
        self.d_ff = kwargs.get('d_ff', 3072)
        self.rope_theta = kwargs.get('rope_theta', 10000.0)
        
        # Training hyperparameters
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 0.1)
        self.max_epochs = kwargs.get('max_epochs', 10)
        self.gradient_clip_norm = kwargs.get('gradient_clip_norm', 1.0)
        self.warmup_steps = kwargs.get('warmup_steps', 1000)
        self.total_steps = kwargs.get('total_steps', 100000)
        
        # Data paths
        self.train_data_path = kwargs.get('train_data_path', 'data/train_tokens.npy')
        self.val_data_path = kwargs.get('val_data_path', 'data/val_tokens.npy')
        self.tokenizer_vocab_path = kwargs.get('tokenizer_vocab_path', 'bpe_result/tiny/vocab.pkl')
        self.tokenizer_merges_path = kwargs.get('tokenizer_merges_path', 'bpe_result/tiny/merge.pkl')
        
        # Checkpoint and logging
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints')
        self.checkpoint_interval = kwargs.get('checkpoint_interval', 1000)
        self.log_interval = kwargs.get('log_interval', 100)
        self.eval_interval = kwargs.get('eval_interval', 1000)
        
        # Device and precision
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mixed_precision = kwargs.get('use_mixed_precision', False)
        self.mixed_precision_dtype = kwargs.get('mixed_precision_dtype', 'bfloat16')
        
        # Weights & Biases
        self.use_wandb = kwargs.get('use_wandb', False)
        self.wandb_project = kwargs.get('wandb_project', 'cs336-transformer')
        self.wandb_run_name = kwargs.get('wandb_run_name', None)
        
        # Resume training
        self.resume_from = kwargs.get('resume_from', None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def save(self, config_path: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class Trainer:
    """Main training class."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Initialize model, tokenizer, and datasets
        self._setup_model()
        self._setup_tokenizer()
        self._setup_datasets()
        self._setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Time tracking for speed calculation
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.total_tokens_processed = 0
        self.interval_tokens_processed = 0
        self.last_log_step = 0
        
        # Setup mixed precision training
        self._setup_mixed_precision()
        
        # Setup Weights & Biases
        if config.use_wandb:
            self._setup_wandb()
    
    def _setup_model(self):
        """Initialize the transformer model."""
        self.model = TransformerLM(
            vocab_size=self.config.vocab_size,
            context_length=self.config.context_length,
            num_layers=self.config.num_layers,
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            d_ff=self.config.d_ff,
            rope_theta=self.config.rope_theta,
            device=self.device,
            dtype=torch.float32,
            shared_lm_head=True
        )
        self.model = torch.compile(self.model)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logging.info(f"Model initialized with {total_params:,} total parameters")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    def _setup_tokenizer(self):
        """Initialize the tokenizer."""
        try:
            self.tokenizer = Tokenizer.from_files(
                self.config.tokenizer_vocab_path,
                self.config.tokenizer_merges_path
            )
            logging.info(f"Tokenizer loaded with vocab size: {self.tokenizer.vocab_size}")
        except Exception as e:
            logging.warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None
    
    def _setup_datasets(self):
        """Initialize training and validation datasets."""
        self.train_dataset = MemmapDataset(
            self.config.train_data_path,
            self.config.context_length
        )
        
        self.val_dataset = MemmapDataset(
            self.config.val_data_path,
            self.config.context_length
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def _setup_optimizer(self):
        """Initialize the optimizer."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        logging.info(f"Optimizer: AdamW with lr={self.config.learning_rate}, weight_decay={self.config.weight_decay}")
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        if self.config.use_mixed_precision and self.device.type == 'cuda':
            self.scaler = GradScaler()
            if self.config.mixed_precision_dtype == 'float16':
                self.autocast_dtype = torch.float16
            elif self.config.mixed_precision_dtype == 'bfloat16':
                self.autocast_dtype = torch.bfloat16
            else:
                logging.warning(f"Unknown mixed precision dtype: {self.config.mixed_precision_dtype}, using float16")
                self.autocast_dtype = torch.float16
            
            logging.info(f"Mixed precision training enabled with dtype: {self.config.mixed_precision_dtype}")
        else:
            self.scaler = None
            self.autocast_dtype = None
            if self.config.use_mixed_precision and self.device.type != 'cuda':
                logging.warning("Mixed precision training requires CUDA device, disabling")
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=self.config.to_dict(),
            resume="allow"
        )
        logging.info("Weights & Biases logging initialized")
    
    def _get_learning_rate(self, step: int) -> float:
        """Get learning rate for current step using cosine schedule."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * (step / self.config.warmup_steps)
        else:
            return cosine_cycle_schedule(
                step,
                self.config.warmup_steps,
                self.config.total_steps,
                lr_min=self.config.learning_rate * 0.1,
                lr_max=self.config.learning_rate
            )
    
    def _train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        inputs, targets = batch
        inputs = inputs.to(self.device, dtype=torch.long)
        targets = targets.to(self.device, dtype=torch.long)
        
        # Update token count for speed calculation
        self.total_tokens_processed += inputs.numel()
        self.interval_tokens_processed += inputs.numel()
        
        # Forward pass with mixed precision if enabled
        if self.scaler is not None:
            with autocast(device_type='cuda', dtype=self.autocast_dtype):
                logits = self.model(inputs)
                loss = stable_cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
        else:
            logits = self.model(inputs)
            loss = stable_cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        
        # Backward pass with gradient scaling if mixed precision is enabled
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            gradient_clipping(self.model.parameters(), self.config.gradient_clip_norm)
        else:
            gradient_clipping(self.model.parameters(), self.config.gradient_clip_norm)
        
        # Update learning rate
        lr = self._get_learning_rate(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Optimizer step with gradient scaling
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Calculate perplexity
        perplexity = perplexity_from_loss(loss)
        
        return {
            'loss': loss.item(),
            'perplexity': perplexity.item(),
            'learning_rate': lr
        }
    
    def _validate(self) -> Dict[str, float]:
        """Perform validation."""
        self.model.eval()
        total_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device, dtype=torch.long)
                targets = targets.to(self.device, dtype=torch.long)
                
                # Use autocast for validation if mixed precision is enabled
                if self.scaler is not None:
                    with autocast(device_type='cuda', dtype=self.autocast_dtype):
                        logits = self.model(inputs)
                        loss = stable_cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            targets.view(-1)
                        )
                else:
                    logits = self.model(inputs)
                    loss = stable_cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
                
                perplexity = perplexity_from_loss(loss)
                
                total_loss += loss.item()
                total_perplexity += perplexity.item()
                num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': avg_perplexity
        }
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_step_{self.global_step}.pt"
        )
        
        # Prepare checkpoint data
        checkpoint_data = {
            'iteration': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        # Add gradient scaler state if using mixed precision
        if self.scaler is not None:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint_data, best_path)
            print(f"Saved best model to {best_path}")
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config.checkpoint_dir, "latest.pt")
        torch.save(checkpoint_data, latest_path)
        print(f"Saved latest checkpoint to {latest_path}")
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to console and W&B."""
        current_time = time.time()
        
        # Calculate speed and time estimates
        elapsed_time = current_time - self.start_time
        interval_time = current_time - self.last_log_time
        tokens_per_second = self.interval_tokens_processed / interval_time if elapsed_time > 0 else 0
        
        # Calculate estimated remaining time
        if step > 0:
            steps_remaining = self.config.total_steps - step
            avg_time_per_step = elapsed_time / step
            estimated_remaining_time = steps_remaining * avg_time_per_step
        else:
            estimated_remaining_time = 0
        
        # Format time strings
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = seconds / 60
                return f"{minutes:.1f}m"
            else:
                hours = seconds / 3600
                return f"{hours:.1f}h"
        
        # Add speed and time metrics
        speed_metrics = {
            'tokens_per_sec': tokens_per_second,
            'elapsed_time': format_time(elapsed_time),
            'estimated_remaining': format_time(estimated_remaining_time)
        }
        
        # Console logging with speed info
        log_str = f"Step {step}: "
        log_str += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        log_str += f" | Speed: {tokens_per_second:.1f} tokens/sec"
        log_str += f" | Elapsed: {speed_metrics['elapsed_time']}"
        log_str += f" | ETA: {speed_metrics['estimated_remaining']}"
        logging.info(log_str)
        
        # Update last log time
        self.last_log_time = current_time
        self.interval_tokens_processed = 0
        self.last_log_step = step
        
        # W&B logging (include speed metrics)
        if self.config.use_wandb:
            wandb_metrics = {**metrics, 'tokens_per_sec': tokens_per_second}
            wandb.log(wandb_metrics, step=step)
    
    def train(self):
        """Main training loop."""
        logging.info("Starting training...")
        
        # Calculate expected training statistics
        data_steps = self.config.max_epochs * len(self.train_dataset) // self.config.batch_size
        total_tokens = self.config.total_steps * self.config.batch_size * self.config.context_length
        logging.info(f"Training configuration:")
        logging.info(f"  - Max Total steps: {self.config.total_steps:,}")
        logging.info(f"  - Data steps: {data_steps:,}")
        logging.info(f"  - Batch size: {self.config.batch_size}")
        logging.info(f"  - Context length: {self.config.context_length}")
        logging.info(f"  - Total tokens to process: {total_tokens:,}")
        logging.info(f"  - Log interval: {self.config.log_interval} steps")
        logging.info(f"  - Eval interval: {self.config.eval_interval} steps")
        logging.info(f"  - Checkpoint interval: {self.config.checkpoint_interval} steps")
        
        # Resume from checkpoint if specified
        if self.config.resume_from:
            checkpoint = torch.load(self.config.resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.global_step = checkpoint['iteration']
            
            # Load gradient scaler state if using mixed precision
            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logging.info("Loaded gradient scaler state from checkpoint")
            
            logging.info(f"Resumed training from step {self.global_step}")
        
        # Training loop
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            logging.info(f"Starting epoch {epoch + 1}/{self.config.max_epochs}")
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                metrics = self._train_step(batch)
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    self._log_metrics(metrics, self.global_step)
                
                # Validation
                if self.global_step % self.config.eval_interval == 0:
                    val_metrics = self._validate()
                    self._log_metrics(val_metrics, self.global_step)
                    
                    # Save best model
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self._save_checkpoint(is_best=True)
                        logging.info(f"New best validation loss: {self.best_val_loss:.4f}")
                
                # Checkpointing
                if self.global_step % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()
                
                # Check if we've reached total steps
                if self.global_step >= self.config.total_steps:
                    logging.info(f"Reached total steps ({self.config.total_steps}), stopping training")
                    break
            
            if self.global_step >= self.config.total_steps:
                break
        
        # Final validation and checkpoint
        logging.info("Training completed. Running final validation...")
        final_metrics = self._validate()
        self._log_metrics(final_metrics, self.global_step)
        self._save_checkpoint()
        
        # Final training statistics
        total_time = time.time() - self.start_time
        final_tokens_per_sec = self.total_tokens_processed / total_time if total_time > 0 else 0
        logging.info("Training finished!")
        logging.info(f"Final statistics:")
        logging.info(f"  - Total training time: {total_time/3600:.2f} hours")
        logging.info(f"  - Total tokens processed: {self.total_tokens_processed:,}")
        logging.info(f"  - Average speed: {final_tokens_per_sec:.1f} tokens/sec")
        logging.info(f"  - Final validation loss: {final_metrics['val_loss']:.4f}")
        logging.info(f"  - Final validation perplexity: {final_metrics['val_perplexity']:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train CS336 Transformer Language Model")
    
    # Configuration options
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--train-data", type=str, help="Path to training data (.npy)")
    parser.add_argument("--val-data", type=str, help="Path to validation data (.npy)")
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=1024, help="Context length")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--total-steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume-from", type=str, help="Resume from checkpoint")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="cs336-transformer", help="W&B project name")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--use-mixed-precision", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--mixed-precision-dtype", type=str, default="bfloat16", 
                       choices=["float16", "bfloat16"], help="Mixed precision dtype")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = TrainingConfig.from_file(args.config)
    else:
        # Create config from command line arguments
        config_dict = {
            'vocab_size': args.vocab_size,
            'context_length': args.context_length,
            'num_layers': args.num_layers,
            'd_model': args.d_model,
            'num_heads': args.num_heads,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_epochs': args.max_epochs,
            'total_steps': args.total_steps,
            'checkpoint_dir': args.checkpoint_dir,
            'resume_from': args.resume_from,
            'use_wandb': args.use_wandb,
            'wandb_project': args.wandb_project,
            'use_mixed_precision': args.use_mixed_precision,
            'mixed_precision_dtype': args.mixed_precision_dtype,
        }
        
        if args.train_data:
            config_dict['train_data_path'] = args.train_data
        if args.val_data:
            config_dict['val_data_path'] = args.val_data
        if args.device != "auto":
            config_dict['device'] = args.device
        
        config = TrainingConfig(**config_dict)
    
    # Save configuration
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    config.save(os.path.join(config.checkpoint_dir, "config.json"))
    
    # Start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
