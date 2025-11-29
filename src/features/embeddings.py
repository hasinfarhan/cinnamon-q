"""
Embedding Generator - FinBERT for Semantic Grounding

Research Track Sprint 1: Semantic Grounding with Embeddings

This module generates dense vector embeddings (768-dim) from news headlines
using FinBERT (ProsusAI/finbert), a finance-specialized BERT model.

The embeddings capture semantic meaning beyond scalar scores, allowing the
RL agent to understand nuanced differences (e.g., "Partnership" vs "Regulatory Win").
"""

import logging
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class FinBERTEmbedder:
    """
    FinBERT-based embedding generator for financial news headlines.
    
    Uses ProsusAI/finbert, a BERT model fine-tuned on financial news,
    to generate 768-dimensional dense vector embeddings.
    
    Features:
    - Batch processing for efficiency
    - GPU acceleration (if available)
    - Mean pooling for sentence-level embeddings
    - Normalization for stable training
    
    Example:
        embedder = FinBERTEmbedder()
        headlines = ["Bitcoin reaches new all-time high", "Ethereum upgrade delayed"]
        embeddings = embedder.generate_embeddings(headlines)
        # Returns: np.array of shape (2, 768)
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize the FinBERT embedder.
        
        Args:
            model_name: HuggingFace model identifier (default: "ProsusAI/finbert")
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for tokenization (default: 512)
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        logger.info(f"🔤 Loading FinBERT model: {model_name}")
        logger.info(f"   Device: {self.device}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode (no dropout, etc.)
            
            logger.info(f"✓ FinBERT model loaded successfully")
            logger.info(f"   Embedding dimension: {self.model.config.hidden_size}")
        except Exception as e:
            logger.error(f"✗ Failed to load FinBERT model: {e}")
            raise
    
    def generate_embeddings(
        self,
        text_list: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.
        
        This method processes text in batches for efficiency and uses mean pooling
        to create sentence-level embeddings from token-level outputs.
        
        Args:
            text_list: List of text strings (headlines, news articles, etc.)
            batch_size: Number of texts to process per batch (default: 32)
            normalize: Whether to L2-normalize embeddings (default: True)
        
        Returns:
            np.ndarray: Embeddings array of shape (N, 768) where N = len(text_list)
        
        Note:
            - Empty strings are replaced with a placeholder ("No news available")
            - Mean pooling is used to aggregate token embeddings into sentence embeddings
            - Embeddings are normalized by default for stable training
        """
        if not text_list:
            # Return zero vector if no text provided
            return np.zeros((0, 768), dtype=np.float32)
        
        # Replace empty strings with placeholder
        processed_texts = [
            text if text and text.strip() else "No news available"
            for text in text_list
        ]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate embeddings (no gradient computation for inference)
            with torch.no_grad():
                outputs = self.model(**encoded)
                # outputs.last_hidden_state shape: (batch_size, seq_length, hidden_size)
                token_embeddings = outputs.last_hidden_state
                
                # Mean pooling: Average over sequence length (excluding padding)
                # attention_mask: 1 for real tokens, 0 for padding
                attention_mask = encoded['attention_mask']
                # Expand attention_mask to match embedding dimensions
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                # Sum embeddings, weighted by attention mask
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                # Sum of mask values (number of real tokens per sentence)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                # Mean pooling
                batch_embeddings = sum_embeddings / sum_mask
            
            # Convert to numpy and normalize if requested
            batch_embeddings_np = batch_embeddings.cpu().numpy().astype(np.float32)
            
            if normalize:
                # L2 normalization
                norms = np.linalg.norm(batch_embeddings_np, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
                batch_embeddings_np = batch_embeddings_np / norms
            
            all_embeddings.append(batch_embeddings_np)
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        
        logger.debug(f"✓ Generated {len(embeddings)} embeddings (shape: {embeddings.shape})")
        
        return embeddings
    
    def generate_single_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Convenience method for single-text processing.
        
        Args:
            text: Single text string
            normalize: Whether to L2-normalize embedding (default: True)
        
        Returns:
            np.ndarray: Embedding array of shape (768,)
        """
        embeddings = self.generate_embeddings([text], batch_size=1, normalize=normalize)
        return embeddings[0] if len(embeddings) > 0 else np.zeros(768, dtype=np.float32)

