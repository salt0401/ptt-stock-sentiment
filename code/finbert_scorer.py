"""FinBERT contextual sentiment scorer.

Uses a pre-trained Financial BERT model for sequence classification to score
raw sentence strings contextually, replacing the static Word2Vec dictionary approach.

Scores are mapped from classification logits to a continuous [-1, 1] range.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR


class FinBERTScorer:
    """Contextual sentiment scorer using a pre-trained FinBERT model."""
    
    def __init__(self, model_name='yiyanghkust/finbert-tone', device=None, max_length=128):
        """Initialize the tokenizer and model.
        
        Using yiyanghkust/finbert-tone as the default as it's a widely used 
        financial sentiment BERT. Its labels are:
        0: Neutral, 1: Positive, 2: Negative.
        
        Parameters
        ----------
        model_name : str
            HuggingFace model ID.
        device : str or torch.device, optional
            'cuda', 'mps' (for Apple Silicon), or 'cpu'. Auto-detected if None.
        max_length : int
            Max token length for BERT truncation.
        """
        self.model_name = model_name
        self.max_length = max_length
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def _logits_to_score(self, logits):
        """Convert classification logits to a continuous [-1, 1] score.
        
        For yiyanghkust/finbert-tone:
        0 = Neutral, 1 = Positive, 2 = Negative
        
        Score = P(Positive) - P(Negative)
        """
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # P(Positive)[idx 1] - P(Negative)[idx 2]
        scores = probs[:, 1] - probs[:, 2]
        return scores
        
    def score_sentences(self, sentences, batch_size=64, verbose=True):
        """Score a list of raw text sentences.
        
        Parameters
        ----------
        sentences : list[str]
            Raw text strings to score.
        batch_size : int
            Batch size for inference.
        verbose : bool
            Whether to show a progress bar.
            
        Returns
        -------
        np.ndarray : shape (len(sentences),) continuous sentiment scores in [-1, 1].
        """
        if not sentences:
            return np.array([])
            
        all_scores = []
        n_batches = (len(sentences) + batch_size - 1) // batch_size
        
        iterator = range(n_batches)
        if verbose:
            iterator = tqdm(iterator, desc="FinBERT Scoring", total=n_batches)
            
        with torch.no_grad():
            for b in iterator:
                start = b * batch_size
                end = min(start + batch_size, len(sentences))
                batch_text = sentences[start:end]
                
                # Ensure all are strings
                batch_text = [str(t) if t is not None else "" for t in batch_text]
                
                encoded = self.tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                outputs = self.model(**encoded)
                scores = self._logits_to_score(outputs.logits)
                all_scores.extend(scores.tolist())
                
        return np.array(all_scores, dtype=np.float64)

