import re
from typing import List, Dict, Tuple, Set
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset

class TeluguPreprocessor:
    """Preprocessor for Telugu text data in English script (transliterated Telugu)."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.word2idx = {}
        self.idx2word = {}
        self.word_freqs = Counter()
        self.vocab_size = 0
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
    
    def fit(self, text: str):
        """
        Build vocabulary from text.
        
        Args:
            text (str): Input text to process (Telugu words in English script)
        """
        # Clean and tokenize text
        words = self._clean_text(text)
        
        # Count word frequencies
        self.word_freqs = Counter(words)
        
        # Create vocabulary
        vocab = [self.PAD_TOKEN, self.UNK_TOKEN]  # Special tokens first
        vocab.extend(word for word in self.word_freqs.keys())
        
        # Create mappings
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(vocab)
    
    def _clean_text(self, text: str) -> List[str]:
        """
        Clean and tokenize text.
        
        Args:
            text (str): Input text (Telugu words in English script)
            
        Returns:
            List[str]: List of cleaned tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Split into words
        words = text.split()
        
        # Remove punctuation and numbers
        words = [re.sub(r'[^\w\s]', '', word) for word in words]
        words = [re.sub(r'\d+', '', word) for word in words]
        
        # Remove empty strings
        words = [word for word in words if word]
        
        return words
    
    def prepare_cbow_training_data(self, window_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training data for CBOW model.
        
        Args:
            window_size (int): Size of context window
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Context words and target words
        """
        # Get all words
        words = []
        for word, freq in self.word_freqs.items():
            words.extend([word] * freq)
        
        # Create training pairs
        context_words = []
        target_words = []
        
        for i in range(window_size, len(words) - window_size):
            # Get context words
            context = (
                words[i - window_size:i] +  # Left context
                words[i + 1:i + window_size + 1]  # Right context
            )
            
            # Convert words to indices
            context_idx = [self.word2idx.get(w, self.word2idx[self.UNK_TOKEN]) for w in context]
            target_idx = self.word2idx.get(words[i], self.word2idx[self.UNK_TOKEN])
            
            context_words.append(context_idx)
            target_words.append(target_idx)
        
        if not context_words:
            raise ValueError(
                "Not enough words to create training data. "
                "Try reducing the window size or adding more text."
            )
        
        # Convert to tensors
        X = torch.tensor(context_words, dtype=torch.long)
        y = torch.tensor(target_words, dtype=torch.long)
        
        return X, y

class CBOWDataset(Dataset):
    def __init__(self, texts: List[str], preprocessor: TeluguPreprocessor, context_size: int = 2):
        """
        Initialize the CBOW dataset.
        
        Args:
            texts (List[str]): List of input texts
            preprocessor (TeluguPreprocessor): Text preprocessor
            context_size (int): Size of context window on each side
        """
        self.preprocessor = preprocessor
        self.context_size = context_size
        
        # Convert texts to indices
        self.data = []
        for text in texts:
            indices = self.preprocessor.text_to_indices(text)
            if len(indices) >= 2 * context_size + 1:
                self.data.extend(indices)
        
        self.data = np.array(self.data)

    def __len__(self) -> int:
        return len(self.data) - 2 * self.context_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training example.
        
        Args:
            idx (int): Index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (context_words, target_word)
        """
        # Get context window indices
        context_indices = []
        for i in range(-self.context_size, self.context_size + 1):
            if i != 0:  # Skip the target word
                context_indices.append(idx + i + self.context_size)
        
        # Get target word index
        target_idx = idx + self.context_size
        
        # Convert to tensors
        context = torch.tensor([self.data[i] for i in context_indices], dtype=torch.long)
        target = torch.tensor(self.data[target_idx], dtype=torch.long)
        
        return context, target 