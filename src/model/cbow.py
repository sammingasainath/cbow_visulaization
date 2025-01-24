import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Callable, Any
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class CBOWModel(nn.Module):
    """Continuous Bag of Words (CBOW) model for word embeddings."""
    
    def __init__(self, vocab_size: int, embedding_size: int = 100):
        """
        Initialize the CBOW model.
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_size (int): Size of word embeddings
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        
        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        
        # Linear layer for prediction
        self.linear = nn.Linear(embedding_size, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, context_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, vocab_size)
        """
        # Get embeddings for all context words
        embeds = self.embeddings(x)  # Shape: (batch_size, context_size, embedding_size)
        
        # Average the embeddings
        hidden = torch.mean(embeds, dim=1)  # Shape: (batch_size, embedding_size)
        
        # Predict target word
        out = self.linear(hidden)  # Shape: (batch_size, vocab_size)
        
        return out
    
    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        learning_rate: float = 0.001,
        epochs: int = 10,
        progress_callback: Optional[Callable[[int, float], None]] = None
    ):
        """
        Train the model.
        
        Args:
            X (torch.Tensor): Context word indices
            y (torch.Tensor): Target word indices
            learning_rate (float): Learning rate
            epochs (int): Number of training epochs
            progress_callback (Callable): Callback for progress updates
        """
        # Create optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_x, batch_y in progress_bar:
                # Forward pass
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            
            # Calculate average loss
            avg_loss = total_loss / len(dataloader)
            
            # Update progress
            if progress_callback:
                progress_callback(epoch, avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def get_word_vector(self, word_idx: int) -> torch.Tensor:
        """
        Get the embedding vector for a word.
        
        Args:
            word_idx (int): Index of the word
            
        Returns:
            torch.Tensor: Word embedding vector
        """
        return self.embeddings.weight[word_idx].detach()
    
    def get_similar_words(
        self,
        word_idx: int,
        n: int = 5,
        preprocessor: Optional[Any] = None
    ) -> List[Tuple[str, float]]:
        """
        Find similar words based on cosine similarity.
        
        Args:
            word_idx (int): Index of the query word
            n (int): Number of similar words to return
            preprocessor (Any): Preprocessor for word-index mapping
            
        Returns:
            List[Tuple[str, float]]: List of (word, similarity) pairs
        """
        # Get query vector
        query_vec = self.get_word_vector(word_idx)
        
        # Calculate cosine similarity with all words
        cos = nn.CosineSimilarity(dim=0)
        similarities = []
        
        for i in range(self.vocab_size):
            if i == word_idx:
                continue
            
            vec = self.get_word_vector(i)
            sim = cos(query_vec, vec).item()
            
            if preprocessor:
                word = preprocessor.idx2word[i]
            else:
                word = str(i)
            
            similarities.append((word, sim))
        
        # Sort by similarity and return top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

class NegativeSamplingLoss(nn.Module):
    def __init__(self, n_samples: int = 5):
        """
        Initialize the Negative Sampling loss function.
        
        Args:
            n_samples (int): Number of negative samples per positive sample
        """
        super(NegativeSamplingLoss, self).__init__()
        self.n_samples = n_samples
        self.sampling_weights = None

    def forward(self, input_vectors: torch.Tensor, output_vectors: torch.Tensor,
                target_indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """
        Compute the negative sampling loss.
        
        Args:
            input_vectors: Context embeddings [batch_size, embedding_dim]
            output_vectors: Output layer weights [vocab_size, embedding_dim]
            target_indices: Target word indices [batch_size]
            vocab_size: Size of vocabulary
            
        Returns:
            torch.Tensor: Computed loss
        """
        batch_size = input_vectors.size(0)
        
        # Generate negative samples
        if self.sampling_weights is None:
            self.sampling_weights = torch.ones(vocab_size)
        
        negative_samples = torch.multinomial(
            self.sampling_weights,
            batch_size * self.n_samples,
            replacement=True
        ).view(batch_size, self.n_samples)
        
        # Positive samples loss
        positive_loss = torch.sum(
            F.logsigmoid(torch.sum(
                input_vectors * output_vectors[target_indices], dim=1
            ))
        )
        
        # Negative samples loss
        negative_vectors = output_vectors[negative_samples]
        negative_loss = torch.sum(
            F.logsigmoid(-torch.bmm(
                negative_vectors,
                input_vectors.unsqueeze(2)
            ).squeeze())
        )
        
        return -(positive_loss + negative_loss) / batch_size 