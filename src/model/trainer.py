import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Callable
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.data.preprocessor import CBOWDataset, TeluguPreprocessor
from src.model.cbow import CBOWModel, NegativeSamplingLoss

class CBOWTrainer:
    def __init__(
        self,
        model: CBOWModel,
        preprocessor: TeluguPreprocessor,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the CBOW trainer.
        
        Args:
            model (CBOWModel): The CBOW model
            preprocessor (TeluguPreprocessor): Text preprocessor
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            n_epochs (int): Number of epochs
            device (str): Device to use for training
        """
        self.model = model.to(device)
        self.preprocessor = preprocessor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = NegativeSamplingLoss()
        
        self.train_losses: List[float] = []
        self.current_epoch = 0

    def train(
        self,
        dataset: CBOWDataset,
        progress_callback: Callable[[int, float], None] = None
    ) -> List[float]:
        """
        Train the CBOW model.
        
        Args:
            dataset (CBOWDataset): Training dataset
            progress_callback (Callable): Callback for updating progress
            
        Returns:
            List[float]: Training losses
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.model.train()
        epoch_losses = []
        
        for epoch in range(self.n_epochs):
            self.current_epoch = epoch
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.n_epochs}')
            
            for batch_idx, (contexts, targets) in enumerate(progress_bar):
                # Move data to device
                contexts = contexts.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(contexts)
                
                # Calculate loss
                loss = self.criterion(
                    self.model.embeddings(contexts).mean(dim=1),
                    self.model.embeddings.weight,
                    targets,
                    self.preprocessor.vocab_size
                )
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update progress
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                if progress_callback:
                    progress = (epoch * len(dataloader) + batch_idx) / (self.n_epochs * len(dataloader))
                    progress_callback(progress, avg_loss)
            
            epoch_loss = total_loss / len(dataloader)
            epoch_losses.append(epoch_loss)
            self.train_losses.extend([epoch_loss] * len(dataloader))
            
            print(f'Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.4f}')
        
        return epoch_losses

    def plot_training_progress(self) -> plt.Figure:
        """
        Plot training progress.
        
        Returns:
            plt.Figure: Training progress plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.train_losses)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.grid(True)
        return fig

    def get_word_vector(self, word: str) -> torch.Tensor:
        """
        Get the embedding vector for a word.
        
        Args:
            word (str): Input word
            
        Returns:
            torch.Tensor: Word embedding vector
        """
        if word in self.preprocessor.word2idx:
            idx = self.preprocessor.word2idx[word]
            return self.model.get_word_embedding(idx)
        return None

    def find_similar_words(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar words using cosine similarity.
        
        Args:
            word (str): Input word
            top_k (int): Number of similar words to return
            
        Returns:
            List[Tuple[str, float]]: List of (word, similarity_score) pairs
        """
        if word not in self.preprocessor.word2idx:
            return []
        
        word_idx = self.preprocessor.word2idx[word]
        similar_indices = self.model.get_most_similar(word_idx, top_k)
        
        return [
            (self.preprocessor.idx2word[idx], score)
            for idx, score in similar_indices
        ]

    def save_model(self, path: str):
        """Save the model and preprocessor state."""
        state = {
            'model_state': self.model.state_dict(),
            'preprocessor': self.preprocessor,
            'optimizer_state': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'current_epoch': self.current_epoch
        }
        torch.save(state, path)

    def load_model(self, path: str):
        """Load the model and preprocessor state."""
        state = torch.load(path)
        self.model.load_state_dict(state['model_state'])
        self.preprocessor = state['preprocessor']
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.train_losses = state['train_losses']
        self.current_epoch = state['current_epoch'] 
        self.current_epoch = state['current_epoch'] 