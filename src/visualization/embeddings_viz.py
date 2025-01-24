import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Optional
import torch
import seaborn as sns
from src.model.cbow import CBOWModel
from src.data.preprocessor import TeluguPreprocessor
import random

class EmbeddingsVisualizer:
    """Visualizer for word embeddings with animated scatter and cluster effects."""
    
    def __init__(
        self,
        model: CBOWModel,
        preprocessor: TeluguPreprocessor,
        method: str = 'tsne',
        animation_speed: str = 'normal',
        n_components: int = 2
    ):
        """
        Initialize the visualizer.
        
        Args:
            model (CBOWModel): Trained CBOW model
            preprocessor (TeluguPreprocessor): Text preprocessor
            method (str): Visualization method ('tsne' or 'pca')
            animation_speed (str): Animation speed ('slow', 'normal', 'fast')
            n_components (int): Number of dimensions for visualization
        """
        self.model = model
        self.preprocessor = preprocessor
        self.method = method.lower()
        self.n_components = n_components
        self.animation_speed = animation_speed
        self.speed_settings = {'slow': 2, 'normal': 5, 'fast': 10}
        self.words = []
        self.reduced_embeddings = None
        self.animation_frames = []
        
    def _reduce_dimensions(self, embeddings: np.ndarray, progress_callback: Optional[callable] = None) -> np.ndarray:
        """
        Reduce embedding dimensions with progress updates.
        
        Args:
            embeddings (np.ndarray): Word embeddings
            progress_callback (callable, optional): Progress callback function
            
        Returns:
            np.ndarray: Reduced embeddings
        """
        if progress_callback:
            progress_callback(0.4, f"Reducing dimensions using {self.method.upper()}...")
        
        if self.method == 'tsne':
            reducer = TSNE(
                n_components=self.n_components,
                perplexity=min(30, len(embeddings) - 1),
                n_iter=250,
                random_state=42
            )
        else:
            reducer = PCA(n_components=self.n_components)
        
        reduced = reducer.fit_transform(embeddings)
        
        # Create animation frames with scattering effect
        n_frames = 20  # Number of frames for the animation
        
        # Initial scattered positions (random)
        scattered = np.random.randn(len(reduced), self.n_components) * np.std(reduced) * 2
        
        # Create frames interpolating from scattered to final positions
        self.animation_frames = []
        for i in range(n_frames):
            progress = i / (n_frames - 1)
            # Use smooth easing function
            ease = 1 - (1 - progress) ** 2  # Quadratic easing
            frame = scattered * (1 - ease) + reduced * ease
            self.animation_frames.append(frame)
            
            if progress_callback:
                progress_callback(0.4 + 0.5 * progress, "Generating animation frames...")
        
        return reduced

    def prepare_visualization(
        self,
        n_words: Optional[int] = None,
        specific_words: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ):
        """
        Prepare embeddings for visualization.
        
        Args:
            n_words (int, optional): Number of most frequent words to visualize
            specific_words (List[str], optional): List of specific words to visualize
            progress_callback (callable, optional): Callback for progress updates
        """
        if progress_callback:
            progress_callback(0.1, "Preparing word list...")
            
        if specific_words:
            word_indices = [
                self.preprocessor.word2idx[word]
                for word in specific_words
                if word in self.preprocessor.word2idx
            ]
            self.words = [self.preprocessor.idx2word[idx] for idx in word_indices]
        else:
            if n_words is None:
                n_words = min(50, self.preprocessor.vocab_size)
            
            # Filter out special tokens and get most frequent words
            word_freqs = {
                word: freq for word, freq in self.preprocessor.word_freqs.most_common()
                if word not in [self.preprocessor.PAD_TOKEN, self.preprocessor.UNK_TOKEN]
            }
            
            top_words = list(word_freqs.keys())[:n_words]
            word_indices = [self.preprocessor.word2idx[word] for word in top_words]
            self.words = top_words
        
        if not word_indices:
            raise ValueError("No valid words to visualize")
        
        if progress_callback:
            progress_callback(0.3, "Getting word embeddings...")
            
        # Get embeddings for selected words
        embeddings = self.model.embeddings.weight.data[word_indices].cpu().numpy()
        
        # Reduce dimensions with progress updates
        self.reduced_embeddings = self._reduce_dimensions(embeddings, progress_callback)

    def plot_2d(
        self,
        title: str = 'Word Embeddings Visualization',
        show_clusters: bool = True,
        show_steps: bool = True
    ) -> plt.Figure:
        """
        Create animated 2D visualization of word embeddings.
        
        Args:
            title (str): Plot title
            show_clusters (bool): Whether to show word clusters
            show_steps (bool): Whether to show step-by-step details
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if self.n_components != 2:
            raise ValueError('This method requires 2D embeddings')
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Get clusters if requested
        clusters = None
        if show_clusters and len(self.words) >= 3:
            try:
                from sklearn.cluster import KMeans
                n_clusters = min(5, len(self.words) - 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(self.reduced_embeddings)
            except Exception:
                show_clusters = False
        
        # Create animation
        def update(frame):
            ax.clear()
            
            # Plot points
            if show_clusters and clusters is not None:
                scatter = ax.scatter(
                    self.animation_frames[frame][:, 0],
                    self.animation_frames[frame][:, 1],
                    c=clusters,
                    cmap='tab10',
                    alpha=0.6
                )
            else:
                scatter = ax.scatter(
                    self.animation_frames[frame][:, 0],
                    self.animation_frames[frame][:, 1],
                    alpha=0.6
                )
            
            # Add word labels with tooltips
            for i, word in enumerate(self.words):
                ax.annotate(
                    word,
                    (self.animation_frames[frame][i, 0],
                     self.animation_frames[frame][i, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    bbox=dict(
                        boxstyle='round,pad=0.5',
                        fc='yellow',
                        alpha=0.3
                    ),
                    arrowprops=dict(
                        arrowstyle='->',
                        connectionstyle='arc3,rad=0'
                    )
                )
            
            # Add title and labels
            ax.set_title(f"{title} - Step {frame+1}/{len(self.animation_frames)}")
            ax.set_xlabel(f'Dimension 1 ({self.method.upper()})')
            ax.set_ylabel(f'Dimension 2 ({self.method.upper()})')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add step description if requested
            if show_steps:
                if self.method == 'tsne':
                    step_text = f"t-SNE Iteration {frame+1}: Optimizing word positions..."
                else:
                    progress = (frame + 1) / len(self.animation_frames)
                    step_text = f"PCA Progress: {progress*100:.1f}% complete"
                
                ax.text(
                    0.02, 0.98,
                    step_text,
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top'
                )
        
        # Create animation
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(self.animation_frames),
            interval=1000 // self.speed_settings[self.animation_speed],
            repeat=False
        )
        
        return fig, anim

    def plot_3d(self, title: str = 'Word Embeddings Visualization') -> go.Figure:
        """
        Create 3D visualization of word embeddings.
        
        Args:
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly 3D figure
        """
        if self.n_components != 3:
            raise ValueError('This method requires 3D embeddings')
        
        fig = go.Figure()
        
        # Add word points
        fig.add_trace(go.Scatter3d(
            x=self.reduced_embeddings[:, 0],
            y=self.reduced_embeddings[:, 1],
            z=self.reduced_embeddings[:, 2],
            mode='markers+text',
            text=self.words,
            textposition='top center',
            hoverinfo='text',
            marker=dict(size=5, opacity=0.8)
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ),
            hovermode='closest'
        )
        
        return fig

    def plot_similarity_heatmap(
        self,
        words: List[str],
        title: str = 'Word Similarity Heatmap'
    ) -> plt.Figure:
        """
        Create a heatmap of word similarities.
        
        Args:
            words (List[str]): List of words to compare
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Get word indices
        word_indices = [
            self.preprocessor.word2idx[word]
            for word in words
            if word in self.preprocessor.word2idx
        ]
        valid_words = [self.preprocessor.idx2word[idx] for idx in word_indices]
        
        # Calculate similarity matrix
        embeddings = self.model.embeddings.weight.data[word_indices]
        similarity_matrix = torch.nn.functional.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        ).cpu().numpy()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            xticklabels=valid_words,
            yticklabels=valid_words,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            ax=ax
        )
        
        plt.title(title)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        return fig

    def plot_word_analogies(
        self,
        word1: str,
        word2: str,
        word3: str,
        n_analogies: int = 5
    ) -> go.Figure:
        """
        Visualize word analogies (word1 is to word2 as word3 is to ?).
        
        Args:
            word1 (str): First word in analogy
            word2 (str): Second word in analogy
            word3 (str): Third word in analogy
            n_analogies (int): Number of analogy results to show
            
        Returns:
            go.Figure: Plotly figure with analogy visualization
        """
        # Get word vectors
        vec1 = self.model.get_word_vector(word1)
        vec2 = self.model.get_word_vector(word2)
        vec3 = self.model.get_word_vector(word3)
        
        if vec1 is None or vec2 is None or vec3 is None:
            raise ValueError('One or more words not found in vocabulary')
        
        # Calculate analogy vector
        analogy_vec = vec2 - vec1 + vec3
        
        # Find similar words to the analogy vector
        similarities = torch.nn.functional.cosine_similarity(
            analogy_vec.unsqueeze(0),
            self.model.embeddings.weight,
            dim=1
        )
        
        # Get top results (excluding input words)
        exclude_indices = {
            self.preprocessor.word2idx[w]
            for w in [word1, word2, word3]
            if w in self.preprocessor.word2idx
        }
        similarities[list(exclude_indices)] = -1
        top_indices = similarities.topk(n_analogies).indices.tolist()
        
        # Prepare visualization
        words = [word1, word2, word3] + [
            self.preprocessor.idx2word[idx]
            for idx in top_indices
        ]
        embeddings = torch.stack([
            self.model.get_word_vector(w)
            for w in words
            if w in self.preprocessor.word2idx
        ]).cpu().numpy()
        
        # Reduce dimensions for visualization
        reduced = self._reduce_dimensions(embeddings)
        
        # Create interactive plot
        fig = go.Figure()
        
        # Plot original words
        fig.add_trace(go.Scatter(
            x=reduced[:3, 0],
            y=reduced[:3, 1],
            mode='markers+text',
            text=words[:3],
            textposition='top center',
            name='Input Words',
            marker=dict(size=10, symbol='circle')
        ))
        
        # Plot analogy results
        fig.add_trace(go.Scatter(
            x=reduced[3:, 0],
            y=reduced[3:, 1],
            mode='markers+text',
            text=words[3:],
            textposition='top center',
            name='Analogy Results',
            marker=dict(size=10, symbol='diamond')
        ))
        
        # Add arrows
        for i in range(len(reduced) - 1):
            fig.add_trace(go.Scatter(
                x=[reduced[i, 0], reduced[i+1, 0]],
                y=[reduced[i, 1], reduced[i+1, 1]],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ))
        
        fig.update_layout(
            title=f'Word Analogy: {word1} : {word2} :: {word3} : ?',
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            hovermode='closest'
        )
        
        return fig 