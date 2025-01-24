import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from ttkthemes import ThemedTk
import numpy as np
from typing import Optional, List, Dict, Tuple
import json
import threading
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import plotly.graph_objects as go
from src.model.cbow import CBOWModel
from src.model.trainer import CBOWTrainer
from src.data.preprocessor import TeluguPreprocessor, CBOWDataset
from src.visualization.embeddings_viz import EmbeddingsVisualizer
import torch

class ScrollableFrame(ttk.Frame):
    """A scrollable frame widget."""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind canvas resize to window resize
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Enable mouse wheel scrolling
        self.bind_mouse_wheel()
        
        # Pack scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
    
    def _on_canvas_configure(self, event):
        """Handle canvas resize."""
        self.canvas.itemconfig(self.canvas_frame, width=event.width)
    
    def bind_mouse_wheel(self):
        """Bind mouse wheel to scrolling."""
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _on_up(event):
            self.canvas.yview_scroll(-1, "units")
        
        def _on_down(event):
            self.canvas.yview_scroll(1, "units")
        
        # Bind mouse wheel
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Bind arrow keys
        self.canvas.bind_all("<Up>", _on_up)
        self.canvas.bind_all("<Down>", _on_down)
    
    def unbind_mouse_wheel(self):
        """Unbind mouse wheel from scrolling."""
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Up>")
        self.canvas.unbind_all("<Down>")

class MainWindow:
    def __init__(self):
        """Initialize the main window with Telugu word embeddings visualization."""
        self.root = ThemedTk(theme="arc")
        self.root.title("Telugu Word Embeddings Visualizer")
        self.root.geometry("1200x800")
        
        # Add tooltips
        self.tooltips = {}
        
        # Model components
        self.model: Optional[CBOWModel] = None
        self.trainer: Optional[CBOWTrainer] = None
        self.preprocessor: Optional[TeluguPreprocessor] = None
        self.visualizer: Optional[EmbeddingsVisualizer] = None
        
        self._setup_gui()
        self._setup_menu()
        self._create_tooltips()

    def _create_tooltips(self):
        """Create tooltips for various UI elements."""
        def create_tooltip(widget, text):
            tooltip = tk.Label(self.root, text=text, relief="solid", borderwidth=1)
            tooltip.configure(bg="lightyellow", padx=5, pady=2)
            
            def enter(event):
                x, y, _, _ = widget.bbox("insert")
                x += widget.winfo_rootx() + 25
                y += widget.winfo_rooty() + 25
                tooltip.lift()
                tooltip.place(x=x, y=y)
                
            def leave(event):
                tooltip.place_forget()
                
            widget.bind('<Enter>', enter)
            widget.bind('<Leave>', leave)
            return tooltip

        # Add tooltips for visualization methods
        tsne_text = """t-SNE (t-Distributed Stochastic Neighbor Embedding):
- A technique for dimensionality reduction
- Focuses on preserving local relationships between words
- Better for visualizing clusters and similarities
- Takes longer but often gives better results"""

        pca_text = """PCA (Principal Component Analysis):
- A linear dimensionality reduction technique
- Preserves global structure of the data
- Faster but may miss some local relationships
- Good for initial exploration of the data"""

        create_tooltip(self.viz_method_menu, tsne_text if self.viz_method_var.get() == "tsne" else pca_text)
        
        # Add tooltips for other controls
        create_tooltip(self.embedding_size_entry, 
            "Size of the word vectors (larger = more detail but slower)")
        create_tooltip(self.window_size_entry,
            "Number of context words to consider on each side of the target word")
        create_tooltip(self.learning_rate_entry,
            "How quickly the model learns (smaller = more precise but slower)")

    def _setup_gui(self):
        """Set up the main GUI components."""
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.training_tab = self._create_training_tab()
        self.visualization_tab = self._create_visualization_tab()
        self.analysis_tab = self._create_analysis_tab()
        
        # Add tabs to notebook
        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.visualization_tab, text="Visualization")
        self.notebook.add(self.analysis_tab, text="Analysis")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready")

        # Bind tab change event
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_change)

    def _create_training_tab(self) -> ttk.Frame:
        """Create the training tab with educational content."""
        tab = ttk.Frame(self.notebook)
        
        # Create scrollable frame
        scroll_container = ScrollableFrame(tab)
        scroll_container.pack(fill=tk.BOTH, expand=True)
        
        # Main container with padding
        main_container = ttk.Frame(scroll_container.scrollable_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Stylish header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title = ttk.Label(
            header_frame,
            text="Train Word Embeddings",
            font=('Helvetica', 16, 'bold')
        )
        title.pack(pady=(0, 5))
        
        subtitle = ttk.Label(
            header_frame,
            text="Learn word relationships from your Telugu text",
            font=('Helvetica', 10, 'italic')
        )
        subtitle.pack()
        
        # Educational content with modern styling
        edu_frame = ttk.LabelFrame(
            main_container,
            text="✨ What is Word2Vec?",
            padding=(15, 10)
        )
        edu_frame.pack(fill=tk.X, padx=5, pady=(0, 15))
        
        edu_text = """
        🧠 Word2Vec is a technique for learning word embeddings:
        
        • 🔤 Word embeddings represent words as vectors of numbers
        • 🔄 Similar words have similar vectors
        • 📚 The model learns these vectors by predicting words from their context
        • 🎯 CBOW (Continuous Bag of Words) predicts a word from surrounding words
        • 🔍 This helps capture semantic relationships between words

        📝 How it works:
        1. Each word is initially assigned a random vector
        2. The model tries to predict words from their context
        3. Vectors are adjusted to improve predictions
        4. After training, similar words end up with similar vectors
        """
        
        ttk.Label(
            edu_frame,
            text=edu_text,
            justify=tk.LEFT,
            font=('Helvetica', 9)
        ).pack(padx=5, pady=5)
        
        # Text input with modern styling
        input_frame = ttk.LabelFrame(
            main_container,
            text="📝 Input Text",
            padding=(10, 5)
        )
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 15))
        
        self.text_input = scrolledtext.ScrolledText(
            input_frame,
            height=10,
            font=('Helvetica', 10),
            wrap=tk.WORD
        )
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Example text button with icon
        ttk.Button(
            input_frame,
            text="📋 Load Example Text",
            command=self._load_example_text,
            style='Accent.TButton'
        ).pack(pady=5)
        
        # Training parameters with modern styling
        params_frame = ttk.LabelFrame(
            main_container,
            text="⚙️ Training Parameters",
            padding=(15, 10)
        )
        params_frame.pack(fill=tk.X, padx=5, pady=(0, 15))
        
        # Parameters grid with better spacing
        params_grid = ttk.Frame(params_frame)
        params_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Embedding size
        param_row = ttk.Frame(params_grid)
        param_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(
            param_row,
            text="📊 Embedding Size:",
            font=('Helvetica', 9, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.embedding_size_var = tk.StringVar(value="100")
        self.embedding_size_entry = ttk.Entry(
            param_row,
            textvariable=self.embedding_size_var,
            width=10
        )
        self.embedding_size_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            param_row,
            text="Size of word vectors",
            font=('Helvetica', 9)
        ).pack(side=tk.LEFT, padx=5)
        
        # Window size
        param_row = ttk.Frame(params_grid)
        param_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(
            param_row,
            text="🔍 Window Size:",
            font=('Helvetica', 9, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.window_size_var = tk.StringVar(value="2")
        self.window_size_entry = ttk.Entry(
            param_row,
            textvariable=self.window_size_var,
            width=10
        )
        self.window_size_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            param_row,
            text="Words before/after target",
            font=('Helvetica', 9)
        ).pack(side=tk.LEFT, padx=5)
        
        # Learning rate
        param_row = ttk.Frame(params_grid)
        param_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(
            param_row,
            text="⚡ Learning Rate:",
            font=('Helvetica', 9, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.learning_rate_var = tk.StringVar(value="0.001")
        self.learning_rate_entry = ttk.Entry(
            param_row,
            textvariable=self.learning_rate_var,
            width=10
        )
        self.learning_rate_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            param_row,
            text="Training step size",
            font=('Helvetica', 9)
        ).pack(side=tk.LEFT, padx=5)
        
        # Number of epochs
        param_row = ttk.Frame(params_grid)
        param_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(
            param_row,
            text="🔄 Epochs:",
            font=('Helvetica', 9, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.epochs_var = tk.StringVar(value="10")
        ttk.Entry(
            param_row,
            textvariable=self.epochs_var,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            param_row,
            text="Training iterations",
            font=('Helvetica', 9)
        ).pack(side=tk.LEFT, padx=5)
        
        # Training controls with modern styling
        controls_frame = ttk.Frame(main_container)
        controls_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
        
        # Create custom button style for prominent training button
        style = ttk.Style()
        style.configure(
            'Train.TButton',
            font=('Helvetica', 11, 'bold'),
            padding=10,
            background='#4CAF50'
        )
        
        # Training button container for better visibility
        button_container = ttk.Frame(controls_frame)
        button_container.pack(fill=tk.X, pady=10)
        
        self.train_button = ttk.Button(
            button_container,
            text="🚀 Start Training",
            command=self._start_training,
            style='Train.TButton'
        )
        self.train_button.pack(expand=True, ipadx=20, ipady=5)
        
        # Progress section
        progress_frame = ttk.Frame(controls_frame)
        progress_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Progress bar with modern styling
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            style='Accent.Horizontal.TProgressbar'
        )
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to train")
        status = ttk.Label(
            progress_frame,
            textvariable=self.status_var,
            font=('Helvetica', 9)
        )
        status.pack(side=tk.LEFT, padx=5)
        
        return tab

    def _create_visualization_tab(self) -> ttk.Frame:
        """Create the visualization tab with improved controls and explanations."""
        tab = ttk.Frame(self.notebook)
        
        # Create scrollable frame
        scroll_container = ScrollableFrame(tab)
        scroll_container.pack(fill=tk.BOTH, expand=True)
        
        # Main container with padding and style
        main_container = ttk.Frame(scroll_container.scrollable_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Stylish header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title = ttk.Label(
            header_frame,
            text="Word Embeddings Visualization",
            font=('Helvetica', 16, 'bold')
        )
        title.pack(pady=(0, 5))
        
        subtitle = ttk.Label(
            header_frame,
            text="Explore how words relate to each other in the embedding space",
            font=('Helvetica', 10, 'italic')
        )
        subtitle.pack()
        
        # Visualization controls with modern styling
        controls_frame = ttk.LabelFrame(
            main_container,
            text="✨ Visualization Controls",
            padding=(15, 10)
        )
        controls_frame.pack(fill=tk.X, padx=5, pady=(0, 15))
        
        # Add help text with better formatting
        help_text = ttk.Label(
            controls_frame,
            text="""
            📊 Visualization Methods:
            • t-SNE: Better for seeing word relationships and clusters
            • PCA: Faster, good for overview of word distributions
            
            💡 Pro Tips:
            • Try both methods to see different aspects of the data
            • Adjust animation speed to follow the process
            • Hover over points to see word details
            • Use the toolbar to zoom and pan
            """,
            justify=tk.LEFT,
            font=('Helvetica', 9)
        )
        help_text.grid(row=0, column=0, columnspan=4, padx=5, pady=10, sticky='w')
        
        # Controls grid with better spacing
        controls_grid = ttk.Frame(controls_frame)
        controls_grid.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(5, 0))
        
        # Method selection with icon
        method_frame = ttk.Frame(controls_grid)
        method_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(
            method_frame,
            text="🔍 Method:",
            font=('Helvetica', 9, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.viz_method_var = tk.StringVar(value="tsne")
        self.viz_method_menu = ttk.OptionMenu(
            method_frame,
            self.viz_method_var,
            "t-SNE",
            "t-SNE",
            "PCA",
            command=self._update_method_help
        )
        self.viz_method_menu.pack(side=tk.LEFT)
        
        # Animation speed with icon
        speed_frame = ttk.Frame(controls_grid)
        speed_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(
            speed_frame,
            text="⚡ Speed:",
            font=('Helvetica', 9, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.speed_var = tk.StringVar(value="normal")
        speed_menu = ttk.OptionMenu(
            speed_frame,
            self.speed_var,
            "normal",
            "slow",
            "normal",
            "fast"
        )
        speed_menu.pack(side=tk.LEFT)
        
        # Show steps checkbox with icon
        steps_frame = ttk.Frame(controls_grid)
        steps_frame.pack(side=tk.LEFT, padx=10)
        
        self.show_steps_var = tk.BooleanVar(value=True)
        show_steps_cb = ttk.Checkbutton(
            steps_frame,
            text="📝 Show Step Details",
            variable=self.show_steps_var,
            command=self._update_visualization
        )
        show_steps_cb.pack(side=tk.LEFT)
        
        # Step details text with better styling
        steps_frame = ttk.LabelFrame(
            main_container,
            text="🔍 Process Details",
            padding=(10, 5)
        )
        steps_frame.pack(fill=tk.X, padx=5, pady=(0, 15))
        
        self.steps_text = scrolledtext.ScrolledText(
            steps_frame,
            height=3,
            font=('Helvetica', 9),
            wrap=tk.WORD
        )
        self.steps_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Visualization area with shadow effect
        viz_container = ttk.Frame(main_container)
        viz_container.pack(fill=tk.BOTH, expand=True, padx=5)
        
        self.viz_frame = ttk.Frame(
            viz_container,
            style='Card.TFrame'
        )
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Create custom styles
        style = ttk.Style()
        style.configure(
            'Card.TFrame',
            background='white',
            borderwidth=1,
            relief='solid'
        )
        
        return tab

    def _create_analysis_tab(self) -> ttk.Frame:
        """Create the analysis tab with modern styling."""
        tab = ttk.Frame(self.notebook)
        
        # Create scrollable frame
        scroll_container = ScrollableFrame(tab)
        scroll_container.pack(fill=tk.BOTH, expand=True)
        
        # Main container with padding
        main_container = ttk.Frame(scroll_container.scrollable_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Stylish header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title = ttk.Label(
            header_frame,
            text="Word Relationship Analysis",
            font=('Helvetica', 16, 'bold')
        )
        title.pack(pady=(0, 5))
        
        subtitle = ttk.Label(
            header_frame,
            text="Explore relationships and patterns between words",
            font=('Helvetica', 10, 'italic')
        )
        subtitle.pack()
        
        # Word similarity section
        sim_frame = ttk.LabelFrame(
            main_container,
            text="🔍 Find Similar Words",
            padding=(15, 10)
        )
        sim_frame.pack(fill=tk.X, padx=5, pady=(0, 15))
        
        sim_content = ttk.Frame(sim_frame)
        sim_content.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(
            sim_content,
            text="Word:",
            font=('Helvetica', 9, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.sim_word_var = tk.StringVar()
        ttk.Entry(
            sim_content,
            textvariable=self.sim_word_var,
            width=20,
            font=('Helvetica', 10)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            sim_content,
            text="🔎 Find Similar Words",
            command=self._find_similar_words,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, padx=5)
        
        # Word analogies section
        analogy_frame = ttk.LabelFrame(
            main_container,
            text="🧮 Word Analogies",
            padding=(15, 10)
        )
        analogy_frame.pack(fill=tk.X, padx=5, pady=(0, 15))
        
        # Add explanation text
        ttk.Label(
            analogy_frame,
            text="Find words that complete the analogy pattern (e.g., 'anna : akka :: mama : ?')",
            font=('Helvetica', 9, 'italic'),
            wraplength=600
        ).pack(fill=tk.X, padx=5, pady=(0, 10))
        
        analogy_content = ttk.Frame(analogy_frame)
        analogy_content.pack(fill=tk.X, padx=5, pady=5)
        
        # Word 1
        word_frame = ttk.Frame(analogy_content)
        word_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            word_frame,
            text="Word 1:",
            font=('Helvetica', 9, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.word1_var = tk.StringVar()
        ttk.Entry(
            word_frame,
            textvariable=self.word1_var,
            width=15,
            font=('Helvetica', 10)
        ).pack(side=tk.LEFT)
        
        # Is to
        ttk.Label(
            analogy_content,
            text="is to",
            font=('Helvetica', 9, 'italic')
        ).pack(side=tk.LEFT, padx=5)
        
        # Word 2
        word_frame = ttk.Frame(analogy_content)
        word_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            word_frame,
            text="Word 2:",
            font=('Helvetica', 9, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.word2_var = tk.StringVar()
        ttk.Entry(
            word_frame,
            textvariable=self.word2_var,
            width=15,
            font=('Helvetica', 10)
        ).pack(side=tk.LEFT)
        
        # As
        ttk.Label(
            analogy_content,
            text="as",
            font=('Helvetica', 9, 'italic')
        ).pack(side=tk.LEFT, padx=5)
        
        # Word 3
        word_frame = ttk.Frame(analogy_content)
        word_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            word_frame,
            text="Word 3:",
            font=('Helvetica', 9, 'bold')
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.word3_var = tk.StringVar()
        ttk.Entry(
            word_frame,
            textvariable=self.word3_var,
            width=15,
            font=('Helvetica', 10)
        ).pack(side=tk.LEFT)
        
        # Is to ?
        ttk.Label(
            analogy_content,
            text="is to ?",
            font=('Helvetica', 9, 'italic')
        ).pack(side=tk.LEFT, padx=5)
        
        # Solve button
        ttk.Button(
            analogy_content,
            text="✨ Solve Analogy",
            command=self._solve_analogy,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, padx=5)
        
        # Results area with modern styling
        results_frame = ttk.LabelFrame(
            main_container,
            text="📊 Analysis Results",
            padding=(10, 5)
        )
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 10))
        
        self.results_text = tk.Text(
            results_frame,
            height=10,
            font=('Helvetica', 10),
            wrap=tk.WORD,
            padx=5,
            pady=5
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        return tab

    def _setup_menu(self):
        """Set up the application menu."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Model", command=self._load_model)
        file_menu.add_command(label="Save Model", command=self._save_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def _load_example_text(self):
        """Load example Telugu text in English script."""
        example_text = """
nenu intiki veltunnanu
meeru ekkadiki veltunnaru
na peru ravi
idi na pustakam
nenu telugu nerchukuntunnanu
ivi manci pandlu
repu varsham padutundi
naku telugu istam
nenu annam tintunnanu
meeru em chestunnaru
"""
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", example_text.strip())

    def _load_text_file(self):
        """Load text from a file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                self.text_input.delete(1.0, tk.END)
                self.text_input.insert(tk.END, text)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def _start_training(self):
        """Start the training process."""
        # Get input text
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please enter some text to train on")
            return
        
        try:
            # Get parameters
            embedding_size = int(self.embedding_size_var.get())
            window_size = int(self.window_size_var.get())
            learning_rate = float(self.learning_rate_var.get())
            epochs = int(self.epochs_var.get())
            
            # Validate parameters
            if embedding_size < 1:
                raise ValueError("Embedding size must be positive")
            if window_size < 1:
                raise ValueError("Window size must be positive")
            if learning_rate <= 0:
                raise ValueError("Learning rate must be positive")
            if epochs < 1:
                raise ValueError("Number of epochs must be positive")
            
            # Disable training button
            self.train_button.config(state=tk.DISABLED)
            self.status_var.set("Preprocessing text...")
            self.progress_var.set(0)
            self.root.update()
            
            # Initialize preprocessor and model
            self.preprocessor = TeluguPreprocessor()
            self.preprocessor.fit(text)
            
            self.model = CBOWModel(
                vocab_size=self.preprocessor.vocab_size,
                embedding_size=embedding_size
            )
            
            # Prepare training data
            X, y = self.preprocessor.prepare_cbow_training_data(window_size)
            
            # Train model
            def progress_callback(epoch: int, loss: float):
                progress = ((epoch + 1) / epochs) * 100
                self.progress_var.set(progress)
                self.status_var.set(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
                self.root.update()
            
            self.model.train(
                X=X,
                y=y,
                learning_rate=learning_rate,
                epochs=epochs,
                progress_callback=progress_callback
            )
            
            # Training complete
            self._training_complete()
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            print(f"Training error details: {str(e)}")
            self.train_button.config(state=tk.NORMAL)
            self.status_var.set("Training failed")
    
    def _training_complete(self):
        """Called when training is complete."""
        self.train_button.config(state=tk.NORMAL)
        self.progress_var.set(100)
        self.status_var.set("Training complete!")
        
        try:
            # Switch to visualization tab
            self.notebook.select(1)  # Select visualization tab
            
            # Update visualization
            self._update_visualization()
            
            # Show success message
            messagebox.showinfo(
                "Training Complete",
                "Training completed successfully!\n\n"
                "The visualization tab has been updated.\n"
                "You can now explore word relationships and try different visualization methods."
            )
        except Exception as e:
            messagebox.showerror(
                "Visualization Error",
                f"Training completed, but visualization failed: {str(e)}\n\n"
                "You can try updating the visualization manually from the Visualization tab."
            )

    def _update_visualization(self):
        """Update the visualization with current settings and animations."""
        if not hasattr(self, 'model') or not self.model:
            messagebox.showwarning(
                "Warning",
                "Please train or load a model first."
            )
            return
        
        try:
            # Clear previous visualization
            for widget in self.viz_frame.winfo_children():
                widget.destroy()
            
            # Create status label
            self.viz_status = ttk.Label(self.viz_frame, text="Preparing visualization...")
            self.viz_status.pack(pady=5)
            
            # Get visualization parameters
            method = self.viz_method_var.get()
            speed = self.speed_var.get()
            show_steps = self.show_steps_var.get()
            
            # Create visualizer
            self.visualizer = EmbeddingsVisualizer(
                self.model,
                self.preprocessor,
                method=method,
                animation_speed=speed
            )
            
            # Prepare visualization with progress updates
            def progress_callback(progress, status):
                self.viz_status.config(text=status)
                self.root.update_idletasks()  # Use update_idletasks instead of update
            
            self.visualizer.prepare_visualization(
                n_words=50,
                progress_callback=progress_callback
            )
            
            # Create plot with animation
            fig, anim = self.visualizer.plot_2d(
                title=f"Word Embeddings ({method})",
                show_clusters=True,
                show_steps=show_steps
            )
            
            # Add explanation based on method
            if method == "t-SNE":
                explanation = """t-SNE Visualization Explained:
                
• Points that start close together represent similar words
• The animation shows how t-SNE organizes words to preserve relationships
• Clusters indicate groups of words with similar meanings
• Distance between points shows how similar the words are
• The process iteratively refines positions to show word relationships better
"""
            else:
                explanation = """PCA Visualization Explained:
                
• PCA finds the most important directions in the word vectors
• The animation shows the projection onto these main directions
• Words that appear close share similar patterns in their usage
• The visualization preserves global structure of relationships
• Faster than t-SNE but may miss some local patterns
"""
            
            explanation_label = ttk.Label(
                self.viz_frame,
                text=explanation,
                justify=tk.LEFT,
                wraplength=600
            )
            explanation_label.pack(pady=10)
            
            # Add plot to canvas
            canvas = FigureCanvasTkAgg(fig, self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.viz_frame)
            toolbar.update()
            
            # Save references
            self.fig_canvas = canvas
            self.current_animation = anim
            
            # Update status
            self.viz_status.config(text="Visualization complete! Use toolbar to explore.")
            
        except Exception as e:
            self.viz_status.config(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Visualization failed: {str(e)}\n\nPlease try a different visualization method or parameters.")

    def _find_similar_words(self):
        """Find similar words with improved error handling and visualization."""
        if not self.model or not self.preprocessor:
            messagebox.showerror("Error", "Please train or load a model first")
            return
        
        word = self.sim_word_var.get().strip().lower()
        if not word:
            messagebox.showerror("Error", "Please enter a word")
            return
        
        # Check if word exists in vocabulary
        if word not in self.preprocessor.word2idx:
            # Find closest matching words
            closest_words = self._find_closest_matches(word)
            if closest_words:
                msg = f"Word '{word}' not found. Did you mean:\n" + "\n".join(closest_words)
                if messagebox.askyesno("Word Not Found", msg + "\n\nWould you like to use the first suggestion?"):
                    word = closest_words[0]
                    self.sim_word_var.set(word)
                else:
                    return
            else:
                messagebox.showerror("Error", f"Word '{word}' not found in vocabulary")
                return
        
        # Get word index and find similar words
        word_idx = self.preprocessor.word2idx[word]
        similar_words = self.model.get_similar_words(word_idx, preprocessor=self.preprocessor)
        
        # Display results with visualization
        self._show_similar_words_visualization(word, similar_words)

    def _find_closest_matches(self, word: str, max_matches: int = 3) -> List[str]:
        """Find closest matching words in vocabulary using string similarity."""
        from difflib import get_close_matches
        vocab_words = list(self.preprocessor.word2idx.keys())
        return get_close_matches(word, vocab_words, n=max_matches, cutoff=0.6)

    def _show_similar_words_visualization(self, word: str, similar_words: List[Tuple[str, float]]):
        """Show similar words with an interactive visualization."""
        # Create new window
        viz_window = tk.Toplevel(self.root)
        viz_window.title(f"Similar Words to '{word}'")
        viz_window.geometry("800x600")
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Plot central word
        ax.scatter([0], [0], c='red', s=100, label='Query Word')
        ax.annotate(word, (0, 0), xytext=(5, 5), textcoords='offset points')
        
        # Plot similar words in a circle
        n_words = len(similar_words)
        angles = np.linspace(0, 2*np.pi, n_words, endpoint=False)
        
        for (similar_word, similarity), angle in zip(similar_words, angles):
            x = np.cos(angle) * (1 - similarity)
            y = np.sin(angle) * (1 - similarity)
            ax.scatter([x], [y], c='blue', s=50)
            ax.annotate(
                f"{similar_word}\n(sim: {similarity:.2f})",
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3)
            )
            ax.plot([0, x], [0, y], 'g--', alpha=0.3)
        
        ax.set_title(f"Words Similar to '{word}'")
        ax.legend()
        ax.grid(True)
        
        # Add to window
        canvas = FigureCanvasTkAgg(fig, master=viz_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, viz_window)
        toolbar.update()
        
        # Add explanation text
        explanation = ttk.Label(viz_window, text="""
        • Red point: Query word
        • Blue points: Similar words
        • Distance from center: Dissimilarity (closer = more similar)
        • Green lines: Relationship strength
        • Use toolbar to zoom/pan
        """, justify=tk.LEFT)
        explanation.pack(pady=5)

    def _update_method_help(self, *args):
        """Update help text based on selected visualization method."""
        method = self.viz_method_var.get()
        if method == "t-SNE":
            help_text = """t-SNE Visualization:
            • Preserves local relationships between words
            • Good for finding clusters of similar words
            • Takes longer but shows more detail
            • Hover over points to see words"""
        else:
            help_text = """PCA Visualization:
            • Shows global structure of word relationships
            • Faster but may miss some local details
            • Good for initial exploration
            • Use toolbar to zoom and explore"""
            
        self.steps_text.delete(1.0, tk.END)
        self.steps_text.insert(tk.END, help_text)

    def _solve_analogy(self):
        """Solve word analogy."""
        if not self.model or not self.preprocessor:
            messagebox.showerror("Error", "Please train or load a model first")
            return
        
        word1 = self.word1_var.get().strip()
        word2 = self.word2_var.get().strip()
        word3 = self.word3_var.get().strip()
        
        if not all([word1, word2, word3]):
            messagebox.showerror("Error", "Please enter all words")
            return
        
        try:
            # Check if words exist in vocabulary
            for word in [word1, word2, word3]:
                if word not in self.preprocessor.word2idx:
                    messagebox.showerror("Error", f"Word '{word}' not found in vocabulary")
                    return
            
            # Get word vectors
            vec1 = self.model.get_word_vector(self.preprocessor.word2idx[word1])
            vec2 = self.model.get_word_vector(self.preprocessor.word2idx[word2])
            vec3 = self.model.get_word_vector(self.preprocessor.word2idx[word3])
            
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
            }
            similarities[list(exclude_indices)] = -1
            top_k = min(5, len(self.preprocessor.word2idx) - len(exclude_indices))
            top_indices = similarities.topk(top_k).indices.tolist()
            
            # Create visualization
            analogy_window = tk.Toplevel(self.root)
            analogy_window.title("Word Analogy Results")
            analogy_window.geometry("800x600")
            
            # Create figure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            # Plot original words
            ax.scatter([0], [0], c='red', s=100, label='Input Words')
            ax.annotate(f"{word1} → {word2}", (0, 0), xytext=(5, 5), textcoords='offset points')
            
            # Plot analogy results in a circle
            angles = np.linspace(0, 2*np.pi, len(top_indices), endpoint=False)
            for idx, angle in zip(top_indices, angles):
                word = self.preprocessor.idx2word[idx]
                similarity = similarities[idx].item()
                x = np.cos(angle) * (1 - similarity)
                y = np.sin(angle) * (1 - similarity)
                
                ax.scatter([x], [y], c='blue', s=50)
                ax.annotate(
                    f"{word}\n(sim: {similarity:.2f})",
                    (x, y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3)
                )
                ax.plot([0, x], [0, y], 'g--', alpha=0.3)
            
            ax.set_title(f'Word Analogy: {word1} : {word2} :: {word3} : ?')
            ax.legend()
            ax.grid(True)
            
            # Add to window
            canvas = FigureCanvasTkAgg(fig, master=analogy_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, analogy_window)
            toolbar.update()
            
            # Add explanation
            explanation = ttk.Label(analogy_window, text=f"""
            Analogy Results for: {word1} : {word2} :: {word3} : ?
            
            • The analogy shows words that complete the relationship
            • Closer points are more similar to the expected answer
            • Green lines show the strength of the relationship
            • The visualization preserves relative similarities
            
            Top matches:
            """ + "\n".join([
                f"• {self.preprocessor.idx2word[idx]} (similarity: {similarities[idx]:.2f})"
                for idx in top_indices
            ]), justify=tk.LEFT)
            explanation.pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analogy failed: {str(e)}")
            print(f"Analogy error details: {str(e)}")

    def _load_model(self):
        """Load a saved model."""
        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.trainer.load_model(file_path)
                self.visualizer = EmbeddingsVisualizer(
                    model=self.model,
                    preprocessor=self.preprocessor
                )
                messagebox.showinfo("Success", "Model loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def _save_model(self):
        """Save the current model."""
        if not self.trainer:
            messagebox.showerror("Error", "No model available")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.trainer.save_model(file_path)
                messagebox.showinfo("Success", "Model saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")

    def _show_about(self):
        """Show about dialog with educational content."""
        about_text = """Telugu Word Embeddings Visualizer

A tool for understanding and visualizing word relationships in Telugu text (in English script).

Key Features:
• Train word embeddings using the CBOW (Continuous Bag of Words) model
• Visualize word relationships using t-SNE or PCA
• Explore similar words and analogies
• Interactive animations to understand the process

How to Use:
1. Start by entering Telugu text (in English script) in the Training tab
2. Adjust training parameters if needed (hover for explanations)
3. Click "Start Training" to train the model
4. Switch to Visualization tab to see word relationships
5. Try both t-SNE and PCA visualizations
6. Use the Analysis tab to explore word similarities

Learn More:
• t-SNE: A technique that arranges words to preserve their relationships
• PCA: A method that finds the main patterns in word usage
• Word embeddings: Vector representations that capture word meaning
"""
        messagebox.showinfo("About", about_text)

    def _on_tab_change(self, event):
        """Handle tab changes to manage scrolling bindings."""
        # Unbind all scrolling first
        for tab in [self.training_tab, self.visualization_tab, self.analysis_tab]:
            for child in tab.winfo_children():
                if isinstance(child, ScrollableFrame):
                    child.unbind_mouse_wheel()
        
        # Get current tab and bind its scrolling
        current_tab = self.notebook.select()
        current_tab_widget = self.notebook.nametowidget(current_tab)
        for child in current_tab_widget.winfo_children():
            if isinstance(child, ScrollableFrame):
                child.bind_mouse_wheel()

    def run(self):
        """Start the application."""
        self.root.mainloop()

if __name__ == "__main__":
    app = MainWindow()
    app.run() 