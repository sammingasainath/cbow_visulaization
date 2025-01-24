import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.gui.main_window import MainWindow

def main():
    """Main entry point for the application."""
    app = MainWindow()
    app.run()

if __name__ == "__main__":
    main() 