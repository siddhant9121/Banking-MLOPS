import sys
import os

# Add project root to python path to avoid import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.train_model import main

if __name__ == "__main__":
    # Provides command line interface inherited from train_model.py
    main()
