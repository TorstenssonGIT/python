"""
Main entry point for Heart Disease AI training and evaluation.
This script can be used to retrain models from scratch.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import main

if __name__ == "__main__":
    main()
