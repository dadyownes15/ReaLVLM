import unittest
import numpy as np
import sys
import os
import shutil
import tempfile
import torch

# Add the project root to sys.path to allow direct script execution
# and for Python to find the 'src' module.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.detector_implementations.SVM import OneClassSVMDetector
# Assuming load_embeddings_from_dir is the utility to load all .npy files from a directory
# and concatenate them into a single numpy array.
from src.util import load_embeddings_from_dir 


def main():
    detector = OneClassSVMDetector()
    train_X = load_embeddings_from_dir("output/non_falls_embeddings")
    test_Y = load_embeddings_from_dir("output/falls_embeddings")

    assert train_X is not None
    assert test_Y is not None

    detector.train(train_X)
    print(detector.calculate_scores(test_Y))

if __name__ == "__main__":
    main()