import numpy as np
import sys
import os
from pathlib import Path # Added for consistency, though os.path is used in svm_test
import logging

# Add the project root to sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.detector_implementations.isolation_forest import IsolationForestDetector

# Attempt to import the utility function as seen in svm_test.py
# If this fails, it means src.util or the function isn't structured as expected.
# The previous isolation_forest_test.py had its own load_embeddings_from_dir.
# For simplicity and consistency with svm_test.py, we try to use the shared one.
try:
    from src.util import load_embeddings_from_dir
except ImportError:
    print("Warning: Could not import 'load_embeddings_from_dir' from 'src.util'. \
          Ensure this utility exists and is correctly placed. \
          Falling back to a local dummy implementation for this test.")
    # Dummy fallback if src.util.load_embeddings_from_dir is not available
    # This won't actually load real data but allows the script to run.
    def load_embeddings_from_dir(path_str):
        print(f"Dummy load_embeddings_from_dir called for: {path_str}")
        if "non_falls" in path_str:
            return np.random.rand(10, 5) # Dummy normal data
        elif "falls" in path_str:
            return np.random.rand(5, 5) + 1 # Dummy anomaly data
        return None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IsolationForest_Simple_Test")

def main():
    logger.info("Starting Simplified Isolation Forest Detector Test...")

    # Initialize detector
    # Using random_state for reproducibility if the underlying sklearn model uses randomness
    detector = IsolationForestDetector(random_state=42, contamination='auto') 

    # Define paths (as used in svm_test.py)
    non_falls_path = "output/non_falls_embeddings"
    falls_path = "output/falls_embeddings"

    logger.info(f"Loading training data from: {non_falls_path}")
    train_X = load_embeddings_from_dir(non_falls_path)
    
    logger.info(f"Loading test data from: {falls_path}")
    test_Y = load_embeddings_from_dir(falls_path)

    if train_X is None:
        logger.error(f"Failed to load training data from {non_falls_path}. Exiting.")
        return
    if test_Y is None:
        logger.error(f"Failed to load test data from {falls_path}. Exiting.")
        return

    logger.info(f"Loaded {train_X.shape[0]} training samples and {test_Y.shape[0]} test samples.")

    try:
        logger.info("Training IsolationForestDetector...")
        detector.train(train_X)
        logger.info("Training complete.")

        logger.info("Calculating scores for test (fall) data...")
        scores = detector.calculate_score(test_Y)
        
        logger.info(f"Scores for test (fall) data (first 10): {scores[:10]}")
        logger.info(f"Average score for test (fall) data: {np.mean(scores):.4f}")
        # For Isolation Forest, lower scores indicate anomalies.

        # Optionally, print outlier/inlier predictions
        # predictions = detector.detect(test_Y)
        # num_outliers = np.sum(predictions == -1)
        # logger.info(f"Number of outliers detected in test (fall) data: {num_outliers}/{len(predictions)}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

    logger.info("Simplified Isolation Forest Detector Test finished.")

if __name__ == "__main__":
    main() 