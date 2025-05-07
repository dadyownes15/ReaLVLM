import torch
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler # Optional

# Assuming SVDD class is in src.anomaly_detection.svdd
from src.detector_implementations.svdd import SVDD 
import logging

from util import load_embeddings_from_dir



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SVDD_Training_Example")

non_falls_embeddings_dir = "output/non_falls_embeddings" # Your path
falls_embeddings_dir = "output/falls_embeddings" # Path to your fall embeddings for testing

# 1. Load "normal" training data
X_train_normal = load_embeddings_from_dir(non_falls_embeddings_dir)

if X_train_normal is None:
    logger.error("Could not load training embeddings. Exiting.")
    exit()

# 2. (Optional but Recommended) Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
logger.info("Training data scaled.")

# 3. Initialize and train SVDD
# Adjust nu: nu is an upper bound on the fraction of training errors and a lower bound on the fraction of support vectors.
# If all your training data is 'normal', nu should be small.
svdd_nu = 0.01
logger.info(f"Initializing SVDD with nu = {svdd_nu}")
svdd_model = SVDD(nu=svdd_nu)

try:
    logger.info("Fitting SVDD model...")
    svdd_model.fit(X_train_scaled) # Use scaled data
    logger.info(f"SVDD model fitted. Radius: {svdd_model.get_radius():.4f}, Center shape: {svdd_model.get_center().shape}")

    # 4. (Example) Predict on the training data itself to see decision scores
    labels_train, scores_train = svdd_model.predict(X_train_scaled)
    num_train_outliers = np.sum(labels_train == -1)
    logger.info(f"Predictions on training data: {num_train_outliers}/{len(X_train_scaled)} marked as outliers by SVDD.")
    # logger.info(f"Training scores (first 10): {scores_train[:10]}")


    X_test_falls = load_embeddings_from_dir(falls_embeddings_dir)
    if X_test_falls is not None:
        X_test_falls_scaled = scaler.transform(X_test_falls)
        labels_falls, scores_falls = svdd_model.predict(X_test_falls_scaled)
        num_detected_falls = np.sum(labels_falls == -1)
        logger.info(f"Predictions on FALL data: {num_detected_falls}/{len(X_test_falls_scaled)} correctly marked as outliers (anomalous).")
    
    X_test_normal = load_embeddings_from_dir("path/to/new_non_fall_embeddings")
    if X_test_normal is not None:
        X_test_normal_scaled = scaler.transform(X_test_normal)
        labels_normal_test, scores_normal_test = svdd_model.predict(X_test_normal_scaled)
        num_normal_as_outliers = np.sum(labels_normal_test == -1)
        logger.info(f"Predictions on NEW NON-FALL data: {num_normal_as_outliers}/{len(X_test_normal_scaled)} incorrectly marked as outliers.")

    
except RuntimeError as e:
    logger.error(f"SVDD training or prediction failed: {e}")
except Exception as e:
    logger.error(f"An unexpected error occurred: {e}", exc_info=True)