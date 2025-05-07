import torch
import torch.nn as nn
import logging
import os
from pathlib import Path
import sys
import numpy as np
from torchvision import transforms
from decord import VideoReader, cpu
import math

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import the hiera library
try:
    from hiera import Hiera
    # Import the utility function
    from src.util import load_and_preprocess_video
except ImportError:
    logger.error("hiera library or src.util not found. Please install hiera and ensure src/util.py exists.")
    Hiera = None # Set to None if import fails
    load_and_preprocess_video = None
    # Depending on the desired behavior, you might exit or raise the error here.
    # exit(1)

# --- Constants for Hiera ---
# Default model settings, adjust as needed
HIERA_MODEL_NAME = 'hiera_huge_16x224' # Example, match the desired Hiera variant
HIERA_PRETRAINED_CHECKPOINT = 'mae_k400_ft_k400' # Example checkpoint
RESOLUTION = 224 # Typically 224 for standard Hiera models
FRAMES_PER_SECOND = 3 # Sample rate for frames
NUM_FRAMES_TO_SAMPLE = 16 # Number of frames to process per video (can be adjusted)
BATCH_SIZE = 1 # Hiera processing now happens on the whole clip
DEFAULT_NORM_MEAN = [0.485, 0.456, 0.406] # Standard ImageNet mean
DEFAULT_NORM_STD = [0.229, 0.224, 0.225] # Standard ImageNet std
# ---------------------------

class HieraEncoder:
    """Encapsulates Hiera model loading and video embedding extraction via frame averaging."""

    def __init__(self, config_dict):
        """
        Initializes the HieraEncoder using a configuration dictionary.

        Args:
            config_dict (dict): Dictionary containing all necessary parameters.
        """
        if Hiera is None or load_and_preprocess_video is None:
            raise ImportError("Hiera library or load_and_preprocess_video utility not available. Cannot initialize HieraEncoder.")

        logger.info("Initializing HieraEncoder from configuration dictionary...")
        params = config_dict.get("parameters")
        if params is None:
            raise ValueError("Configuration dictionary must contain a 'parameters' key.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"HieraEncoder determined to use device: {self.device}")

        # --- Extract and Store Configuration (Strict Checking) ---
        required_keys = [
            'model_name', 'pretrained_checkpoint', 'resolution', 
            'frames_per_second', 'num_frames_to_sample', 
            'norm_mean', 'norm_std'
        ]
        
        # Check for missing keys
        missing_keys = [key for key in required_keys if key not in params]
        if missing_keys:
            raise ValueError(f"Missing required parameters in Hiera config: {missing_keys}")

        # Access parameters directly (will raise KeyError if check above failed, but cleaner)
        self.model_name_from_config = params['model_name']
        self.pretrained_checkpoint_from_config = params['pretrained_checkpoint']
        self.resolution = params['resolution']
        self.frames_per_second = params['frames_per_second']
        self.num_frames_to_sample = params['num_frames_to_sample']
        self.norm_mean = params['norm_mean']
        self.norm_std = params['norm_std']
        
        self.full_model_identifier = f"facebook/{self.model_name_from_config}.{self.pretrained_checkpoint_from_config}"
        logger.info(f"HieraEncoder will attempt to load: '{self.full_model_identifier}' on device '{self.device}'")
        # -----------------------------------

        # --- Load Hiera Model ---
        try:
            logger.info(f"Loading Hiera model using identifier: '{self.full_model_identifier}'...")
            self.model = Hiera.from_pretrained(self.full_model_identifier)
            self.model.head = nn.Identity()
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("Hiera model loaded and configured successfully.")

        except Exception as e:
            logger.error(f"Failed to load or configure Hiera model ('{self.full_model_identifier}'): {e}", exc_info=True)
            raise

    @torch.no_grad()
    def encode_video(self, video_path):
        """
        Loads, preprocesses a video, extracts frame features using Hiera,
        and returns the averaged feature vector.

        Args:
            video_path (str): Path to the video file.

        Returns:
            torch.Tensor or None: The averaged embedding tensor (shape [1, embed_dim]) on CPU,
                                  or None if an error occurs.
        """
        logger.info(f"Encoding video: {video_path} using Hiera")
        try:
            # Load and preprocess video frames using utility function
            # Hiera expects CTHW format and specific resize
            frames_clip = load_and_preprocess_video(
                video_path=video_path,
                num_frames_to_sample=self.num_frames_to_sample,
                target_resolution=self.resolution,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
                frames_per_second=self.frames_per_second, # Use time-based sampling
                resize_impr='hiera', # Use Hiera-style resize
                return_format='CTHW' # Request CTHW format
            )

            if frames_clip is None or frames_clip.shape[1] == 0: # Check time dimension (C, T, H, W)
                logger.warning(f"Preprocessing failed or resulted in 0 frames for {video_path}. Skipping.")
                return None

            # Add batch dimension -> [1, C, T, H, W]
            input_tensor = frames_clip.unsqueeze(0).to(self.device)
            logger.debug(f"Passing tensor of shape {input_tensor.shape} to Hiera model.")

            # --- Feature Extraction (Process the whole clip at once) ---
            # Now self.model() should output features before the original head
            # because we replaced it with nn.Identity()
            features = self.model(input_tensor)
            logger.debug(f"Shape after model call (head replaced): {features.shape}")

            # Handle potential pooling within the model (output might be [B, D] or [B, N, D] etc.)
            # We want to get a single vector per video -> [1, D]
            if features.ndim > 2: 
                # Average over all dimensions except the batch dimension (0) and feature dimension (-1)
                # For Hiera output [B, N, D] -> average over N (dim 1)
                # If output is different (e.g. [B, C, T, H, W]), adjust dims_to_average
                dims_to_average = tuple(range(1, features.ndim -1)) 
                if len(dims_to_average) > 0:
                     final_embedding = features.mean(dim=dims_to_average, keepdim=False)
                     logger.debug(f"Averaged features over dims {dims_to_average}. Shape before: {features.shape}, after: {final_embedding.shape}")
                else: # Already [B,D] ?
                      final_embedding = features 
            elif features.ndim == 2: # Shape is already [B, D]
                 final_embedding = features
            else:
                logger.error(f"Unexpected feature shape from Hiera model: {features.shape}. Expected at least 2 dims.")
                return None 

            # Ensure output is [1, D]
            if final_embedding.shape[0] != 1:
                 logger.warning(f"Final embedding batch size is {final_embedding.shape[0]} (expected 1). Averaging over batch dim as fallback.")
                 final_embedding = final_embedding.mean(dim=0, keepdim=True)
                 
            logger.debug(f"Got final Hiera embedding shape: {final_embedding.shape}")
            return final_embedding.cpu() # Return embedding on CPU

        except FileNotFoundError:
            logger.error(f"Video file not found during encoding: {video_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to encode video {video_path} with Hiera: {e}", exc_info=True)
            return None

