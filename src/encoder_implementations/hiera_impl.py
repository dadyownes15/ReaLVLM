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

    def __init__(self, model_name=HIERA_MODEL_NAME, pretrained_checkpoint=HIERA_PRETRAINED_CHECKPOINT, device=None):
        """
        Initializes the HieraEncoder.

        Args:
            model_name (str): The specific Hiera model variant to load (e.g., 'hiera_base_224').
            pretrained_checkpoint (str): The name of the pretrained checkpoint to use (e.g., 'mae_in1k_ft_in1k').
            device (str, optional): Device to run the model on ('cuda', 'cpu'). 
                                     Defaults to 'cuda' if available, else 'cpu'.
        """
        if Hiera is None or load_and_preprocess_video is None:
            raise ImportError("Hiera library or utility function not available. Cannot initialize HieraEncoder.")

            
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing HieraEncoder: model='{model_name}', checkpoint='{pretrained_checkpoint}' on device '{self.device}'")

       
        self.pretrained_checkpoint = pretrained_checkpoint
        self.model_name = "facebook/" + model_name + "." + pretrained_checkpoint
        # Store configuration used for preprocessing
        self.model_config = {
            'resolution': RESOLUTION,
            'frames_per_second': FRAMES_PER_SECOND,
            'num_frames_to_sample': NUM_FRAMES_TO_SAMPLE, # Max frames to sample
            'norm_mean': DEFAULT_NORM_MEAN,
            'norm_std': DEFAULT_NORM_STD,
            'batch_size': BATCH_SIZE
        }

        # --- Load Hiera Model ---
        try:
            logger.info(f"Loading Hiera model '{self.model_name}' with checkpoint '{self.pretrained_checkpoint}'...")
            # Use the hiera library's loading function
            self.model = Hiera.from_pretrained(self.model_name)
            self.model.head = nn.Identity()
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            logger.info("Hiera model loaded successfully.")
        except KeyError:
            logger.error(f"Model name '{self.model_name}' not found in hiera library.")
            raise ValueError(f"Model name '{self.model_name}' not found in hiera library.")
        except Exception as e:
            logger.error(f"Failed to load Hiera model: {e}", exc_info=True)
            raise

        # --- Determine Embedding Dimension ---
        # Hiera models typically have a 'head' which might be for classification.
        # We usually want features before the head. Often accessible via `model.forward_features` or similar.
        # Let's try to get the dimension from the model's structure, assuming a standard feature dim attribute exists
        # or by running a dummy input if necessary.
        self.embedding_dim = None
        try:
             # Common attribute names for feature dimension
             if hasattr(self.model, 'embed_dim'):
                 self.embedding_dim = self.model.embed_dim
             elif hasattr(self.model, 'num_features'):
                  self.embedding_dim = self.model.num_features
             
             # If not found, try a dummy forward pass (more robust)
             if self.embedding_dim is None and hasattr(self.model, 'forward_features'):
                 dummy_input = torch.zeros(1, 3, RESOLUTION, RESOLUTION).to(self.device)
                 with torch.no_grad():
                      dummy_output = self.model.forward_features(dummy_input)
                 # Output shape could be e.g., [B, N, D] or [B, D] after pooling
                 if isinstance(dummy_output, torch.Tensor):
                     if dummy_output.ndim == 3: # e.g., [B, N, D]
                         self.embedding_dim = dummy_output.shape[-1]
                     elif dummy_output.ndim == 2: # e.g., [B, D]
                          self.embedding_dim = dummy_output.shape[-1]
                 del dummy_input, dummy_output
             elif self.embedding_dim is None:
                  # Fallback if standard attributes/methods aren't present
                  # Check the last linear layer if it exists and isn't the classification head
                  modules = list(self.model.modules())[::-1]
                  for layer in modules:
                       if isinstance(layer, torch.nn.Linear):
                            # Avoid the final classification head if possible (often named 'head')
                            # This is heuristic
                            parent_name = ''.join(name for name, mod in self.model.named_modules() if mod is layer)
                            if 'head' not in parent_name:
                                 self.embedding_dim = layer.in_features 
                                 break
                            elif self.embedding_dim is None: # Take head input dim as last resort
                                 self.embedding_dim = layer.in_features
                                 
        except Exception as e:
             logger.warning(f"Could not automatically determine embedding dimension: {e}", exc_info=True)

        if self.embedding_dim:
            logger.info(f"Determined Hiera embedding dimension: {self.embedding_dim}")
        else:
            # Fallback or raise error if dimension is crucial and couldn't be found
            logger.warning("Could not determine Hiera embedding dimension. Please check model structure.")
            # self.embedding_dim = 768 # Example fallback for 'base' models, adjust as needed
            # raise RuntimeError("Could not determine Hiera embedding dimension.")
            # For now, let it be None, but encoding might fail.

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
                num_frames_to_sample=self.model_config['num_frames_to_sample'],
                target_resolution=self.model_config['resolution'],
                norm_mean=self.model_config['norm_mean'],
                norm_std=self.model_config['norm_std'],
                frames_per_second=self.model_config['frames_per_second'], # Use time-based sampling
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

