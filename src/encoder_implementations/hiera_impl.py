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

# Attempt to import the hiera library and utility function
try:
    from hiera import Hiera # Assuming this is the correct class from the library
    # from src.util import load_and_preprocess_video # REMOVED this import
except ImportError:
    logger.error("Hiera library not found or src.util issue. Please install hiera.")
    Hiera = None

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
        if Hiera is None:
            raise ImportError("Hiera library not available. Cannot initialize HieraEncoder.")

        logger.info("Initializing HieraEncoder from configuration dictionary...")
        params = config_dict.get("parameters")
        if params is None:
            raise ValueError("Config missing 'parameters' key.")
        preproc_params = params.get("preprocessing")
        if preproc_params is None:
             raise ValueError("Config missing 'parameters.preprocessing' key.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"HieraEncoder determined to use device: {self.device}")

        # --- Extract and Store Configuration (Strict Checking) ---
        model_param_keys = ['model_name', 'pretrained_checkpoint']
        preproc_keys = [
            'resolution', 'clip_duration_frames', 'clip_stride',
            'sampling_method', 'resize_method', 'norm_mean', 'norm_std'
        ]
        missing_model_keys = [key for key in model_param_keys if key not in params]
        missing_preproc_keys = [key for key in preproc_keys if key not in preproc_params]
        if missing_model_keys or missing_preproc_keys:
            raise ValueError(f"Missing required Hiera config keys. Model: {missing_model_keys}, Preproc: {missing_preproc_keys}")

        self.model_name_from_config = params['model_name']
        self.pretrained_checkpoint_from_config = params['pretrained_checkpoint']
        self.full_model_identifier = f"facebook/{self.model_name_from_config}.{self.pretrained_checkpoint_from_config}"
        logger.info(f"HieraEncoder will attempt to load: '{self.full_model_identifier}'")

        self.preproc_params = preproc_params
        self.resolution = preproc_params['resolution'] 
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

    # --- Preprocessing (Example - Needs Implementation) ---
    def _preprocess_clip(self, clip_frames):
        """(Private) Preprocesses a batch/list of frames for a single clip."""
        resolution = self.preproc_params['resolution']
        norm_mean = self.preproc_params['norm_mean']
        norm_std = self.preproc_params['norm_std']
        resize_method = self.preproc_params['resize_method']

        frames_tensor = torch.from_numpy(clip_frames).permute(0, 3, 1, 2).float() / 255.0
        t, c, h, w = frames_tensor.shape
        if h == 0 or w == 0: raise ValueError("Clip frame dimensions invalid.")

        transform_list = []
        if resize_method == 'hiera_larger_crop':
            resize_size = resolution + 32 
            transform_list.append(transforms.Resize(resize_size))
        # Add other resize methods if needed ('jepa_short_side', etc.)
        else:
            transform_list.append(transforms.Resize((resolution, resolution), antialias=True))

        transform_list.append(transforms.CenterCrop(resolution))
        transform_list.append(transforms.Normalize(mean=norm_mean, std=norm_std))
        transform = transforms.Compose(transform_list)

        frames_processed = torch.stack([transform(frame) for frame in frames_tensor])
        
        # Format for model: B, C, T, H, W
        frames_final = frames_processed.permute(1, 0, 2, 3) # C, T, H, W
        return frames_final.unsqueeze(0) # Add Batch dim

    @torch.no_grad()
    def encode_video(self, video_path):
        """
        Loads, preprocesses video clip-by-clip, extracts features using Hiera,
        and returns a list of embeddings for all clips.

        Args:
            video_path (str): Path to the video file.

        Returns:
            List[torch.Tensor] or None: A list of PyTorch Tensors, where each tensor
                                        is an embedding for a clip (shape [embed_dim]) on CPU,
                                        or None if an error occurs or no clips processed.
        """
        logger.info(f"Encoding video (clip-based): {video_path} using Hiera")
        try:
            # --- Get parameters from config --- 
            clip_len = self.preproc_params['clip_duration_frames']
            stride = self.preproc_params['clip_stride']
            # sampling_method = self.preproc_params['sampling_method'] # Currently assumes stride
            if stride <= 0:
                raise ValueError("clip_stride must be positive.")
            # ---------------------------------
            
            vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
            video_len = len(vr)
            logger.debug(f"Video length: {video_len} frames. Clip length: {clip_len}, Stride: {stride}")
            
            if video_len == 0:
                logger.warning(f"Video has 0 frames: {video_path}")
                return None

            clip_embeddings_list = []
            processed_clip_count = 0

            # --- Calculate Clip Start Indices --- 
            # Generate indices based on stride
            clip_start_indices = list(range(0, video_len, stride))
            
            # Ensure we process segments that are at least clip_len long
            valid_start_indices = [idx for idx in clip_start_indices if idx <= video_len - clip_len]
            
            # Check if the very last frames constitute a valid clip that wasn't started by the stride
            last_possible_start = video_len - clip_len
            if last_possible_start >= 0 and last_possible_start not in valid_start_indices:
                # Find the closest previous start index generated by the stride
                closest_prior_start = -1
                for idx in reversed(valid_start_indices):
                    if idx < last_possible_start:
                        closest_prior_start = idx
                        break
                # Add the last segment only if it doesn't overlap too much with the previous strided clip
                # Add if the stride would NOT have included it (e.g. stride 16, len 35, clip 16 -> starts 0, 16. Add start 19?)
                # Let's add it if the last possible start wasn't already captured. Simpler.
                if not valid_start_indices or valid_start_indices[-1] < last_possible_start:
                    valid_start_indices.append(last_possible_start)
                    valid_start_indices = sorted(list(set(valid_start_indices))) # Ensure uniqueness and order
            
            # Handle videos shorter than one clip length
            if not valid_start_indices and video_len > 0:
                 logger.warning(f"Video length ({video_len}) is shorter than clip length ({clip_len}). Processing the whole video as one clip (padded).")
                 valid_start_indices = [0] # Process from the beginning
                 # Padding will be handled below when getting frames
                 
            if not valid_start_indices:
                 logger.warning(f"No valid clips to process for video: {video_path}")
                 return None
            # ------------------------------------

            logger.info(f"Processing {len(valid_start_indices)} clips for video: {video_path}")
            for i, start_idx in enumerate(valid_start_indices):
                end_idx = start_idx + clip_len
                logger.debug(f"Clip {i+1}/{len(valid_start_indices)}: Frames [{start_idx}, {end_idx}) (exclusive end)")
                
                # Ensure we don't read past the end, Decord handles this but explicit check is okay
                actual_end_idx = min(end_idx, video_len) 
                frame_indices = np.arange(start_idx, actual_end_idx) 
                
                if len(frame_indices) == 0: # Should not happen with logic above, but safety check
                    logger.warning(f"Calculated empty frame indices for clip starting at {start_idx}. Skipping.")
                    continue
                    
                frames = vr.get_batch(frame_indices).asnumpy() # T, H, W, C

                # --- Handle Padding if needed --- 
                # Hiera might require a fixed input length (clip_len)
                current_clip_len = frames.shape[0]
                if current_clip_len < clip_len:
                    logger.warning(f"Padding clip (len {current_clip_len}) to {clip_len} by repeating last frame.")
                    pad_len = clip_len - current_clip_len
                    if current_clip_len > 0:
                        last_frame = frames[-1:] # Keep dimensions for np.repeat
                        padding = np.repeat(last_frame, pad_len, axis=0)
                        frames = np.concatenate((frames, padding), axis=0)
                    else:
                        # This case should be rare if video_len > 0 initially
                        logger.error(f"Cannot pad empty frame array for clip starting at {start_idx}. Skipping clip.")
                        continue
                elif current_clip_len > clip_len:
                     # This shouldn't happen with current logic, but indicates error
                     logger.error(f"Extracted clip has {current_clip_len} frames, expected {clip_len}. Skipping clip.")
                     continue
                # --------------------------------
                
                # Preprocess and encode the clip
                input_tensor = self._preprocess_clip(frames).to(self.device) # Expects [1, C, T, H, W]
                
                # Verify preprocessed shape
                if input_tensor.shape[2] != clip_len:
                    logger.error(f"Preprocessed clip time dimension ({input_tensor.shape[2]}) doesn't match expected clip length ({clip_len}). Skipping clip.")
                    continue
                
                # --- Model Forward Pass --- 
                features = self.model(input_tensor)
                logger.debug(f"Clip {i+1}: Raw features shape: {features.shape}")
                
                # --- Feature Averaging/Pooling --- 
                # Logic copied from previous version - adapt if model output changes
                if features.ndim > 2: 
                    dims_to_average = tuple(range(1, features.ndim -1))
                    final_embedding = features.mean(dim=dims_to_average, keepdim=False) if len(dims_to_average) > 0 else features
                elif features.ndim == 2: 
                    final_embedding = features
                else:
                    logger.error(f"Unexpected feature shape from Hiera model: {features.shape}. Skipping clip.")
                    continue 
                # --- Batch Dimension Handling (should be 1) --- 
                if final_embedding.shape[0] != 1:
                    logger.warning(f"Embedding batch size is {final_embedding.shape[0]} (expected 1). Averaging over batch dim.")
                    final_embedding = final_embedding.mean(dim=0, keepdim=True)
                # ---------------------------------------------
                
                # Squeeze the batch dimension if it exists and is 1, then move to CPU
                if final_embedding.ndim > 1 and final_embedding.shape[0] == 1:
                    clip_embeddings_list.append(final_embedding.squeeze(0).cpu()) 
                else: # If it's already 1D or batch dim isn't 1 (should not happen with above logic)
                    clip_embeddings_list.append(final_embedding.cpu())
                
                processed_clip_count += 1
                logger.debug(f"Stored embedding for clip {i+1}. Shape: {final_embedding.shape}")

            del vr # Release video reader memory
            
            if not clip_embeddings_list:
                logger.warning(f"No embeddings generated for video: {video_path}")
                return None

            # Return the list of tensors directly
            logger.info(f"Finished encoding video with Hiera. Generated {len(clip_embeddings_list)} clip embeddings.")
            return clip_embeddings_list

        except FileNotFoundError:
            logger.error(f"Video file not found during encoding: {video_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to encode video {video_path} with Hiera: {e}", exc_info=True)
            return None

