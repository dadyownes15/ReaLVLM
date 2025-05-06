import json
import logging
import numpy as np
import torch
from pathlib import Path
import os
import sys
import time

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Import specific encoder implementations using relative paths ---
try:
    # Go up one level from models to src, then down to encoder_implementations
    from ..encoder_implementations.jepa_impl import JepaEncoder 
    from ..encoder_implementations.hiera_impl import HieraEncoder 
    from ..encoder_implementations.internvideo import InternVideoEncoder
    # Import video extensions from util
    from ..util import VIDEO_EXTENSIONS 
except ImportError as e:
    logger.error(f"Failed to import encoder implementations or VIDEO_EXTENSIONS from util. Error: {e}", exc_info=True)
    # Log the current path for debugging if needed
    logger.debug(f"Attempting import from: {os.path.dirname(__file__)}")
    JepaEncoder = None # Set to None if import fails
    HieraEncoder = None # Set to None if import fails
    InternVideoEncoder = None # Set to None if import fails
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv') # Fallback if import fails
# ------------------------------------------------------------------

def _load_checkpoint(checkpoint_path):
    """Loads checkpoint file, handles missing file or invalid JSON."""
    default_checkpoint = {'training_videos_processed': [], 'test_videos_processed': []}
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint file not found at {checkpoint_path}. Starting fresh.")
        # Ensure parent directory exists for saving later
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        return default_checkpoint
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        # Ensure keys exist and are lists
        if not isinstance(checkpoint.get('training_videos_processed'), list):
            checkpoint['training_videos_processed'] = []
        if not isinstance(checkpoint.get('test_videos_processed'), list):
            checkpoint['test_videos_processed'] = []
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    except json.JSONDecodeError:
        logger.error(f"Checkpoint file {checkpoint_path} is corrupted. Starting fresh.", exc_info=True)
        return default_checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}. Starting fresh.", exc_info=True)
        return default_checkpoint

def _save_checkpoint(checkpoint_path, data):
    """Saves checkpoint data to JSON file."""
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        # Save atomically: write to temp file, then rename
        temp_checkpoint_path = checkpoint_path + ".tmp"
        with open(temp_checkpoint_path, 'w') as f:
            json.dump(data, f, indent=4) # Use indent for readability
        os.replace(temp_checkpoint_path, checkpoint_path) # Atomic rename
        # logger.debug(f"Checkpoint saved to {checkpoint_path}") # Optional: log every save
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}", exc_info=True)

def _get_video_files(data_path):
    """Gets a list of video filenames from a directory, filtering by extension."""
    if not os.path.isdir(data_path):
        logger.error(f"Directory not found: {data_path}")
        return []
    try:
        all_files = os.listdir(data_path)
        # Filter out hidden files and files without the correct extension
        video_files = [
            f for f in all_files
            if not f.startswith('.') and f.lower().endswith(VIDEO_EXTENSIONS)
        ]
        logger.info(f"Found {len(video_files)} video files in {data_path}")
        return video_files
    except Exception as e:
        logger.error(f"Failed to list files in {data_path}: {e}", exc_info=True)
        return []

class Encoder:
    def __init__(self, encoder_type, device="cpu", **kwargs):
        """
        Initializes the main Encoder class, which acts as a wrapper around specific implementations.

        Args:
            encoder_type (str): The type of encoder to use (e.g., 'jepa', 'heira').
            device (str, optional): Device to run the model on ('cuda', 'cpu'). 
                                     Defaults to 'cuda' if available, else 'cpu'.
            **kwargs: Additional keyword arguments specific to the chosen encoder type.
                      These will be passed down to the underlying encoder implementation's
                      constructor (e.g., JepaEncoder, HieraEncoder).
                      Examples for 'jepa': encoder_checkpoint_path, probe_checkpoint_path,
                                          model_name, resolution, frames_per_clip, etc.
                      Examples for 'heira': hiera_model_name, hiera_pretrained_checkpoint.
                      Examples for 'internvideo': internvideo_model_id.
        """
        self.encoder_type = encoder_type.lower()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Encoder of type '{self.encoder_type}' on device '{self.device}'")
       
        self.encoder_impl = None
        self.embedding_dim = None

        if self.encoder_type == 'jepa':
            if JepaEncoder is None:
                raise ImportError("JepaEncoder implementation could not be imported. Cannot initialize.")
            
            # Collect arguments intended for JepaEncoder from kwargs
            # These keys should match the parameter names in JepaEncoder.__init__
            jepa_init_keys = [
                'encoder_checkpoint_path', 'probe_checkpoint_path', 'model_name', 
                'patch_size', 'resolution', 'frames_per_clip', 'tubelet_size', 
                'checkpoint_key_encoder', 'checkpoint_key_probe', 'model_kwargs', 
                'norm_mean', 'norm_std'
            ]
            jepa_args = {key: kwargs[key] for key in jepa_init_keys if key in kwargs}
            
            try:
                # Pass only the collected arguments. JepaEncoder will use its defaults
                # for any arguments not present in jepa_args.
                self.encoder_impl = JepaEncoder(
                    **jepa_args, 
                    device=self.device
                )
                self.embedding_dim = self.encoder_impl.embedding_dim

            except Exception as e:
                logger.error(f"Failed to initialize JepaEncoder: {e}", exc_info=True)
                raise # Re-raise the exception after logging
        
        elif self.encoder_type == 'hiera':
            if HieraEncoder is None:
                raise ImportError("HieraEncoder implementation could not be imported. Cannot initialize.")
            # HieraEncoder takes optional model_name and pretrained_checkpoint
            # We extract them from kwargs if provided, otherwise HieraEncoder uses its defaults.
            hiera_args = {
                'model_name': kwargs.get('model_name'), # Use specific names to avoid conflicts
                'pretrained_checkpoint': kwargs.get('pretrained_checkpoint'),
                'finetune_checkpoint': kwargs.get('finetune_checkpoint')
            }
            # Filter out None values so HieraEncoder uses its defaults
            hiera_args = {k: v for k, v in hiera_args.items() if v is not None}
            
            try:
                self.encoder_impl = HieraEncoder(
                    **hiera_args,
                    device=self.device
                )
                self.embedding_dim = self.encoder_impl.embedding_dim 
            except Exception as e:
                 logger.error(f"Failed to initialize HieraEncoder: {e}", exc_info=True)
                 raise # Re-raise the exception after logging

        elif self.encoder_type == 'internvideo':
            if InternVideoEncoder is None:
                 raise ImportError("InternVideoEncoder implementation could not be imported. Cannot initialize.")
            # InternVideoEncoder takes an optional model_id
            internvideo_args = {
                 'model_name': kwargs.get('model_name') # Use specific name
            }
            # Filter out None value so InternVideoEncoder uses its default
            internvideo_args = {k: v for k, v in internvideo_args.items() if v is not None}
            
            try:
                self.encoder_impl = InternVideoEncoder(
                    **internvideo_args,
                    device=self.device
                )
                self.embedding_dim = self.encoder_impl.embedding_dim
            except Exception as e:
                 logger.error(f"Failed to initialize InternVideoEncoder: {e}", exc_info=True)
                 raise

        else:
            raise ValueError(f"Unsupported encoder type: '{self.encoder_type}'")

        if self.encoder_impl is None:
            raise RuntimeError(f"Encoder implementation for type '{self.encoder_type}' failed to initialize.")
            
        logger.info(f"Encoder '{self.encoder_type}' initialized successfully. Embedding dimension: {self.embedding_dim}")


    def encode(self, video_path):
        """
        Encodes a video using the initialized encoder implementation.

        Args:
            video_path (str): Path to the video file.

        Returns:
            torch.Tensor or None: The resulting embedding tensor (usually on CPU), 
                                  or None if encoding fails.
        """
        if not self.encoder_impl:
            logger.error("Encoder implementation not initialized.")
            return None
            
        if not isinstance(video_path, str) or not Path(video_path).is_file():
            logger.error(f"Invalid video path provided: {video_path}")
            return None

        logger.debug(f"Encoding video: {video_path} using {self.encoder_type} encoder.")
        try:
            # Delegate to the specific encoder's method
            # Assuming the specific encoder has an `encode_video` method
            embedding = self.encoder_impl.encode_video(video_path)
            return embedding
        except AttributeError:
             logger.error(f"The configured '{self.encoder_type}' encoder does not have an 'encode_video' method.")
             return None
        except Exception as e:
            logger.error(f"Error during encoding video {video_path} with {self.encoder_type}: {e}", exc_info=True)
            return None

    
    
    def encode_dataset(self, save_path, training_data_path, test_data_path):
        """
        Encodes a dataset of videos sequentially using the initialized encoder implementation,
        with improved reliability and checkpointing.

        Args:
            save_path (str): Base directory to save embeddings and checkpoint.
            training_data_path (str): Path to the training data directory.
            test_data_path (str): Path to the test data directory.
        """
        start_time = time.time()
        logger.info("Starting dataset encoding sequentially.")

        checkpoint_path = os.path.join(save_path, "checkpoint.json")
        training_embedding_dir = os.path.join(save_path, "training_embeddings")
        test_embedding_dir = os.path.join(save_path, "test_embeddings")

        # Create output directories if they don't exist
        os.makedirs(training_embedding_dir, exist_ok=True)
        os.makedirs(test_embedding_dir, exist_ok=True)

        checkpoint = _load_checkpoint(checkpoint_path)
        processed_training_files = set(checkpoint.get('training_videos_processed', []))
        processed_test_files = set(checkpoint.get('test_videos_processed', []))

        # --- Get lists of files to process ---
        all_training_files = set(_get_video_files(training_data_path))
        all_test_files = set(_get_video_files(test_data_path))

        training_videos_to_process = sorted(list(all_training_files - processed_training_files))
        test_videos_to_process = sorted(list(all_test_files - processed_test_files))

        logger.info(f"Found {len(training_videos_to_process)} training videos to process.")
        logger.info(f"Found {len(test_videos_to_process)} test videos to process.")

        total_processed_in_run = 0

        # --- Process Training Videos ---
        logger.info("--- Processing Training Videos ---")
        for i, filename in enumerate(training_videos_to_process):
            input_video_path = os.path.join(training_data_path, filename)
            output_embedding_path = os.path.join(training_embedding_dir, filename + ".npy")
            logger.info(f"Processing training video {i+1}/{len(training_videos_to_process)}: {filename}")

            try:
                embedding = self.encode(input_video_path)

                if embedding is None:
                    logger.warning(f"Encoding returned None for training video: {filename}. Skipping save and checkpoint.")
                    continue # Skip to the next video

                # Convert to NumPy array if needed
                if hasattr(embedding, 'numpy'):
                     if hasattr(embedding, 'device') and 'cuda' in str(embedding.device):
                         embedding_np = embedding.cpu().numpy()
                     else:
                         embedding_np = embedding.numpy()
                else:
                     embedding_np = embedding

                # Save the embedding
                np.save(output_embedding_path, embedding_np)
                logger.debug(f"Saved training embedding to: {output_embedding_path}")

                # Add to checkpoint file *immediately* after successful save
                checkpoint['training_videos_processed'].append(filename)
                _save_checkpoint(checkpoint_path, checkpoint) # Save checkpoint after each success
                total_processed_in_run += 1

            except Exception as e:
                logger.error(f"Failed to process training video {filename}: {e}", exc_info=True)
                # Continue to the next video without updating checkpoint for this one

        # --- Process Test Videos ---
        logger.info("--- Processing Test Videos ---")
        for i, filename in enumerate(test_videos_to_process):
            input_video_path = os.path.join(test_data_path, filename)
            output_embedding_path = os.path.join(test_embedding_dir, filename + ".npy")
            logger.info(f"Processing test video {i+1}/{len(test_videos_to_process)}: {filename}")

            try:
                embedding = self.encode(input_video_path)

                if embedding is None:
                    logger.warning(f"Encoding returned None for test video: {filename}. Skipping save and checkpoint.")
                    continue

                # Convert to NumPy array if needed
                if hasattr(embedding, 'numpy'):
                     if hasattr(embedding, 'device') and 'cuda' in str(embedding.device):
                         embedding_np = embedding.cpu().numpy()
                     else:
                         embedding_np = embedding.numpy()
                else:
                     embedding_np = embedding

                # Save the embedding
                np.save(output_embedding_path, embedding_np)
                logger.debug(f"Saved test embedding to: {output_embedding_path}")

                # Add to checkpoint file *immediately* after successful save
                checkpoint['test_videos_processed'].append(filename)
                _save_checkpoint(checkpoint_path, checkpoint)
                total_processed_in_run += 1

            except Exception as e:
                logger.error(f"Failed to process test video {filename}: {e}", exc_info=True)

        end_time = time.time()
        logger.info(f"Processed {total_processed_in_run} new videos in this run.")
        logger.info(f"Dataset encoding finished in {end_time - start_time:.2f} seconds.")


    def test(self):
        # Add specific tests if required
        logger.warning("Encoder.test() method is not implemented.")
        pass

  