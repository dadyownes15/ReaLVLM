import json
import logging
import torch
# import numpy as np # Not directly used in this snippet
from pathlib import Path
import os
import sys
# import time # Not directly used in this snippet

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Add project root to sys.path for imports from src ---
# Assuming this script is in VLLM_oneshot/src/models/
_project_root_for_encoder_model = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root_for_encoder_model not in sys.path:
    sys.path.insert(0, _project_root_for_encoder_model)
# --------------------------------------------------------

# --- Import specific encoder implementations using relative paths ---
try:
    from src.encoder_implementations.jepa_impl import JepaEncoder
    from src.encoder_implementations.hiera_impl import HieraEncoder
    from src.encoder_implementations.internvideo import InternVideoEncoder
    from src.util import VIDEO_EXTENSIONS # VIDEO_EXTENSIONS is in util.py
except ImportError as e:
    logger.error(f"Failed to import encoder implementations or VIDEO_EXTENSIONS. Error: {e}", exc_info=True)
    JepaEncoder = HieraEncoder = InternVideoEncoder = FluxViTEncoder = None
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv') # Fallback

# --- Helper functions for encode_dataset (can remain here or move to util if preferred) ---
# _load_checkpoint, _save_checkpoint, _get_video_files from your previous version
# Ensure _get_video_files uses the imported VIDEO_EXTENSIONS

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

DEFAULT_JEPA_CONFIG_PATH = "config/embeddings/jepa_vit_huge.k400_384.json"
DEFAULT_HIERA_CONFIG_PATH = "config/embeddings/hiera_huge_16x224.mae_k400_ft_k400.json"
DEFAULT_FLUXVIT_CONFIG_PATH = "config/embeddings/fluxvit_s14_k400.json"
# Define other default config paths if needed:
# DEFAULT_HIERA_CONFIG_PATH = "config/embeddings/hiera_default.json"
# DEFAULT_INTERNVIDEO_CONFIG_PATH = "config/embeddings/internvideo_default.json"

class Encoder:
    def __init__(self, encoder_type, **kwargs):
        """
        Initializes the main Encoder class.

        Args:
            encoder_type (str): Type of encoder ('jepa', 'hiera', 'internvideo').
            **kwargs:
                jepa_config_path (str, optional): Path to JEPA JSON config.
                hiera_config_path (str, optional): Path to Hiera JSON config.
                internvideo_config_path (str, optional): Path to InternVideo JSON config.
                Alternatively, encoder-specific parameters can be passed directly
                if the respective encoder supports it (though config file is preferred).
        """
        self.encoder_type = encoder_type.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Encoder '{self.encoder_type}' will use device: '{self.device}' (determined by Encoder class)")
        # Note: Individual encoders will also determine and log their device.

        self.encoder_impl = None
        self.embedding_dim = None
        
        # Determine project root to resolve relative config paths
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


        if self.encoder_type == 'jepa':
            if JepaEncoder is None:
                raise ImportError("JepaEncoder implementation could not be imported.")
            
            config_path_rel = kwargs.get('jepa_config_path', DEFAULT_JEPA_CONFIG_PATH)
            config_path_abs = os.path.join(_project_root, config_path_rel) if not os.path.isabs(config_path_rel) else config_path_rel
            
            logger.info(f"Loading JEPA configuration from: {config_path_abs}")
            if not os.path.exists(config_path_abs):
                raise FileNotFoundError(f"JEPA config file not found: {config_path_abs}")
            try:
                with open(config_path_abs, 'r') as f:
                    jepa_config_dict = json.load(f)
                
                # Pass the loaded config dictionary. JepaEncoder now expects this.
                # Any specific kwargs for JepaEncoder *not* in the JSON could be merged here if desired,
                # but current design prioritizes JSON.
                self.encoder_impl = JepaEncoder(config_dict=jepa_config_dict)
                self.embedding_dim = self.encoder_impl.embedding_dim
            except Exception as e:
                logger.error(f"Failed to initialize JepaEncoder with config {config_path_abs}: {e}", exc_info=True)
                raise
        
        elif self.encoder_type == 'hiera':
            if HieraEncoder is None:
                raise ImportError("HieraEncoder implementation could not be imported.")
            
            # Determine config path (provided or default)
            config_path_rel = kwargs.get('hiera_config_path', DEFAULT_HIERA_CONFIG_PATH)
            config_path_abs = os.path.join(_project_root, config_path_rel) if not os.path.isabs(config_path_rel) else config_path_rel
            
            logger.info(f"Loading Hiera configuration from: {config_path_abs}")
            if not os.path.exists(config_path_abs):
                raise FileNotFoundError(f"Hiera config file not found: {config_path_abs}")
            try:
                # Load config JSON
                with open(config_path_abs, 'r') as f:
                    hiera_config_dict = json.load(f)
                
                # Initialize HieraEncoder with the config dictionary
                self.encoder_impl = HieraEncoder(config_dict=hiera_config_dict)
          
            except Exception as e:
                 logger.error(f"Failed to initialize HieraEncoder with config {config_path_abs}: {e}", exc_info=True)
                 raise 

       
        elif self.encoder_type == 'internvideo':
            # Similar logic for InternVideo
            if InternVideoEncoder is None:
                 raise ImportError("InternVideoEncoder implementation could not be imported.")
            # internvideo_config_path = kwargs.get('internvideo_config_path', DEFAULT_INTERNVIDEO_CONFIG_PATH)
            # ... load json ...
            # self.encoder_impl = InternVideoEncoder(config_dict=internvideo_config_dict)
            # For now, keep old InternVideo init
            internvideo_args = {
                 'model_path_or_id': kwargs.get('internvideo_model_id'),
                 'weight_filename': kwargs.get('internvideo_weight_filename')
            }
            internvideo_args = {k: v for k, v in internvideo_args.items() if v is not None}
            try:
                self.encoder_impl = InternVideoEncoder(**internvideo_args) # InternVideo still uses kwargs
                self.embedding_dim = self.encoder_impl.embedding_dim
            except Exception as e:
                 logger.error(f"Failed to initialize InternVideoEncoder: {e}", exc_info=True)
                 raise
        else:
            raise ValueError(f"Unsupported encoder type: '{self.encoder_type}'")

        if self.encoder_impl is None:
            raise RuntimeError(f"Encoder implementation for type '{self.encoder_type}' failed to initialize.")
            
        logger.info(f"Encoder '{self.encoder_type}' (impl: {self.encoder_impl.__class__.__name__}) initialized successfully. Embedding dimension: {self.embedding_dim}")

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
            
        video_path = Path(video_path)          # ← normalise immediately
        if not video_path.is_file():
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

  