import torch
import os
# import shutil # No longer needed
from pathlib import Path
import sys
import logging

# --- Setup logging for tests ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EncoderTests")
# -----------------------------

# --- Add project root to path to allow imports from src ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --------------------------------------------------------

# --- Try importing necessary classes ---
try:
    from src.models.encoder_model import Encoder
    from src.encoder_implementations.jepa_impl import JepaEncoder
    from src.encoder_implementations.hiera_impl import HieraEncoder
    # Add InternVideoEncoder import if you want to test it
    # from src.encoder_implementations.internvideo import InternVideoEncoder 
except ImportError as e:
    logger.error(f"Failed to import necessary modules. Ensure script is run correctly and src is accessible: {e}")
    sys.exit(1)
# ------------------------------------

# --- Test Configuration ---
# Path to the real test video, relative to the project root
REAL_VIDEO_DIR = os.path.join(project_root, "data", "dummy_data")
REAL_VIDEO_PATH = os.path.join(REAL_VIDEO_DIR, "test.mp4")
# -------------------------

def test_jepa():
    """Tests JEPA initialization (using defaults) and encoding capability with a real video."""
    logger.info("--- Running JEPA Test ---")
    encoder = None
    initialization_passed = False
    try:
        logger.info("Attempting JEPA Encoder initialization (using default checkpoints)...")
        encoder = Encoder(
            encoder_type='jepa',
            device='cpu', # Force CPU for testing
        )
        assert isinstance(encoder.encoder_impl, JepaEncoder)
        assert encoder.encoder_type == 'jepa'
        logger.info(f"JEPA Encoder Initialization SUCCESSFUL. Embedding Dim: {encoder.embedding_dim}")
        initialization_passed = True
        
    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"JEPA Encoder initialization FAILED: {e}", exc_info=True)
        logger.error("Ensure default JEPA checkpoints exist at 'weights/jepa/vit_huge16-384/'")
    except Exception as e:
         logger.error(f"JEPA Encoder initialization FAILED with unknown error: {e}", exc_info=True)

    if not initialization_passed:
         logger.error("Skipping JEPA encoding test due to initialization failure.")
         return False # Indicate test failure
         
    # --- Test Encoding ---
    encoding_passed = False
    try:
        logger.info(f"Attempting JEPA encoding using video: {REAL_VIDEO_PATH}")
        if not os.path.exists(REAL_VIDEO_PATH):
            logger.error(f"Test video not found at {REAL_VIDEO_PATH}. Skipping encoding.")
            raise FileNotFoundError(f"Test video not found: {REAL_VIDEO_PATH}")
            
        embedding = encoder.encode(REAL_VIDEO_PATH)
        
        # Now expect a valid tensor output
        if embedding is not None:
            logger.info(f"JEPA encoding returned a tensor. Shape: {embedding.shape}, Device: {embedding.device}")
            assert isinstance(embedding, torch.Tensor)
            assert embedding.shape[0] == 1
            assert str(embedding.device) == 'cpu'
            encoding_passed = True
        else:
            logger.error("JEPA encoding returned None unexpectedly for a real video.")
            
    except FileNotFoundError as e:
         logger.error(f"JEPA encoding failed: {e}") # Log specific file error
    except Exception as e:
        logger.error(f"JEPA encode method FAILED unexpectedly: {e}", exc_info=True)

    logger.info(f"--- JEPA Test Complete (Success: {initialization_passed and encoding_passed}) ---")
    return initialization_passed and encoding_passed

def test_hiera():
    """Tests Hiera initialization and encoding capability with a real video."""
    logger.info("--- Running Hiera Test ---")
    encoder = None
    initialization_passed = False
    try:
        logger.info("Attempting Hiera Encoder initialization...")
        encoder = Encoder(
            encoder_type='hiera',
            device='cpu', # Force CPU for testing
        )
        assert isinstance(encoder.encoder_impl, HieraEncoder)
        assert encoder.encoder_type == 'hiera'
        assert isinstance(encoder.embedding_dim, int)
        logger.info(f"Hiera Encoder Initialization SUCCESSFUL. Embedding Dim: {encoder.embedding_dim}")
        initialization_passed = True
        
    except ImportError as e:
         if "hiera" in str(e).lower():
              logger.warning(f"Hiera library not found, skipping Hiera test: {e}")
              return True # Not a failure if library isn't installed
         else:
              logger.error(f"Hiera Encoder initialization FAILED (ImportError): {e}", exc_info=True)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Hiera Encoder initialization FAILED: {e}", exc_info=True)
    except Exception as e:
         logger.error(f"Hiera Encoder initialization FAILED with unknown error: {e}", exc_info=True)

    if not initialization_passed:
         logger.error("Skipping Hiera encoding test due to initialization failure.")
         return False # Indicate test failure

    # --- Test Encoding ---
    encoding_passed = False
    try:
        logger.info(f"Attempting Hiera encoding using video: {REAL_VIDEO_PATH}")
        if not os.path.exists(REAL_VIDEO_PATH):
            logger.error(f"Test video not found at {REAL_VIDEO_PATH}. Skipping encoding.")
            raise FileNotFoundError(f"Test video not found: {REAL_VIDEO_PATH}")

        embedding = encoder.encode(REAL_VIDEO_PATH)

        # Now expect a valid tensor output
        if embedding is not None:
            logger.info(f"Hiera encoding returned a tensor. Shape: {embedding.shape}, Device: {embedding.device}")
            assert isinstance(embedding, torch.Tensor)
            assert embedding.shape[0] == 1
            assert str(embedding.device) == 'cpu'
            encoding_passed = True
        else:
            logger.error("Hiera encoding returned None unexpectedly for a real video.")
            
    except FileNotFoundError as e:
         logger.error(f"Hiera encoding failed: {e}") # Log specific file error
    except Exception as e:
        logger.error(f"Hiera encode method FAILED unexpectedly: {e}", exc_info=True)

    logger.info(f"--- Hiera Test Complete (Success: {initialization_passed and encoding_passed}) ---")
    return initialization_passed and encoding_passed

if __name__ == "__main__":
    logger.info("Starting Encoder Tests...")
    
    # Check if test video exists before running tests
    if not os.path.exists(REAL_VIDEO_PATH):
        logger.error(f"Test video not found at: {REAL_VIDEO_PATH}")
        logger.error("Please ensure the video exists at 'data/dummy_data/test.mp4' relative to the project root.")
        sys.exit(1)
        
    results = {}
    try:
        results["jepa"] = test_jepa()
        results["hiera"] = test_hiera()
        # Add calls to other tests here (e.g., test_internvideo)
    finally:
        # No cleanup needed anymore
        pass
        
    logger.info("--- Test Summary ---")
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"Test {name}: {status}")
        if not passed:
             all_passed = False
             
    logger.info("Encoder Tests Finished.")
    sys.exit(0 if all_passed else 1) 