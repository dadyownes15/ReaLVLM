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
    from models.encoder import Encoder
    from src.encoder_implementations.jepa_impl import JepaEncoder
    from src.encoder_implementations.hiera_impl import HieraEncoder
    from src.encoder_implementations.flux_vit_impl import FluxViTEncoder
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
        
        # Now expect a list of tensor outputs
        if embedding is not None and isinstance(embedding, list):
            if embedding: # Check if the list is not empty
                logger.info(f"JEPA encoding returned a list of {len(embedding)} embedding(s).")
                first_embedding = embedding[0]
                assert isinstance(first_embedding, torch.Tensor)
                assert str(first_embedding.device) == 'cpu' # Assuming CPU for tests
                logger.info(f"First JEPA embedding - Shape: {first_embedding.shape}, Device: {first_embedding.device}, dtype: {first_embedding.dtype}")
                encoding_passed = True
            else:
                logger.info("JEPA encoding returned an empty list.")
                # Decide if an empty list is a pass or fail for your test case
                # encoding_passed = True # Or False, depending on expectation
        elif embedding is None:
            logger.error("JEPA encoding returned None unexpectedly for a real video.")
        else:
            logger.error(f"JEPA encoding returned an unexpected type: {type(embedding)}")
            
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
        )
        assert isinstance(encoder.encoder_impl, HieraEncoder)
        assert encoder.encoder_type == 'hiera'
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

        # Now expect a list of tensor outputs
        if embedding is not None and isinstance(embedding, list):
            if embedding: # Check if the list is not empty
                logger.info(f"Hiera encoding returned a list of {len(embedding)} embedding(s).")
                first_embedding = embedding[0]
                assert isinstance(first_embedding, torch.Tensor)
                # For Hiera, embedding_dim might not be set on the top-level Encoder if model.head is Identity()
                # So, we check the device of the tensor itself.
                assert str(first_embedding.device) == 'cpu' # Assuming CPU for tests
                logger.info(f"First Hiera embedding - Shape: {first_embedding.shape}, Device: {first_embedding.device}, dtype: {first_embedding.dtype}")
                encoding_passed = True
            else:
                logger.info("Hiera encoding returned an empty list.")
                # Decide if an empty list is a pass or fail for your test case
        elif embedding is None:
            logger.error("Hiera encoding returned None unexpectedly for a real video.")
        else:
            logger.error(f"Hiera encoding returned an unexpected type: {type(embedding)}")
            
    except FileNotFoundError as e:
         logger.error(f"Hiera encoding failed: {e}") # Log specific file error
    except Exception as e:
        logger.error(f"Hiera encode method FAILED unexpectedly: {e}", exc_info=True)

    logger.info(f"--- Hiera Test Complete (Success: {initialization_passed and encoding_passed}) ---")
    return initialization_passed and encoding_passed

def test_fluxvit():
    """Tests FluxViT initialization and encoding capability with a real video."""
    logger.info("--- Running FluxViT Test ---")
    encoder = None
    initialization_passed = False
    try:
        logger.info("Attempting FluxViT Encoder initialization (using default config)...")
        # Assuming fluxvit_s14_k400.json is the default and points to a valid checkpoint
        # or that the FluxViTEncoder handles a missing checkpoint gracefully (e.g. random weights for testing structure)
        encoder = Encoder(
            encoder_type='fluxvit',
            # device='cpu', # FluxViTEncoder will determine its device, can be overridden if needed
        )
        assert isinstance(encoder.encoder_impl, FluxViTEncoder)
        assert encoder.encoder_type == 'fluxvit'
        logger.info(f"FluxViT Encoder Initialization SUCCESSFUL. Embedding Dim: {encoder.embedding_dim}")
        initialization_passed = True
        
    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"FluxViT Encoder initialization FAILED: {e}", exc_info=True)
        logger.error("Ensure default FluxViT config 'config/embeddings/fluxvit_s14_k400.json' exists and points to a valid checkpoint, "
                     "and fluxvit_model_def.py is at the project root.")
    except Exception as e:
         logger.error(f"FluxViT Encoder initialization FAILED with unknown error: {e}", exc_info=True)

    if not initialization_passed:
         logger.error("Skipping FluxViT encoding test due to initialization failure.")
         return False # Indicate test failure
         
    # --- Test Encoding ---
    encoding_passed = False
    try:
        logger.info(f"Attempting FluxViT encoding using video: {REAL_VIDEO_PATH}")
        if not os.path.exists(REAL_VIDEO_PATH):
            logger.error(f"Test video not found at {REAL_VIDEO_PATH}. Skipping encoding.")
            raise FileNotFoundError(f"Test video not found: {REAL_VIDEO_PATH}")
            
        embedding_list = encoder.encode(REAL_VIDEO_PATH)
        
        if embedding_list is not None and isinstance(embedding_list, list):
            if embedding_list: # Check if the list is not empty
                logger.info(f"FluxViT encoding returned a list of {len(embedding_list)} embedding(s).")
                first_embedding = embedding_list[0]
                assert isinstance(first_embedding, torch.Tensor)
                # Encoder class itself defaults to cuda if available, but individual encoders might send to cpu after processing.
                # The FluxViTEncoder.encode_video currently returns embeddings on CPU.
                assert str(first_embedding.device) == 'cpu' 
                logger.info(f"First FluxViT embedding - Shape: {first_embedding.shape}, Device: {first_embedding.device}, dtype: {first_embedding.dtype}")
                encoding_passed = True
            else:
                logger.info("FluxViT encoding returned an empty list.")
                # Decide if an empty list is a pass or fail for your test case
                # For now, assume non-empty is expected for a valid video and model
                encoding_passed = False 
        elif embedding_list is None:
            logger.error("FluxViT encoding returned None unexpectedly for a real video.")
        else:
            logger.error(f"FluxViT encoding returned an unexpected type: {type(embedding_list)}")
            
    except FileNotFoundError as e:
         logger.error(f"FluxViT encoding failed: {e}") # Log specific file error
    except Exception as e:
        logger.error(f"FluxViT encode method FAILED unexpectedly: {e}", exc_info=True)

    logger.info(f"--- FluxViT Test Complete (Success: {initialization_passed and encoding_passed}) ---")
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
        #results["jepa"] = test_jepa()
        #results["hiera"] = test_hiera()
        results["fluxvit"] = test_fluxvit()
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