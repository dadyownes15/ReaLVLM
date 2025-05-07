from pathlib import Path
import sys

# Add src to Python path to allow direct import of EmbeddingTask
# Assumes the test script is in the workspace root and src is a subdirectory
workspace_root = Path("/Users/mikkeldahl/Documents/bachelor_thesis/VLLM_oneshot")
sys.path.insert(0, str(workspace_root))

from src.models.embedding_task import EmbeddingTask

def run_jepa_dummy_test():
    """
    Test script to load a JEPA encoder, use dummy data for training and testing,
    and run the embedding generation process.
    """
    model_config_file = workspace_root / "config/embeddings/jepa_vit_huge.k400_384.json"
    # Use the provided dummy_data for both training and test sets
    # The attached dummy_data folder contains test.mp4
    dummy_data_dir = workspace_root / "data/dummy_data"
    dummy_data_dir_test = dummy_data_dir / "test"
    dummy_data_dir_train = dummy_data_dir / "train"

    dataset_output_name = "jepa_on_dummy_data"

    print(f"--- Test Script Started ---")
    print(f"Workspace Root: {workspace_root}")
    print(f"Using Model Config: {model_config_file}")
    print(f"Using Data Directory (for Training & Test): {dummy_data_dir}")

    # Basic path validation
    if not model_config_file.exists():
        print(f"ERROR: Model config file not found at {model_config_file}")
        return
    if not dummy_data_dir.exists() or not dummy_data_dir.is_dir():
        print(f"ERROR: Dummy data directory not found at {dummy_data_dir}")
        return
    if not list(dummy_data_dir.iterdir()): # Check if directory is empty
        print(f"ERROR: Dummy data directory {dummy_data_dir} is empty. Please ensure it contains data (e.g., test.mp4).")
        return

    print("\n--- Initializing EmbeddingTask ---")
    try:
        task = EmbeddingTask(
            model_config_path=str(model_config_file),  # EmbeddingTask expects str path
            dataset_name=dataset_output_name,
            training_data_path=dummy_data_dir_train,
            test_data_path=dummy_data_dir_test
        )
        print("--- EmbeddingTask Initialized Successfully ---")
    except Exception as e:
        print(f"ERROR: Failed to initialize EmbeddingTask: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- Running EmbeddingTask ---")
    try:
        task.run()
        print("--- EmbeddingTask run() method completed ---")
    except Exception as e:
        print(f"ERROR: An error occurred during task.run(): {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- Test Script Finished ---")
    output_location = workspace_root / 'output' / dataset_output_name
    print(f"Output should be generated in: {output_location}")
    if task.encoder_name: # Check if encoder_name was set
      checkpoint_file_path = output_location / task.encoder_name / 'checkpoint.json'
      print(f"Checkpoint file should be at: {checkpoint_file_path}")
    else:
      print("Could not determine checkpoint file path as encoder_name was not set on the task.")


if __name__ == "__main__":
    run_jepa_dummy_test() 