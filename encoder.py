from src.models.embedding_task import EmbeddingTask
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent # Gets directory of encoder.py
    project_root = script_dir.parent # Assumes encoder.py is in project_root

    # If encoder.py is in a 'scripts' subdir, then project_root = script_dir.parent
    # Adjust project_root definition based on where encoder.py lives relative to 'src'

    # Embed using jepa_vit_huge.k400_384.json"
    embedding_task1 = EmbeddingTask(
        model_config_path=project_root / "src/models/config/jepa_vit_huge.k400_384.json",
        dataset_name="shanghai_campus",
        training_data_path=project_root / "src/data/shanghai_campus/train/",
        test_data_path=project_root / "src/data/shanghai_campus/test/",
    )
    
    # Embed using hiera_huge_16x224.mae_k400_ft_k400.json
    embedding_task2 = EmbeddingTask(
        model_config_path="src/models/config/hiera_huge_16x224.mae_k400_ft_k400.json",
        dataset_name="shanghai_campus",
        training_data_path="src/data/shanghai_campus/train/",
        test_data_path="src/data/shanghai_campus/test/",
    )

    embedding_task1.run()
    embedding_task2.run()
    

if __name__ == "__main__":
    main()
