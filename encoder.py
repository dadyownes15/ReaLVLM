from src.models.embedding_task import EmbeddingTask
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent # Gets directory of encoder.py
    project_root = script_dir.parent # Assumes encoder.py is in project_root

    print(project_root)
    # If encoder.py is in a 'scripts' subdir, then project_root = script_dir.parent
    # Adjust project_root definition based on where encoder.py lives relative to 'src'


    
    # Embed using hiera_huge_16x224.mae_k400_ft_k400.json
    embedding_task2 = EmbeddingTask(
        model_config_path="D:\ReaLVLM\config\embeddings\hiera_huge_16x224.mae_k400_ft_k400.json",
        dataset_name="shanghai_campus",
        training_data_path="D:/ReaLVLM/data/shanghai_campus/training/videos",
        test_data_path="D:/ReaLVLM/data/shanghai_campus/testing/videos",
    )

    embedding_task2.run()
    

if __name__ == "__main__":
    main()
