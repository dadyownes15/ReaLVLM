"""
Load all embedding tasks from config/embedding_task/

EmbeddingTask:
    - dataset_name
    - model_file_name

    
Go through through all embedding. Check if the embedding_checkpoints exist, and act accordingly.
    - Load the encoder
    - Load the dataset
    - Run the encoder
    - Save the encodings
        - Ensure that the encodings are saved in checkpoints with good print statements for updates
"""