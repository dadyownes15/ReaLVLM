
from src.models.detector_task import DetectorTask
from src.detector_implementations.SVM import OneClassSVMDetector


def main():
    detector_task = DetectorTask(
        detector_model=OneClassSVMDetector(),
        training_data_path="output/shanghai_campus/non_falls_embeddings",
        test_data_path="output/shanghai_campus/falls_embeddings",
        dataset_name="shanghai_campus"
    )
    detector_task.run()

if __name__ == "__main__":
    main()