
from models.results import Results
from src.models.detector_task import DetectorTask
from src.detector_implementations.SVM import OneClassSVMDetector
from models.reducer import PCA, UMAP

def main():
    
    detector_task_jepa = DetectorTask(
        detector_model=OneClassSVMDetector(),
        training_data_path="output/scene_1_shanghai_campus/hiera-huge-224/training",
        test_data_path="output/scene_1_shanghai_campus/hiera-huge-224/test",
        dataset_name="scene_1_shanghai_campus",
        encoder_name="hiera-huge-224",
    )   
    detector_task_jepa.run()
"""     jepa_vit_huge_test_results_path = "output/scene_1_shanghai_campus/detector_outputs/OneClassSVMDetector/jepa-huge-384"
    results = Results(jepa_vit_huge_test_results_path, ["01_006", "01_001", "01_003", "01_007"])
    results.load_scores() """
    

"""     detector_task_hiera = DetectorTask(
        detector_model=OneClassSVMDetector(),
        training_data_path="output/scene_1_shanghai_campus/hiera-huge-224/training",
        test_data_path="output/scene_1_shanghai_campus/hiera-huge-224/test",
        dataset_name="scene_1_shanghai_campus",
        encoder_name="hiera-huge-224"
    )
    detector_task_hiera.run() """
if __name__ == "__main__":
    main()