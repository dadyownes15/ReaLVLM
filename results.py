import os

import os
from src.models.results import Results


"""
Reults
    - Output: Total number of videos
    - Total number of detected videos
    - Report the list of videos that was not detected

"""

def main():
    ## Load results embeddings
    jepa_vit_huge_test_results_path = "output/scene_1_shanghai_campus/detector_outputs/OneClassSVMDetector/jepa-huge-384"

    results = Results(jepa_vit_huge_test_results_path, ["01_006", "01_001", "01_003", "01_007","01_010", "01_011", "01_012", "01_013"], show_normal=True, show_other=True)
    results.load()

    #results.deepdive(["01_006", "01_001", "01_003", "01_007","01_010", "01_011", "01_012", "01_013"])
    rule = lambda s: (s.min() < 0).item()
    results.results_summary(rule)

    results.results_statistics()
if __name__ == "__main__":
    main()