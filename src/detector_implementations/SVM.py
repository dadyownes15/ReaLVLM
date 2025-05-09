import numpy as np
from sklearn.svm import OneClassSVM
import os

# Assuming the Detector base class is in src.models.detector_model
from src.models.detector_model import Detector 
# Assuming a utility function to load embeddings from a directory
# This path might need adjustment based on your project structure.
from src.util import load_embeddings_from_dir as _original_load_embeddings_from_dir

class OneClassSVMDetector(Detector):
    """
    Detector implementation using scikit-learn's OneClassSVM.
    """
    def __init__(self, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=1e-3, nu=0.5, 
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1, **kwargs):
        """
        Initializes the OneClassSVM detector.

        Args:
            kernel (str, optional): Specifies the kernel type to be used in the algorithm. 
                                    It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'. 
                                    Defaults to 'rbf'.
            degree (int, optional): Degree of the polynomial kernel function ('poly'). 
                                    Ignored by all other kernels. Defaults to 3.
            gamma (str or float, optional): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
                                          If gamma is 'scale' (default) then 1 / (n_features * X.var()) is used.
                                          If 'auto', uses 1 / n_features.
                                          If float, must be non-negative.
            coef0 (float, optional): Independent term in kernel function. 
                                     It is only significant in 'poly' and 'sigmoid'. Defaults to 0.0.
            tol (float, optional): Tolerance for stopping criterion. Defaults to 1e-3.
            nu (float, optional): An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. 
                                  Should be in the interval (0, 1]. Defaults to 0.5.
            shrinking (bool, optional): Whether to use the shrinking heuristic. Defaults to True.
            cache_size (float, optional): Specify the size of the kernel cache (in MB). Defaults to 200.
            verbose (bool, optional): Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context. Defaults to False.
            max_iter (int, optional): Hard limit on iterations within solver, or -1 for no limit. Defaults to -1.
            **kwargs: Additional keyword arguments passed to the OneClassSVM constructor.
        """
        super().__init__()
        self.detector = OneClassSVM(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            nu=nu,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
            **kwargs
        )
        self.type = "OneClassSVM"
        self._is_fitted = False

    def train(self, embeddings: np.ndarray):
        """
        Trains the OneClassSVM model.

        Args:
            embeddings (np.ndarray): The training samples. Shape (n_samples, n_features).
        """
        if embeddings is None or embeddings.shape[0] == 0:
            raise ValueError("Embeddings cannot be None or empty for training.")
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be 2D (n_samples, n_features), but got shape {embeddings.shape}")

        self.detector.fit(embeddings)
        self._is_fitted = True
        # Store some information from the fitted model if needed
        # For example, number of support vectors:
        # self.n_support_ = self.detector.n_support_ 

    def calculate_score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculates the anomaly score for the given embeddings.
        For OneClassSVM, this is the signed distance to the separating hyperplane.

        Args:
            embeddings (np.ndarray): The input samples. Shape (n_samples, n_features).

        Returns:
            np.ndarray: The decision function scores. Higher scores typically mean more normal.
                        Values are negative for outliers and positive for inliers by default
                        after fitting on normal data.
        
        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("The detector has not been fitted yet. Call train() first.")
        if embeddings is None or embeddings.shape[0] == 0:
            raise ValueError("Embeddings cannot be None or empty for scoring.")
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be 2D (n_samples, n_features), but got shape {embeddings.shape}")
            
        # decision_function returns the signed distance to the separating hyperplane.
        # The sign might depend on the convention (e.g. positive for inliers, negative for outliers)
        # OneClassSVM returns positive for "inliers" and negative for "outliers"
        # To make it an "anomaly score" where higher means more anomalous, we can negate it.
        # However, the base class `Detector` doesn't strictly define the score's direction.
        # Let's stick to raw decision_function output first.
        # Users can interpret scores where more negative = more anomalous.
        return self.detector.decision_function(embeddings)

    def detect(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predicts whether each sample is an inlier (1) or an outlier (-1).

        Args:
            embeddings (np.ndarray): The input samples. Shape (n_samples, n_features).

        Returns:
            np.ndarray: Prediction labels (1 for inlier, -1 for outlier).
            
        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("The detector has not been fitted yet. Call train() first.")
        if embeddings is None or embeddings.shape[0] == 0:
            raise ValueError("Embeddings cannot be None or empty for detection.")
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be 2D (n_samples, n_features), but got shape {embeddings.shape}")

        return self.detector.predict(embeddings)

    def calculate_scores(self, embeddings_dir: str) -> np.ndarray:
        """
        Loads embeddings from a directory and calculates their anomaly scores.

        Args:
            embeddings_dir (str): Path to the directory containing embedding files.

        Returns:
            np.ndarray: Anomaly scores for all loaded embeddings.
            
        Raises:
            RuntimeError: If the model has not been fitted yet.
            FileNotFoundError: If load_embeddings_from_dir is not found or embeddings_dir is invalid.
        """
        if not self._is_fitted:
            raise RuntimeError("The detector has not been fitted yet. Call train() first before calculating scores from a directory.")
        
        # Assuming load_embeddings_from_dir loads and concatenates all embeddings
        # from the given directory into a single NumPy array.
        # This function's specific behavior (e.g., file format, concatenation logic)
        # needs to be defined in src.utils.file_utils.
        try:
            # Use the potentially mocked version if __name__ == '__main__' is running,
            # otherwise, it uses the original import.
            all_embeddings = _original_load_embeddings_from_dir(embeddings_dir)
        except Exception as e:
            # Catching a generic exception if load_embeddings_from_dir fails for any reason
            # (e.g., directory not found, no files, parsing error)
            raise FileNotFoundError(f"Failed to load embeddings from directory '{embeddings_dir}': {e}")

        if all_embeddings is None or all_embeddings.shape[0] == 0:
            raise ValueError(f"No embeddings found or loaded from directory: {embeddings_dir}")
        if len(all_embeddings.shape) != 2:
            raise ValueError(f"Loaded embeddings from directory must be 2D (n_samples, n_features), but got shape {all_embeddings.shape}")

        return self.calculate_score(all_embeddings)

    # Optional: Add methods to save/load the model if needed
    def save_model(self, path: str):
        """
        Saves the fitted OneClassSVM model to a file.

        Args:
            path (str): The path to save the model file.
        """
        import joblib
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model. Call train() first.")
        try:
            joblib.dump(self.detector, path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model to {path}: {e}")
            raise

    def load_model(self, path: str):
        """
        Loads a OneClassSVM model from a file.

        Args:
            path (str): The path to the model file.
        """
        import joblib
        try:
            self.detector = joblib.load(path)
            self._is_fitted = True # Assume loaded model is already fitted
            # self.type is already set in __init__ and should remain "OneClassSVM"
            # No need to call super().__init__() again unless base class state needs reset,
            # which is not typical for loading a detector model.
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            self._is_fitted = False # Ensure model is not marked as fitted if loading fails
            raise

# if __name__ == '__main__': # User has removed this block
    # ... example code previously here ...
