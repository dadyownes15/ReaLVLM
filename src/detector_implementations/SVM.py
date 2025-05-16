import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.svm import OneClassSVM
import os
import torch
from typing import List, Union

# Assuming the Detector base class is in src.models.detector_model
from models.detector import Detector 
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

    def _preprocess_embeddings(self, embeddings: Union[np.ndarray, torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        """
        Converts input embeddings to a 2D NumPy array (n_samples, n_features).
        """
        if isinstance(embeddings, np.ndarray):
            processed_embeddings = embeddings
        elif isinstance(embeddings, torch.Tensor):
            processed_embeddings = embeddings.cpu().detach().numpy() # Ensure CPU and NumPy, detach from graph
        elif isinstance(embeddings, list):
            if not embeddings:
                raise ValueError("Input list of embeddings cannot be empty.")
            if not all(isinstance(e, torch.Tensor) for e in embeddings):
                raise ValueError("All elements in the list must be PyTorch Tensors.")
            
            numpy_arrays = []
            for t in embeddings:
                arr = t.cpu().detach().numpy()
                if arr.ndim == 1: # (D,) -> (1, D)
                    arr = arr.reshape(1, -1)
                elif arr.ndim == 0:
                    raise ValueError(f"0-dimensional tensor found in list: {t}")
                # If arr.ndim > 2 or arr.ndim == 2 and arr.shape[0] > 1 for list elements,
                # np.concatenate will handle it, assuming features match.
                numpy_arrays.append(arr)
            
            try:
                processed_embeddings = np.concatenate(numpy_arrays, axis=0)
            except ValueError as e:
                raise ValueError(f"Error concatenating tensors: {e}. Ensure all embeddings have consistent feature dimensions.")
        else:
            raise TypeError(f"Unsupported embeddings type: {type(embeddings)}. Expected np.ndarray, torch.Tensor, or List[torch.Tensor].")

        # Final validation for 2D shape and non-empty
        if processed_embeddings.ndim == 1:
            processed_embeddings = processed_embeddings.reshape(1, -1) # Single sample, 1D to 2D
        elif processed_embeddings.ndim != 2:
            raise ValueError(f"Processed embeddings must be 2D (n_samples, n_features), but got shape {processed_embeddings.shape}")
        
        if processed_embeddings.shape[0] == 0:
            raise ValueError("Processed embeddings resulted in an empty array (0 samples).")
        if processed_embeddings.shape[1] == 0:
            raise ValueError("Processed embeddings resulted in 0 features.")
            
        return processed_embeddings

    def train(self, embeddings: Union[np.ndarray, torch.Tensor, List[torch.Tensor]]):
        """
        Fits the entire pipeline (reducer → scaler → OneClassSVM) **and** runs a
        cross-validated random grid-search so that the best-performing hyper-
        parameter set is selected automatically.

        The search is *unsupervised*: we maximise the **mean decision-function
        margin** on the validation fold (larger ⇒ samples lie farther inside
        the learned frontier).
        """
        # 1. ---------- prepare data -------------------------------------------------
        X = self._preprocess_embeddings(embeddings)
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("Embeddings cannot be empty after processing.")

        # 2. ---------- build CV splitter -------------------------------------------
        n_splits = min(5, max(2, n_samples))        # handle tiny datasets gracefully
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)

        # 3. ---------- define a surrogate scoring function --------------------------
        #    (mean signed distance to boundary; higher is better)
        def _avg_margin(estimator, X_val, _y=None):
            return np.mean(estimator.decision_function(X_val))

       

        # 4. ---------- search space -------------------------------------------------
        param_dist = {
            "nu":     np.linspace(0.01, 0.25, 25),                # 0.01 … 0.25
            "gamma":  np.logspace(-6, 0, 40),                     # 1e-6 … 1
            "kernel": ["rbf", "sigmoid"],                         # both use γ
        }
        # Feel free to extend this dict (e.g. reducer__n_components) when needed.

        # 5. ---------- run the search ----------------------------------------------
        search = RandomizedSearchCV(
            estimator           = self.detector,   # the full Pipeline
            param_distributions = param_dist,
            n_iter              = 100,              # ← tweak to taste
            scoring             = _avg_margin,
            cv                  = cv,
            n_jobs              = -1,
            verbose             = 1,
            random_state        = 0,
        )
        search.fit(X)                              # unsupervised → no y
        self.detector  = search.best_estimator_    # keep the winner
        self._is_fitted = True

        # (Optional) expose metadata for later inspection
        self.best_params_ = search.best_params_
        self.cv_results_  = search.cv_results_

        print(self.best_params_)
        print(self.detector)
    def calculate_score(self, embeddings: Union[np.ndarray, torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        """
        Calculates the anomaly score for the given embeddings.
        For OneClassSVM, this is the signed distance to the separating hyperplane.

        Args:
            embeddings (Union[np.ndarray, torch.Tensor, List[torch.Tensor]]): 
                The input samples. Can be a 2D NumPy array, a 2D PyTorch Tensor, or a list of PyTorch Tensors.

        Returns:
            np.ndarray: The decision function scores. Higher scores typically mean more normal.
                        Values are negative for outliers and positive for inliers by default
                        after fitting on normal data.
        
        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("The detector has not been fitted yet. Call train() first.")
        
        processed_embeddings = self._preprocess_embeddings(embeddings)
        if processed_embeddings.shape[0] == 0:
            raise ValueError("Embeddings cannot be empty for scoring after processing.")
            
        return self.detector.decision_function(processed_embeddings)

    def detect(self, embeddings: Union[np.ndarray, torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        """
        Predicts whether each sample is an inlier (1) or an outlier (-1).

        Args:
            embeddings (Union[np.ndarray, torch.Tensor, List[torch.Tensor]]): 
                The input samples. Can be a 2D NumPy array, a 2D PyTorch Tensor, or a list of PyTorch Tensors.

        Returns:
            np.ndarray: Prediction labels (1 for inlier, -1 for outlier).
            
        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("The detector has not been fitted yet. Call train() first.")

        processed_embeddings = self._preprocess_embeddings(embeddings)
        if processed_embeddings.shape[0] == 0:
            raise ValueError("Embeddings cannot be empty for detection after processing.")

        return self.detector.predict(processed_embeddings)

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
