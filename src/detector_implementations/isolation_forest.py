import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

# Assuming the Detector base class is in src.models.detector_model
from models.detector import Detector

class IsolationForestDetector(Detector):
    """
    Detector implementation using scikit-learn's IsolationForest.
    """
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto', 
                 max_features=1.0, bootstrap=False, n_jobs=None, random_state=None, 
                 verbose=0, warm_start=False, **kwargs):
        """
        Initializes the IsolationForest detector.

        Args:
            n_estimators (int, optional): The number of base estimators in the ensemble. Defaults to 100.
            max_samples (str or int or float, optional): The number of samples to draw from X to train each base estimator.
                - If int, then draw `max_samples` samples.
                - If float, then draw `max_samples * X.shape[0]` samples.
                - If 'auto', then `max_samples=min(256, n_samples)`.
                Defaults to 'auto'.
            contamination (str or float, optional): The amount of contamination of the data set, i.e. the proportion
                of outliers in the data set. Used when fitting to define the threshold on the scores of the samples.
                - If 'auto', the threshold is determined as in the original paper.
                - If float, the contamination should be in the range (0, 0.5].
                Defaults to 'auto'.
            max_features (int or float, optional): The number of features to draw from X to train each base estimator.
                - If int, then draw `max_features` features.
                - If float, then draw `max_features * X.shape[1]` features.
                Defaults to 1.0.
            bootstrap (bool, optional): If True, individual trees are fit on random subsets of the training data
                sampled with replacement. If False, sampling without replacement is performed. Defaults to False.
            n_jobs (int, optional): The number of jobs to run in parallel for both `fit` and `predict`.
                ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                ``-1`` means using all processors. Defaults to None.
            random_state (int, RandomState instance or None, optional): Controls the pseudo-randomness of         
                the selection of the feature and split values for each branching step and each tree in the forest.
                Pass an int for reproducible results across multiple function calls. Defaults to None.
            verbose (int, optional): Controls the verbosity of the tree building process. Defaults to 0.
            warm_start (bool, optional): When set to ``True``, reuse the solution of the previous call to fit
                and add more estimators to the ensemble, otherwise, just fit a whole new forest. Defaults to False.
            **kwargs: Additional keyword arguments passed to the IsolationForest constructor.
        """
        super().__init__()
        self.detector = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            **kwargs
        )
        self.type = "IsolationForest"
        self._is_fitted = False

    def train(self, embeddings: np.ndarray):
        """
        Trains the IsolationForest model.

        Args:
            embeddings (np.ndarray): The training samples. Shape (n_samples, n_features).
        """
        if embeddings is None or embeddings.shape[0] == 0:
            raise ValueError("Embeddings cannot be None or empty for training.")
        if len(embeddings.shape) != 2:
            # Isolation Forest can technically handle 1D data if X.reshape(-1,1) is used.
            # However, typical embedding scenarios involve multi-dimensional features.
            # Forcing 2D for consistency with other detectors.
            raise ValueError(f"Embeddings must be 2D (n_samples, n_features), but got shape {embeddings.shape}")

        self.detector.fit(embeddings)
        self._is_fitted = True
        
    def calculate_score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculates the anomaly score for the given embeddings.
        For IsolationForest, this is the anomaly score as computed by the `decision_function` method.
        Lower scores indicate more anomalous.

        Args:
            embeddings (np.ndarray): The input samples. Shape (n_samples, n_features).

        Returns:
            np.ndarray: The anomaly scores. Lower scores are more anomalous.
        
        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If embeddings are None, empty, or not 2D.
        """
        if not self._is_fitted:
            raise RuntimeError("The detector has not been fitted yet. Call train() first.")
        if embeddings is None or embeddings.shape[0] == 0:
            raise ValueError("Embeddings cannot be None or empty for scoring.")
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be 2D (n_samples, n_features), but got shape {embeddings.shape}")
            
        # decision_function returns the anomaly score of each sample.
        # Negative scores are typically considered outliers. The lower the score, the more abnormal.
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
            ValueError: If embeddings are None, empty, or not 2D.
        """
        if not self._is_fitted:
            raise RuntimeError("The detector has not been fitted yet. Call train() first.")
        if embeddings is None or embeddings.shape[0] == 0:
            raise ValueError("Embeddings cannot be None or empty for detection.")
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be 2D (n_samples, n_features), but got shape {embeddings.shape}")

        return self.detector.predict(embeddings)

    def calculate_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        This method is often redundant if calculate_score is already well-defined.
        Kept for consistency with a potential base class structure that might differentiate them.
        It simply calls calculate_score.

        Args:
            embeddings (np.ndarray): The input samples. Shape (n_samples, n_features).

        Returns:
            np.ndarray: Anomaly scores for the embeddings, as returned by `calculate_score`.
        
        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If embeddings are problematic (handled by `calculate_score`).
        """
        # This method might seem redundant given calculate_score. 
        # It's kept if the base 'Detector' class or other interfaces expect it.
        # The primary scoring logic is in calculate_score.
        return self.calculate_score(embeddings)

    def save_model(self, path: str):
        """
        Saves the fitted IsolationForest model to a file.

        Args:
            path (str): The path to save the model file.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model. Call train() first.")
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.detector, path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model to {path}: {e}")
            raise

    def load_model(self, path: str):
        """
        Loads an IsolationForest model from a file.

        Args:
            path (str): The path to the model file.
        """
        try:
            self.detector = joblib.load(path)
            self._is_fitted = True 
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            self._is_fitted = False
            raise 