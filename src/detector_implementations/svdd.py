import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class SVDD:
    """
    Support Vector Data Description (SVDD)
    Implementation based on the primal problem formulation.
    """
    def __init__(self, nu=0.1, kernel=None, **kernel_params):
        """
        Args:
            nu (float): Hyperparameter v in (0, 1], controlling the trade-off between
                        the sphere volume and the number of outliers.
            kernel (str, optional): If specified, indicates the type of kernel to use.
                                    For primal, we typically assume features are pre-computed.
                                    If kernel is used, X_train would be original data, and a
                                    kernel matrix K would be computed or phi(X) directly.
                                    This implementation will focus on pre-computed features first.
            **kernel_params: Parameters for the kernel if one were used.
        """
        if not (0 < nu <= 1):
            raise ValueError("Hyperparameter nu (v) must be in (0, 1]")
        self.nu = nu
        self.R_squared = None # R^2
        self.center = None    # c
        self.support_vectors_ = None # Data points on or outside the boundary (after training)
        self.decision_scores_ = None # Distances from the center
        self.n_samples = None
        self.n_features = None

        # For simplicity with the primal, we assume features are already mapped (phi_k(x_i))
        # If kernel trick were used, self.kernel and self.kernel_params would be relevant here.
        self.kernel = kernel
        self.kernel_params = kernel_params
        if self.kernel:
            logger.warning("Kernel specified, but this SVDD primal implementation expects pre-computed feature vectors."
                           " Ensure input X to fit() is already in feature space or modify to use kernel trick.")

    def _objective_function(self, params, X):
        """
        Objective function: R^2 + (1/(v*n)) * sum(xi_i)
        params = [R^2, c_1, ..., c_d, xi_1, ..., xi_n]
        X = feature vectors (phi_k(x_i))
        """
        n_samples, n_features = X.shape
        R_squared = params[0]
        # center_c = params[1 : 1 + n_features] # Not needed directly in objective
        xi_values = params[1 + n_features :]

        term_sum_xi = np.sum(xi_values)
        objective = R_squared + (1.0 / (self.nu * n_samples)) * term_sum_xi
        return objective

    def _constraints(self, params, X):
        """
        Constraints:
        1. ||phi_k(x_i) - c||^2 - R^2 - xi_i <= 0  (for each i)
        2. -xi_i <= 0                            (for each i)
        3. -R_squared <=0                         (R_squared must be non-negative)
        params = [R^2, c_1, ..., c_d, xi_1, ..., xi_n]
        """
        n_samples, n_features = X.shape
        R_squared = params[0]
        center_c = params[1 : 1 + n_features]
        xi_values = params[1 + n_features :]

        constraints = []

        # Constraint 1: ||phi(x_i) - c||^2 - R^2 - xi_i <= 0
        for i in range(n_samples):
            dist_sq = np.sum((X[i] - center_c)**2)
            constraints.append(dist_sq - R_squared - xi_values[i])
        
        # Constraint 2: -xi_i <= 0  (xi_i >= 0)
        for i in range(n_samples):
            constraints.append(-xi_values[i])
            
        # Constraint 3: -R_squared <= 0 (R_squared >= 0)
        # R_squared is the first parameter
        constraints.append(-R_squared)
        
        # Return as a list of dictionaries for scipy.optimize.minimize
        # In scipy, constraints of the form g(x) >= 0 are used for 'ineq'
        # So, if we have h(x) <= 0, we use -h(x) >= 0.
        # Here, the constraints are already in the form g_j(params) <= 0
        # So we return them as is, and scipy will handle it as type 'ineq' implies g(x) >= 0.
        # To match scipy's expectation of g(x) >= 0, we need to return - (constraint value)
        # if the original constraint is constraint_value <= 0.
        # Or, more simply, we define them as h_j(params) and scipy expects h_j(params) >= 0 for 'ineq'
        # So if our constraints are g_j <= 0, we need to express them as -g_j >= 0.
        
        # Corrected constraint formulation for scipy: g(x) >= 0
        scipy_constraints = []
        # Constraint 1: R^2 + xi_i - ||phi(x_i) - c||^2 >= 0
        for i in range(n_samples):
            dist_sq = np.sum((X[i] - center_c)**2)
            scipy_constraints.append(R_squared + xi_values[i] - dist_sq)
        
        # Constraint 2: xi_i >= 0
        for i in range(n_samples):
            scipy_constraints.append(xi_values[i])
        
        # Constraint 3: R_squared >= 0
        scipy_constraints.append(R_squared)
        
        return np.array(scipy_constraints) # Scipy expects an array for multiple constraints

    def fit(self, X):
        """
        Fit the SVDD model to the data X.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
                            Assumed to be already in the feature space (phi_k(x_i)).
        """
        self.n_samples, self.n_features = X.shape
        if self.n_samples == 0:
            raise ValueError("Input data X cannot be empty.")
            
        logger.info(f"Fitting SVDD to {self.n_samples} samples with {self.n_features} features.")

        # Parameters to optimize: [R^2, c_1, ..., c_d, xi_1, ..., xi_n]
        # Total number of parameters = 1 (for R^2) + n_features (for c) + n_samples (for xi_i)
        num_params = 1 + self.n_features + self.n_samples

        # Initial guess for parameters
        # R^2: Initial guess could be related to variance or just a small positive number
        initial_R_squared = 1.0 
        # c: Initial guess can be the mean of the data
        initial_c = np.mean(X, axis=0)
        # xi_i: Initial guess can be zeros (assuming most points are initially inside)
        initial_xi = np.zeros(self.n_samples)

        initial_params = np.concatenate(([initial_R_squared], initial_c, initial_xi))

        # Define constraints for scipy.optimize.minimize
        # Constraints are of the form: constr_func(params) >= 0
        constraints_spec = ({
            'type': 'ineq',
            'fun': self._constraints, 
            'args': (X,)
        })

        # Bounds for parameters (optional but good practice)
        # R^2 >= 0, xi_i >= 0. Center c is unbounded.
        bounds = [(0, None)] + [(None, None)] * self.n_features + [(0, None)] * self.n_samples

        logger.info("Starting optimization...")
        result = minimize(
            self._objective_function,
            initial_params,
            args=(X,),
            method='SLSQP', # Sequential Least Squares Programming, good for constrained problems
            bounds=bounds,
            constraints=constraints_spec,
            options={'disp': True, 'maxiter': 1000, 'ftol': 1e-7} # Adjust options as needed
        )

        if result.success:
            logger.info(f"Optimization successful: {result.message}")
            optimized_params = result.x
            self.R_squared = optimized_params[0]
            self.center = optimized_params[1 : 1 + self.n_features]
            # xi values are also available if needed: optimized_params[1 + self.n_features:]
            
            # Calculate decision scores (squared distance from center - R^2)
            self.decision_scores_ = np.array([np.sum((X[i] - self.center)**2) for i in range(self.n_samples)]) - self.R_squared

            # Identify support vectors (points for which xi_i > some_tolerance or on boundary)
            # For primal form, this is less direct than dual. Often, points close to boundary are considered.
            # A simple heuristic: points whose squared distance is close to R^2
            # More accurately, using xi_values: params[1+self.n_features:]
            xi_optimized = optimized_params[1 + self.n_features:]
            # Support vectors could be those with xi > small_tolerance or those whose distance constraint is active.
            # For simplicity, let's use decision scores near 0 as potential support vectors
            # or points where slack was used.
            # dist_sq_to_center = np.array([np.sum((X[i] - self.center)**2) for i in range(self.n_samples)])
            # on_boundary_or_outside = dist_sq_to_center >= self.R_squared - 1e-5 # Tolerance for floating point
            # self.support_vectors_ = X[on_boundary_or_outside]
            # logger.info(f"Identified {len(self.support_vectors_)} potential support vectors.")
            
            # Using slack variables to find support vectors more directly
            # Points for which the constraint ||phi(x_i) - c||^2 <= R^2 + xi_i is active or xi_i > 0
            dist_sq = np.array([np.sum((X[i] - self.center)**2) for i in range(self.n_samples)])
            # Constraint: dist_sq - R_squared - xi_optimized <= 0
            # Active if dist_sq - R_squared - xi_optimized is close to 0
            # Or if xi_optimized > epsilon
            support_indices = np.where((np.abs(dist_sq - self.R_squared - xi_optimized) < 1e-5) | (xi_optimized > 1e-5))[0]
            self.support_vectors_ = X[support_indices]
            logger.info(f"Identified {len(self.support_vectors_)} support vectors (indices: {support_indices}).")

        else:
            logger.error(f"Optimization failed: {result.message}")
            # Handle failure: maybe raise an error or set a flag
            raise RuntimeError(f"SVDD optimization failed: {result.message}")

    def predict(self, X):
        """
        Predict labels for X (1 for inlier/normal, -1 for outlier/anomalous).
        Also computes decision scores (distance from boundary).

        Args:
            X (np.ndarray): Data to predict, shape (n_samples, n_features).
                            Assumed to be already in the feature space.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - labels: -1 for outliers, 1 for inliers.
                - decision_scores: Squared distance to center minus R^2. 
                                   Negative for inliers, positive for outliers.
        """
        if self.center is None or self.R_squared is None:
            raise RuntimeError("SVDD model has not been fitted yet. Call fit() first.")
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"Input data X has {X.shape[1]} features, but model was fitted with {self.n_features}.")

        # Calculate squared Euclidean distance from the center c
        # decision_scores_ = sum_i (X_i - c_i)^2 - R^2
        decision_scores = np.array([np.sum((X[i] - self.center)**2) for i in range(X.shape[0])]) - self.R_squared
        
        # Anomalies are points where distance_sq > R_squared, so decision_score > 0
        labels = np.ones(X.shape[0])
        labels[decision_scores > 0] = -1 # Outliers / Anomalous
        
        return labels, decision_scores

    def get_radius(self):
        return np.sqrt(self.R_squared) if self.R_squared is not None else None

    def get_center(self):
        return self.center

# Example Usage (Illustrative)
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    # Generate some sample data
    # One main cluster (inliers) and some scattered points (outliers)
    n_samples_inliers = 200
    n_samples_outliers = 20
    X_inliers, _ = make_blobs(n_samples=n_samples_inliers, centers=[[2, 2]], cluster_std=0.8, random_state=42)
    X_outliers, _ = make_blobs(n_samples=n_samples_outliers, centers=[[-2, -2]], cluster_std=0.5, random_state=42)
    X_outliers_scatter = np.random.uniform(low=-5, high=5, size=(n_samples_outliers, 2))

    X_train = np.vstack((X_inliers, X_outliers_scatter[:10])) # Mix some outliers into training
    X_test_inliers = X_inliers[int(n_samples_inliers*0.8):]
    X_test_outliers = np.vstack((X_outliers, X_outliers_scatter[10:]))
    X_test = np.vstack((X_test_inliers, X_test_outliers))
    y_test_true = np.concatenate([np.ones(len(X_test_inliers)), -np.ones(len(X_test_outliers))])

    # Scale data (important for distance-based methods)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("--- SVDD Example ---")
    # Initialize and fit SVDD
    svdd_model = SVDD(nu=0.1) # nu around proportion of expected outliers
    try:
        svdd_model.fit(X_train_scaled)
        logger.info(f"SVDD Fitted. Radius: {svdd_model.get_radius():.4f}")
        logger.info(f"Center: {svdd_model.get_center()}")

        # Predict on test data
        labels_pred, scores_pred = svdd_model.predict(X_test_scaled)
        
        accuracy = np.mean(labels_pred == y_test_true)
        logger.info(f"Test Accuracy (1=inlier, -1=outlier): {accuracy:.4f}")

        # Plotting (only for 2D data)
        if X_train_scaled.shape[1] == 2:
            plt.figure(figsize=(10, 8))
            
            # Plot training data
            plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c='lightblue', label='Training Data', s=50, edgecolors='k')
            
            # Plot decision boundary (circle)
            center = svdd_model.get_center()
            radius = svdd_model.get_radius()
            circle = plt.Circle(center, radius, color='red', fill=False, linewidth=2, linestyle='--', label='SVDD Boundary')
            plt.gca().add_artist(circle)
            
            # Plot support vectors
            if svdd_model.support_vectors_ is not None and len(svdd_model.support_vectors_) > 0:
                 plt.scatter(svdd_model.support_vectors_[:, 0], svdd_model.support_vectors_[:, 1], 
                             facecolors='none', edgecolors='red', s=100, linewidth=2, label='Support Vectors')

            # Plot test data predictions
            plt.scatter(X_test_scaled[labels_pred == 1, 0], X_test_scaled[labels_pred == 1, 1], 
                        c='green', marker='o', s=60, edgecolors='k', label='Test Inliers (Pred)')
            plt.scatter(X_test_scaled[labels_pred == -1, 0], X_test_scaled[labels_pred == -1, 1], 
                        c='purple', marker='x', s=60, edgecolors='k', label='Test Outliers (Pred)')

            plt.title(f'SVDD Anomaly Detection (nu={svdd_model.nu})')
            plt.xlabel('Feature 1 (Scaled)')
            plt.ylabel('Feature 2 (Scaled)')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            # Create a dummy directory for the plot if it doesn't exist
            os.makedirs("plots", exist_ok=True)
            plot_path = "plots/svdd_example_primal.png"
            plt.savefig(plot_path)
            logger.info(f"Plot saved to {plot_path}")
            # plt.show() # Uncomment to display plot interactively
            
    except RuntimeError as e:
        logger.error(f"SVDD example failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in SVDD example: {e}", exc_info=True) 