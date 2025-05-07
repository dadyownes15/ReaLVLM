import numpy as np
from scipy.optimize import minimize
import logging
import os # Make sure os is imported if used in example

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
            **kernel_params: Parameters for the kernel if one were used.
        """
        if not (0 < nu <= 1):
            raise ValueError("Hyperparameter nu (v) must be in (0, 1]")
        self.nu = nu
        self.R_squared = None # R^2
        self.center = None    # c
        self.support_vectors_ = None 
        self.decision_scores_ = None 
        self.n_samples = None
        self.n_features = None
        self.kernel = kernel
        self.kernel_params = kernel_params


    def _objective_function(self, params, X):
        """
        Objective: R^2 + (1/(v*n)) * sum(epsilon_i)
        params = [R^2, c_1,...,c_d, epsilon_1,...,epsilon_n]
        """
        n_samples, n_features = X.shape # Get n_features from X for correct slicing
        R_squared = params[0]
        # center_c is params[1 : 1 + n_features]
        # epsilon_values are the last n_samples parameters
        Eps = params[1 + n_features : 1 + n_features + n_samples]
        
        if len(Eps) != n_samples:
            logger.error(f"Objective: Mismatch in xi_values length. Expected {n_samples}, got {len(Eps)}. Params length: {len(params)}, n_features: {n_features}")
            # This indicates a fundamental issue with how params are structured or sliced.
            # For debugging, you might raise an error here or return a very large number.
            raise ValueError("Slicing error for xi_values in objective function")

        term_sum_epsilon = np.sum(Eps)
        objective = R_squared + (1.0 / (self.nu * n_samples)) * term_sum_epsilon
        return objective

    def _constraints(self, params, X):
        """
        Constraints (for scipy, g(x) >= 0):
        1. R^2 + epsilon_i - ||phi_k(x_i) - c||^2 >= 0
        2. epsilon_i >= 0
        3. R_squared >= 0
        params = [R^2, c_1,...,c_d, epsilon_1,...,epsilon_n]
        """
        n_samples, n_features = X.shape # Get n_features from X for correct slicing
        R_squared = params[0]
        center_c = params[1 : 1 + n_features]
        # xi_values are the last n_samples parameters
        epsilon_values = params[1 + n_features : 1 + n_features + n_samples]

        if len(epsilon_values) != n_samples:
            logger.error(f"Constraints: Mismatch in epsilon_values length. Expected {n_samples}, got {len(epsilon_values)}. Params length: {len(params)}, n_features: {n_features}")
            raise ValueError("Slicing error for epsilon_values in constraints function")

        # Vectorized Constraint 1
        diff = X - center_c 
        dist_sq_all = np.sum(diff**2, axis=1) 
        constr1 = R_squared + epsilon_values - dist_sq_all

        # Constraint 2
        constr2 = epsilon_values

        # Constraint 3
        constr3 = np.array([R_squared]) 

        all_constraints = np.concatenate((constr1, constr2, constr3))
        return all_constraints

    def _iteration_callback_wrapper(self, X_for_obj_and_constr):
        """Wrapper to create a callback with access to X."""
        self.iter_count = 0 # Initialize iteration counter for each fit call
        def callback(xk):
            """Callback function to be called after each iteration."""
            self.iter_count += 1
            current_objective = self._objective_function(xk, X_for_obj_and_constr)
            # Print every N iterations to avoid too much output, or every iteration if desired
            if self.iter_count % 1 == 0: # Print every 1 iteration, adjust as needed (e.g., % 10)
                logger.info(f"Iter: {self.iter_count:4d}, Objective: {current_objective:.6e}")
            # You could also store xk or objective values for later analysis if needed
        return callback

    def fit(self, X):
        """
        Fit the SVDD model to the data X.
        Args:
            X (np.ndarray): Training data (n_samples, n_features), pre-computed features.
        """
        # Set n_samples and n_features from the input X for this fit call
        self.n_samples, self.n_features = X.shape 
        if self.n_samples == 0:
            raise ValueError("Input data X cannot be empty.")
            
        logger.info(f"Fitting SVDD to {self.n_samples} samples with {self.n_features} features.")

        # Total number of parameters based on current X's shape
        num_params = 1 + self.n_features + self.n_samples

        initial_R_squared = 1.0 
        initial_c = np.mean(X, axis=0)
        initial_eps_i = np.zeros(self.n_samples)
        initial_params = np.concatenate(([initial_R_squared], initial_c, initial_eps_i))
        
        # Check initial_params length
        if len(initial_params) != num_params:
            raise ValueError(f"Length of initial_params ({len(initial_params)}) does not match expected num_params ({num_params})")

        constraints_spec = ({
            'type': 'ineq',
            'fun': self._constraints, 
            'args': (X,)
        })

        # A bit weird notation, but simply means that we create a  tuple for all parameters, and define lower (left) and upper bound (right)
        bounds = [(0, None)] + [(None, None)] * self.n_features + [(0, None)] * self.n_samples
        # Create the callback function instance for this fit call
        # This ensures X is correctly scoped for the callback
        iter_callback_func = self._iteration_callback_wrapper(X)

        logger.info("Starting optimization...")
        result = minimize(
            self._objective_function,
            initial_params,
            args=(X,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_spec,
            options={'disp': True, 'maxiter': 1000, 'ftol': 1e-5}, # User set maxiter to 20
            callback=iter_callback_func # Add the callback here
        )

        if result.success:
            logger.info(f"Optimization successful: {result.message}")
            optimized_params = result.x
            self.R_squared = optimized_params[0]
            self.center = optimized_params[1 : 1 + self.n_features]
            xi_optimized = optimized_params[1 + self.n_features:]
            
            self.decision_scores_ = np.array([np.sum((X[i] - self.center)**2) for i in range(self.n_samples)]) - self.R_squared
            
            dist_sq = np.sum((X - self.center)**2, axis=1)
            support_indices = np.where((np.abs(dist_sq - self.R_squared - xi_optimized) < 1e-5) | (xi_optimized > 1e-5))[0]
            self.support_vectors_ = X[support_indices]
            logger.info(f"Identified {len(self.support_vectors_)} support vectors.")
        else:
            logger.error(f"Optimization failed: {result.message}")
            # Store partial results if any, for debugging, but don't raise error if it's just maxiter
            if 'maximum number of iterations' not in result.message.lower():
                 raise RuntimeError(f"SVDD optimization failed: {result.message}")
            else: # Max iterations reached, store what we have
                logger.warning("Max iterations reached. Model may not be fully converged.")
                optimized_params = result.x
                self.R_squared = optimized_params[0]
                self.center = optimized_params[1 : 1 + self.n_features]
                # Decision scores and support vectors might be less reliable here
                self.decision_scores_ = np.array([np.sum((X[i] - self.center)**2) for i in range(self.n_samples)]) - self.R_squared

    def predict(self, X):
        """
        Predict labels for X (1 for inlier/normal, -1 for outlier/anomalous).
        Args:
            X (np.ndarray): Data to predict (n_samples, n_features).
        Returns:
            Tuple[np.ndarray, np.ndarray]: labels, decision_scores
        """
        if self.center is None or self.R_squared is None:
            # If R_squared is 0.0 (can happen if maxiter is too low and init R_sq was 0), treat as not fitted for predict
            if self.R_squared == 0.0 and self.center is not None: 
                 logger.warning("Model R_squared is 0.0, predict might be unreliable. Was fit() truncated early?")
            else:
                 raise RuntimeError("SVDD model has not been fitted or converged properly. Call fit() first.")
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"Input data X has {X.shape[1]} features, but model was fitted with {self.n_features}.")

        decision_scores = np.sum((X - self.center)**2, axis=1) - self.R_squared
        labels = np.ones(X.shape[0])
        labels[decision_scores > 1e-7] = -1 # Add small tolerance for strict inequality
        return labels, decision_scores

    def get_radius(self):
        return np.sqrt(self.R_squared) if self.R_squared is not None and self.R_squared >= 0 else None

    def get_center(self):
        return self.center

# Example Usage (Illustrative)
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt # Ensure matplotlib is imported for the example
    import os # Ensure os is imported for makedirs

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
       
            
    except RuntimeError as e:
        logger.error(f"SVDD example failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in SVDD example: {e}", exc_info=True) 