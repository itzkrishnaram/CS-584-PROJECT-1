import numpy as np
from scipy import linalg

class LassoHomotopyModel:
    def __init__(self, max_iter=1000, tol=1e-6, lambda_min_ratio=1e-6, random_state=None):
        """
        Initialize the LASSO Homotopy model.

        Parameters:
        - max_iter: Maximum number of iterations.
        - tol: Tolerance for stopping criterion.
        - lambda_min_ratio: Minimum lambda ratio for regularization path.
        - random_state: Seed for reproducibility.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_min_ratio = lambda_min_ratio
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        """
        Fit the LASSO model using a homotopy approach.

        Parameters:
        - X: Feature matrix (n_samples, n_features).
        - y: Target values (n_samples, ).

        Returns:
        - LassoHomotopyResults object containing coefficients and lambda path.
        """
        # Convert input data to NumPy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape  # Number of samples and features

        # Compute initial correlation between features and target
        correlation = np.abs(X.T @ y)
        lambda_max = np.max(correlation)  # Maximum lambda (largest absolute correlation)
        lambda_min = lambda_max * self.lambda_min_ratio  # Minimum lambda based on the ratio

        # Initialize coefficients and active set
        beta = np.zeros(n_features)  # Start with all coefficients as zero
        active_set = []  # Indices of active features
        active_signs = []  # Signs of active features

        lambda_current = lambda_max  # Start at maximum lambda
        lambda_path = [lambda_current]  # Store lambda values for tracking
        coef_path = [beta.copy()]  # Store coefficient paths

        for _ in range(self.max_iter):
            # Compute residuals and correlations
            residual = y - X @ beta
            correlation = X.T @ residual

            # If no active set, pick the feature with highest correlation
            if not active_set:
                j = np.argmax(np.abs(correlation))
                active_set.append(j)
                active_signs.append(np.sign(correlation[j]))

            # Extract active features and corresponding signs
            X_active = X[:, active_set]
            signs = np.array(active_signs)

            try:
                # Compute direction using pseudo-inverse
                gram_matrix = X_active.T @ X_active + np.eye(len(active_set)) * self.tol
                direction = linalg.pinv(gram_matrix) @ signs

                # Compute update direction for coefficients
                delta_beta = np.zeros(n_features)
                for i, idx in enumerate(active_set):
                    delta_beta[idx] = direction[i]

                # Compute correlation change
                delta_correlation = X.T @ (X @ delta_beta)

                # Compute step sizes for entering/exiting active set
                lambda_gamma = []
                for j in range(n_features):
                    if j not in active_set and abs(delta_correlation[j]) > self.tol:
                        gamma1 = (lambda_current - correlation[j]) / delta_correlation[j]
                        gamma2 = (lambda_current + correlation[j]) / delta_correlation[j]
                        if gamma1 > 0:
                            lambda_gamma.append((gamma1, j, 1))
                        if gamma2 > 0:
                            lambda_gamma.append((gamma2, j, -1))

                beta_gamma = []
                for i, idx in enumerate(active_set):
                    if delta_beta[idx] * active_signs[i] < 0:
                        gamma = -beta[idx] / delta_beta[idx]
                        if gamma > 0:
                            beta_gamma.append((gamma, i, 0))

                # Determine the smallest valid step
                gamma_list = lambda_gamma + beta_gamma
                if not gamma_list:
                    # If no valid step, take a small step to decrease lambda
                    small_step = lambda_current * 0.1
                    beta_new = beta + small_step * delta_beta
                    if np.max(np.abs(beta_new - beta)) < self.tol:
                        break  # Stop if change is too small
                    else:
                        lambda_current -= small_step
                        beta = beta_new
                else:
                    # Take the smallest step to update beta
                    min_gamma, min_idx, min_type = min(gamma_list)
                    beta += min_gamma * delta_beta
                    lambda_current -= min_gamma

                    # Update active set based on step type
                    if min_type == 0:
                        active_set.pop(min_idx)  # Remove feature from active set
                        active_signs.pop(min_idx)
                    else:
                        active_set.append(min_idx)  # Add feature to active set
                        active_signs.append(min_type)

                # Store lambda path and coefficients for analysis
                lambda_path.append(lambda_current)
                coef_path.append(beta.copy())

                # Stop when lambda reaches minimum
                if lambda_current <= lambda_min:
                    break

            except np.linalg.LinAlgError:
                break  # Stop if matrix is singular

        # Store final model parameters
        self.coef_ = beta
        self.active_set_ = active_set
        self.lambda_path_ = np.array(lambda_path)
        self.coef_path_ = np.array(coef_path)

        return LassoHomotopyResults(self)


class LassoHomotopyResults:
    def __init__(self, model):
        """
        Store results from LASSO Homotopy model.

        Parameters:
        - model: The trained LassoHomotopyModel instance.
        """
        self.coef_ = model.coef_
        self.lambda_path_ = model.lambda_path_

    def predict(self, X):
        """
        Make predictions using the fitted model.

        Parameters:
        - X: Feature matrix (n_samples, n_features).

        Returns:
        - Predicted target values.
        """
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_