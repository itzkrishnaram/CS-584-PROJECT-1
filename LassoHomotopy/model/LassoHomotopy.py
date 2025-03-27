import numpy as np
from scipy import linalg

class LassoHomotopyModel:
    def __init__(self, max_iter=1000, tol=1e-6, lambda_min_ratio=1e-6,random_state=None):
        """Initialize the LASSO Homotopy model."""
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_min_ratio = lambda_min_ratio
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        """Fit the LASSO model using a homotopy approach."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape
        correlation = np.abs(X.T @ y)
        lambda_max = np.max(correlation)
        lambda_min = lambda_max * self.lambda_min_ratio

        beta = np.zeros(n_features)
        active_set = []
        active_signs = []

        lambda_current = lambda_max
        lambda_path = [lambda_current]
        coef_path = [beta.copy()]

        for _ in range(self.max_iter):
            residual = y - X @ beta
            correlation = X.T @ residual

            if not active_set:
                j = np.argmax(np.abs(correlation))
                active_set.append(j)
                active_signs.append(np.sign(correlation[j]))

            X_active = X[:, active_set]
            signs = np.array(active_signs)

            try:
                gram_matrix = X_active.T @ X_active + np.eye(len(active_set)) * self.tol
                direction = linalg.pinv(gram_matrix) @ signs

                delta_beta = np.zeros(n_features)
                for i, idx in enumerate(active_set):
                    delta_beta[idx] = direction[i]

                delta_correlation = X.T @ (X @ delta_beta)

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

                gamma_list = lambda_gamma + beta_gamma
                if not gamma_list:
                    small_step = lambda_current * 0.1
                    beta_new = beta + small_step * delta_beta
                    if np.max(np.abs(beta_new - beta)) < self.tol:
                        break
                    else:
                        lambda_current -= small_step
                        beta = beta_new
                else:
                    min_gamma, min_idx, min_type = min(gamma_list)
                    beta += min_gamma * delta_beta
                    lambda_current -= min_gamma

                    if min_type == 0:
                        active_set.pop(min_idx)
                        active_signs.pop(min_idx)
                    else:
                        active_set.append(min_idx)
                        active_signs.append(min_type)

                lambda_path.append(lambda_current)
                coef_path.append(beta.copy())

                if lambda_current <= lambda_min:
                    break

            except np.linalg.LinAlgError:
                break

        self.coef_ = beta
        self.active_set_ = active_set
        self.lambda_path_ = np.array(lambda_path)
        self.coef_path_ = np.array(coef_path)

        return LassoHomotopyResults(self)


class LassoHomotopyResults:
    def __init__(self, model):
        """Store results from LASSO Homotopy model."""
        self.coef_ = model.coef_
        self.lambda_path_ = model.lambda_path_

    def predict(self, X):
        """Make predictions using the fitted model."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_