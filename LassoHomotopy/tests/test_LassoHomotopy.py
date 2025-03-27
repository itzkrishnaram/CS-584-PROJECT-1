import csv
import os
import numpy as np
import pytest
from model.LassoHomotopy import LassoHomotopyModel

# File paths for test datasets
SMALL_TEST_FILE = "/Users/krishnaramsaravanakumar/Desktop/Krishnaram/CODING/ML/CS-584-PROJECT-1/LassoHomotopy/tests/small_test.csv"
COLLINEAR_TEST_FILE = "/Users/krishnaramsaravanakumar/Desktop/Krishnaram/CODING/ML/CS-584-PROJECT-1/LassoHomotopy/tests/collinear_data.csv"

def load_csv_data(filename):
    """Loads CSV data and returns feature matrix X and target vector y."""
    data = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append({k: float(v) for k, v in row.items()})

    # Extract features (starting with 'x' or 'X')
    X = np.array([[v for k, v in datum.items() if k.lower().startswith('x')] for datum in data])

    # Extract target ('y' or 'target')
    if 'y' in data[0]:
        y = np.array([datum['y'] for datum in data])
    elif 'target' in data[0]:
        y = np.array([datum['target'] for datum in data])
    else:
        raise KeyError("Neither 'y' nor 'target' column found in the data")

    return X, y

@pytest.fixture(scope="module")
def small_test_data():
    """Fixture to load small test dataset once for all test cases."""
    return load_csv_data(SMALL_TEST_FILE)

@pytest.fixture(scope="module")
def collinear_test_data():
    """Fixture to load collinear test dataset once for all test cases."""
    return load_csv_data(COLLINEAR_TEST_FILE)

def test_model_runs_without_error(small_test_data):
    """Ensures the model can train without errors."""
    X, y = small_test_data
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    assert results is not None, "Model should return a valid result object."

def test_model_learns_nonzero_coefficients(small_test_data):
    """Checks that the model learns at least some nonzero coefficients."""
    X, y = small_test_data
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    assert np.any(results.coef_ != 0), "Model should learn non-zero coefficients."

def test_model_sparse_solution_with_collinear_data(collinear_test_data):
    """Ensures that LASSO produces a sparse solution when data is collinear."""
    X, y = collinear_test_data
    model = LassoHomotopyModel(lambda_min_ratio=1e-5)
    results = model.fit(X, y)

    zero_coefs = np.sum(np.abs(results.coef_) < 1e-10)
    assert zero_coefs > 0, "LASSO should produce sparse solutions with collinear data."

def test_model_prediction_shape(small_test_data):
    """Ensures that predictions match the shape of the target variable."""
    X, y = small_test_data
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    preds = results.predict(X)
    assert preds.shape == y.shape, "Predictions should have the same shape as y."

def test_lambda_effect_on_sparsity(small_test_data):
    """Tests that higher lambda results in more zero coefficients."""
    X, y = small_test_data
    model_high_reg = LassoHomotopyModel(lambda_min_ratio=0.5)
    results_high_reg = model_high_reg.fit(X, y)

    model_low_reg = LassoHomotopyModel(lambda_min_ratio=1e-6)
    results_low_reg = model_low_reg.fit(X, y)

    high_reg_zeros = np.sum(np.abs(results_high_reg.coef_) < 1e-10)
    low_reg_zeros = np.sum(np.abs(results_low_reg.coef_) < 1e-10)

    assert high_reg_zeros >= low_reg_zeros, "Higher lambda should result in more zero coefficients."

def test_model_stability_on_repeated_runs(small_test_data):
    """Checks if the model produces consistent results when trained multiple times."""
    X, y = small_test_data
    model = LassoHomotopyModel()
    results_1 = model.fit(X, y)
    results_2 = model.fit(X, y)
    assert np.allclose(results_1.coef_, results_2.coef_), "Model coefficients should be consistent across runs."

def test_model_with_all_zero_features():
    """Tests if the model can handle all-zero features without crashing."""
    X = np.zeros((100, 10))  # 100 samples, 10 zero-valued features
    y = np.random.randn(100)
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    assert np.all(results.coef_ == 0), "Model should output zero coefficients for zero-valued features."

def test_model_with_high_dimensional_data():
    """Tests if the model can handle high-dimensional data where features >> samples."""
    X = np.random.randn(20, 100)  # 20 samples, 100 features
    y = np.random.randn(20)
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    assert results.coef_.shape[0] == 100, "Model should handle high-dimensional data properly."

def test_random_seed_impact(small_test_data):
    """Checks if model output is consistent regardless of random seed."""
    X, y = small_test_data
    model_1 = LassoHomotopyModel(random_state=42)
    results_1 = model_1.fit(X, y)

    model_2 = LassoHomotopyModel(random_state=43)  # Different seed
    results_2 = model_2.fit(X, y)

    # Expect results to be identical since there's no randomness in the model
    assert np.allclose(results_1.coef_, results_2.coef_), "Results should be identical since model is deterministic."


'''''
def test_random_seed_impact(small_test_data):
    """Checks if model output is sensitive to different random seeds."""
    X, y = small_test_data
    model_1 = LassoHomotopyModel(random_state=42)
    model_2 = LassoHomotopyModel(random_state=99)
    
    results_1 = model_1.fit(X, y)
    results_2 = model_2.fit(X, y)

    assert not np.allclose(results_1.coef_, results_2.coef_), "Different random seeds should yield different results."
'''
def test_model_output_consistency_on_same_input(small_test_data):
    """Ensures the model produces the same predictions given the same input multiple times."""
    X, y = small_test_data
    model = LassoHomotopyModel()
    results = model.fit(X, y)
    
    preds_1 = results.predict(X)
    preds_2 = results.predict(X)

    assert np.allclose(preds_1, preds_2), "Predictions should be consistent on repeated runs."