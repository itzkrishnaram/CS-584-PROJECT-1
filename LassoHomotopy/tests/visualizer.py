import numpy as np
import matplotlib.pyplot as plt
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from test_LassoHomotopy import load_csv_data  # Ensure test_script.py has the load_csv_data function
from model.LassoHomotopy import LassoHomotopyModel
from sklearn.linear_model import Lasso  # Scikit-learn's Lasso model
# Define the file path for the test dataset
TEST_FILE = os.path.join(ROOT_DIR, "tests", "small_test.csv")

# Load test data
X, y = load_csv_data(TEST_FILE)

# ---------- Your Model ----------
my_model = LassoHomotopyModel()
my_results = my_model.fit(X, y)
my_preds = my_results.predict(X)

# ---------- Scikit-Learn Model ----------
sklearn_model = Lasso(alpha=1.0)  # Adjust alpha if needed
sklearn_model.fit(X, y)
sklearn_preds = sklearn_model.predict(X)

# Define the save location for the plot
save_dir = os.path.join(ROOT_DIR, "plots")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "comparison_plot.png")

# ---------- Plot the Predictions ----------
plt.figure(figsize=(10, 5))
plt.plot(y, label="Actual Target (y)", linestyle='-', marker='o', color='blue')
plt.plot(my_preds, label="My Model Prediction", linestyle='--', marker='s', color='red')
plt.plot(sklearn_preds, label="Scikit-Learn Lasso Prediction", linestyle='-.', marker='x', color='green')

plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.title("Comparison: Lasso Homotopy vs. Scikit-Learn Lasso")
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(save_path)
print(f"Comparison plot saved at: {save_path}")

# Show plot 
plt.show()