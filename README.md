# Lasso Regression using Homotopy method

This project implements the **Lasso Homotopy Algorithm**, a specialized approach for solving Lasso regression.

## Team Members:

1. Krishna Ram Saravanakumar - A20578833 (ksaravankumar@hawk.iit.edu)
2. Govind Kurapati - A20581868 (gkurapati@hawk.iit.edu)

## Installation

### **1️. Create a Virtual Environment**

```sh
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### **2️. Install Required Dependencies**

```sh
pip install -r requirements.txt
```

---

### 3. Running the Project

Execute the test script:

```sh
pytest ./LassoHomotopy/tests/test_LassoHomotopy.py
```

## Questions:

### 1. What does the implemented model do, and when should it be used?

This model is an implementation of the LASSO regression (Least Absolute Shrinkage and Selection Operator) using the homotopy method.

#### What it does:

• It solves LASSO problems by tracing the solution path as the regularization parameter lambda decreases.
• The model maintains an active set of selected features and iteratively updates the coefficients.
• Unlike standard solvers, it efficiently updates solutions instead of solving from scratch for each lambda.

#### When to use it:

• When you need feature selection (LASSO encourages sparsity, i.e., it forces some coefficients to be exactly zero).
• When solving high-dimensional regression problems with more features than samples.
• When computational efficiency is important, as the homotopy method is faster than naive approaches for a sequence of lambda values.

### 2. How did you test the model to determine if it is working correctly?

The model was tested using:

##### Basic correctness tests:

- Compared against known LASSO solutions (e.g., `sklearn.linear_model.Lasso`).
- Checked if increasing lambda results in sparser coefficients (expected behavior).

##### Numerical stability tests:

- Verified that coefficients converge within the defined tol (tolerance).
- Ensured solutions are stable under small perturbations in input data.

##### Edge cases:

- Tested zero data (all X values are zero) to see if coefficients remain zero.
- Checked highly correlated features to ensure consistent feature selection.

### 3. What parameters have been exposed for performance tuning?

The model exposes the following parameters:

#### `max_iter`:

- Controls how many iterations the algorithm will run before stopping.
- Higher values allow for more accurate solutions but increase computation time.

#### `tol`:

- The tolerance for stopping the algorithm.
- Lower values improve accuracy but may slow down convergence.

#### `lambda_min_ratio`:

- Defines the smallest regularization value relative to lambda_max.
- Controls how far the algorithm follows the solution path.

#### `random_state`:

- Could introduce random feature selection if desired.

### 4. Are there specific inputs that the implementation struggles with?

Yes, the model has difficulty with:

Perfectly correlated features:

- The algorithm struggles to pick between them, causing numerical instability.
- Workaround: To apply feature selection before running the model.

Extremely small lambda:

- Can lead to overfitting and numerical instability in matrix inversions.
- Workaround: To set a reasonable lambda_min_ratio.

Very large feature sets (>10,000 features):

- The Gram matrix inversion in linalg.pinv(gram_matrix) can be expensive.
- Workaround: To use approximate solvers or modify the method for efficiency.

#### Could these issues be solved with more time?

Yes, correlated features can be addressed by using elastic net regularization, which combines LASSO and Ridge penalties to handle multicollinearity more effectively.  
Yes, for datasets with a large number of features, switching to a coordinate descent algorithm can significantly improve scalability and computational efficiency.  
However, if the problem is caused by choosing a very small lambda, it leads to overfitting, which is a fundamental limitation of LASSO and cannot be completely avoided.
