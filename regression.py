import numpy as np
import numpy.linalg as la
import scipy.stats as st
import matplotlib.pyplot as plt 

def linear_regression(X, y):
    """Calculates linear regression coefficients and related statistics.
    Args:
        X (np.ndarray): Design matrix of shape (n_samples, n_features).
        y (np.ndarray): Target vector of shape (n_samples,).
    Returns:
        tuple: A tuple containing:
            - params (np.ndarray): Regression coefficients.
            - residuals (np.ndarray): Residuals of the regression.
            - MSE (float): Mean squared error.
            - std_err (np.ndarray): Standard errors of the coefficients.
            - t_values (np.ndarray): t-values of the coefficients.
    """
    params = la.lstsq(X, y, rcond=None)[0]
    y_predicted = X @ params
    residuals = y - y_predicted
    MSE = np.sum(residuals**2) / (len(y) - X.shape[1])
    var_covar = MSE * la.inv(X.T @ X)
    std_err = np.sqrt(np.diag(var_covar))
    t_values = params / std_err
    return params, residuals, MSE, std_err, t_values



def confidence_interval_params(params, std_err, X, confidence=0.95):
    """Calculates confidence intervals for regression coefficients.

    Args:
        params (np.ndarray): Regression coefficients.
        std_err (np.ndarray): Standard errors of the coefficients.
        X (np.ndarray): Design matrix.
        confidence (float, optional): Confidence level (default: 0.95).

    Returns:
        np.ndarray: Confidence intervals for the coefficients.
    """
    alpha = 1 - confidence
    df = X.shape[0] - X.shape[1]
    critical_value = st.t.ppf(1 - alpha / 2, df)
    confidence_intervals = np.array([(params[i] - critical_value * std_err[i], 
                                    params[i] + critical_value * std_err[i])
                                    for i in range(len(params))])
    return confidence_intervals

# Example usage:
np.random.seed(0)
X = np.array([[1, x1, x2] for x1, x2 in zip(np.random.rand(100), np.random.rand(100))])
y = np.array([2*x[1] + 3*x[2] + np.random.normal(0, 1, 1) for x in X]).flatten()

params, residuals, MSE, std_err, t_values = linear_regression(X, y)
confidence_intervals = confidence_interval_params(params, std_err, X)
y_predicted = X @ params


print("Coefficients:", params)
print("Standard Errors:", std_err)
print("Confidence Intervals:", confidence_intervals)

plt.scatter(X[:, 1], y, label='Data')
plt.plot(X[:, 1], y_predicted, color='red', label='Regression Line')


plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Confidence Intervals')
plt.show()