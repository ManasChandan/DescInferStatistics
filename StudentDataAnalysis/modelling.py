import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats

# Generate some example data
np.random.seed(0)
X = np.random.rand(100, 1)  # Independent variable
y = 2 * X + 1 + np.random.randn(100, 1)  # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate the confidence intervals and t-test for coefficients
coef = model.coef_[0]
intercept = model.intercept_

# Calculate standard error of coefficient
y_pred = model.predict(X_train)
residuals = y_train - y_pred
mse = np.mean(residuals**2)
std_err = np.sqrt(mse / (len(X_train) - 2))
std_err_coef = std_err / np.sqrt(np.sum((X_train - np.mean(X_train))**2))

# Calculate t-statistic and p-value for coefficient
t_stat = coef / std_err_coef
p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(X_train) - 2))

# Calculate confidence interval for coefficient
alpha = 0.05  # Significance level
t_critical = stats.t.ppf(1 - alpha / 2, df=len(X_train) - 2)
margin_of_error = t_critical * std_err_coef
confidence_interval_coef = (coef - margin_of_error, coef + margin_of_error)

# Calculate standard error of prediction
std_err_pred = np.sqrt(mse * (1 + 1 / len(X_train) + (X_train - np.mean(X_train))**2 / np.sum((X_train - np.mean(X_train))**2)))

# Calculate t-statistic and p-value for prediction
t_stat_pred = (y_train - y_pred) / std_err_pred
p_value_pred = 2 * (1 - stats.t.cdf(np.abs(t_stat_pred), df=len(X_train) - 2))

# Calculate confidence interval for prediction
margin_of_error_pred = t_critical * std_err_pred
confidence_interval_pred = (y_pred - margin_of_error_pred, y_pred + margin_of_error_pred)

# Calculate predictions on the test data
y_test_pred = model.predict(X_test)

# Calculate standard error of prediction for test data
std_err_pred_test = np.sqrt(mse * (1 + 1 / len(X_train) + (X_test - np.mean(X_train))**2 / np.sum((X_train - np.mean(X_train))**2)))

# Calculate confidence interval for prediction on test data
margin_of_error_pred_test = t_critical * std_err_pred_test
confidence_interval_pred_test = (y_test_pred - margin_of_error_pred_test, y_test_pred + margin_of_error_pred_test)

print("Coefficient:", coef)
print("Intercept:", intercept)
print("T-Statistic (Coefficient):", t_stat)
print("P-Value (Coefficient):", p_value)
print("Confidence Interval (Coefficient):", confidence_interval_coef)
print("T-Statistic (Prediction):", t_stat_pred)
print("P-Value (Prediction):", p_value_pred)
print("Confidence Interval (Prediction):", confidence_interval_pred)
print("Predictions on Test Data:", y_test_pred)
print("Confidence Intervals on Test Data Predictions:", confidence_interval_pred_test)
