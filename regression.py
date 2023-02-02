# Regression
# The process of estimating the relationship between input and output variable
# Regression is performed on continuous data, while classification is performed on discrete data.

# import liberary
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# generate the random datasets
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# sklearn implementation
# model initializing
regression_model = LinearRegression()

# fit the data (train the model)
regression_model.fit(x, y)

# predict
y_predicted = regression_model.predict(x)

# model evaluation
rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)

# printing the value
print('Slope:', regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

# plotting
plt.scatter(x, y, s = 10)
plt.xlabel('x')
plt.ylabel('y')

# predicted values
plt.plot(x, y_predicted, color = 'r')
plt.show()

