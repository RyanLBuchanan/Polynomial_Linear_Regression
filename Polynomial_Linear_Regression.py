# Polynomial Linear Regression tutorial from Machine Learning A-Z - SuperDataScience
# Input by Ryan L Buchanan 17SEP20

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

# Train the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Train the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualize the Linear Regression results
plt.scatter(X, y, color='red')
plt.legend()
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.plot(X, lin_reg.predict(X), color = 'blue', label='blaaah')
plt.show()

# Visualize the Linear Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predict a new result with Linear Regression
print("Predicted salary = $", lin_reg.predict([[6.5]]))

# Predict a new result with Polynomial Linear Regression
print("Predicted salary = $", lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))