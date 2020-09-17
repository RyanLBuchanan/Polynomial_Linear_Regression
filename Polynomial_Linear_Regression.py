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

# Train the Polynomial Regression model on the whole dataset

# Visualize the Linear Regression results

# Visualize the Linear Regression results (for higher resolution and smoother curve)

# Predict a new result with Linear Regression

# Predict a new result with Polynomial Linear Regression