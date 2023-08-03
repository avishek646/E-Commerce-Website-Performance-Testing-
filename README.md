# E-Commerce-Website-Performance-Testing-
Supply Chain Analytics on an E Commerce Website

A minor project on Supply Chain Analytics on an E-Commerce Organization
                                                                                        -By Avishek Chakraborty

Introduction
Supply chain analytics is a valuable part of data-driven decision-making in various industries such as manufacturing, retail, healthcare, and logistics. It is the process of collecting, analyzing and interpreting data related to the movement of products and services from suppliers to customers.
Here is a dataset we collected from a Fashion and Beauty startup. The dataset is based on the supply chain of Makeup products. Below are all the features in the dataset:
Order ID
TPT
Ships Ahead Day Count
Ships Late Day Count
Product ID
Unit Quantity
Weight
We will endeavor to plot a linear regression line between unit quantity and weight to find the optimal condition which will be beneficial for the organization.  
Research Design

Data Understanding

CODE- 
import pandas as pd
data= pd.read_csv('https://drive.google.com/file/d/1zND0swoxkRJY2mYdfMJvyHUtV5VzMn1e/view?usp=drivesdk')
data.describe()

Output- 


Data Preparation

Sampling

Sampling is done when collecting data from entire population is not feasible. The sample is selected in such a manner that it should represent the entire population.

In the above mentioned data,since we want to make statistical inferences we will use probabilistic sampling. Statistical predictions can be well made using probability sampling.

Stratified Sampling - 
In the above data we will progress towards stratified Sampling as we desire to make homogeneous groups. The homogeneous groups have been selected from the heterogeneous population.

Infinite Population

Since the sample data is very large,we will assume it to be an infinite Population.



Data Cleaning

Missing Values

data.dropna(inplace= True, axis= 0, subset=['Unit quantity'] )
data.dropna(inplace= True, axis= 0, subset=['Weight'] )
data







Output


Box Plot to Detect Outliers

import seaborn as sns
sns.boxplot(x=data['Weight'])


q1=1.407430
q3=13.325673
iqr=q3-q1
 
def outliers(values):
  if values>q3:
    return(values)
  if values<q1:
    return(values)
   
data=data['Weight'].apply(outliers)
q1=1.407430
q3=13.325673
iqr=q3-q1
 
def outliers(values):
  if values>q3:
    return(values)
  if values<q1:
    return(values)
   
data=data['Unit Quantity'].apply(outliers)

 
 

Code to Check Multicollinearity using VIF

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
 
# Read data from Excel file
data= pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Python Projects/Supply-chain-logisitcs-problem.csv')
 
# Assuming your Excel file has columns for multiple independent variables
 
# Extract data columns
X = data.drop(columns=['Weight'])  # Drop the dependent variable 'Y'
 
 
vif_data = X
vif_data["Feature"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
 
# Display VIF values
print("VIF Values:")
print(vif_data)
 
# You can set a threshold and identify features with high VIF values indicating multicollinearity
 
# For example, let's consider a threshold of 5
threshold = 5
high_vif_features = vif_data[vif_data["VIF"] > threshold]["Feature"]
 
print("\nFeatures with VIF greater than threshold:")
print(high_vif_features)
 




Code for Gradient Descent

import pandas as pd
import numpy as np
 
 
data= pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Python Projects/Supply-chain-logisitcs-problem.csv')
 
 
 
# Extract data columns
X = data['Weight'].values
y = data['Unit quantity'].values
 
# Hyperparameters
learning_rate = 0.01
num_iterations = 1000
 
# Initialize parameters
theta0 = 0
theta1 = 0
 
# Gradient Descent
for _ in range(num_iterations):
    predictions = theta0 + theta1 * X
    error = predictions - y
   
    # Compute gradients
    gradient0 = (1 / len(X)) * np.sum(error)
    gradient1 = (1 / len(X)) * np.sum(error * X)
   
    # Update parameters
    theta0 -= learning_rate * gradient0
    theta1 -= learning_rate * gradient1
 
# Print the final parameters
print("Final Parameters:")
print("theta0:", theta0)
print("theta1:", theta1)
 




Linear Regression:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
 
# Read data from Excel file
data= pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Python Projects/Supply-chain-logisitcs-problem.csv')
 
# Assuming your Excel file has columns 'X' and 'Y' for independent and dependent variables
 
# Extract data columns
x = data['Weight']
y = data['Unit quantity']
 
# Convert Series to a NumPy array and reshape
X = x.to_numpy().reshape(-1, 1)
Y= y.to_numpy().reshape(-1, 1)
 
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
 
# Create a linear regression model
model = LinearRegression()
 
# Train the model
model.fit(X_train, Y_train)
 
# Make predictions
Y_pred = model.predict(X_test)
 
# Calculate mean squared error
mse = mean_squared_error(Y_test, Y_pred)
 
# Print the model's coefficients and MSE
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
 

Output



Plotting

import matplotlib.pyplot as plt
# Plot the data points and the regression line
plt.scatter(X, Y, label='Data points')
plt.plot(X, predictions, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.show()





Code to check accuracy of the model using Performance Indicators
 Adjusted R sq.

# Assuming your Excel file has columns 'X' and 'Y' for independent and dependent variables
# Calculate R-squared
ss_residual = np.sum((Y_test - Y_pred) ** 2)
ss_total = np.sum((Y_test - np.mean(Y_test)) ** 2)
r_squared = 1 - (ss_residual / ss_total)
 
# Calculate adjusted R-squared
n = len(X)
p = 1  # Number of independent variables (features)
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
 
# Plot the data points and the regression line
plt.scatter(X, y, label='Data points')
plt.plot(X, predictions, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression and R-squared')
plt.legend()
 
# Annotate R-squared and Adjusted R-squared on the plot
plt.annotate(f'R-squared: {r_squared:.4f}', xy=(0.05, 0.9), xycoords='axes fraction')
plt.annotate(f'Adjusted R-squared: {adjusted_r_squared:.4f}', xy=(0.05, 0.85), xycoords='axes fraction')
 
plt.show()










