import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'House Size (sq ft)': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    'House Price ($)': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}

#convert the dictionary to a pandas Dataframe
df = pd.DataFrame(data)
# print(df.head())

#Visualizing the data
#Plot the data
plt.scatter(df['House Size (sq ft)'], df['House Price ($)'])
plt.title('House Size vs Price')
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price ($)')
# plt.show()

#Preparing the data
#Split the data into the feature (x) and the target variable (y), then split it into training, and testing sets.

#Define features (X) and target (Y)
x = df[['House Size (sq ft)']]
y = df[['House Price ($)']]

#Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#trainign the Linear Regression mode :
#Now, we'll use scikit-learn's LinearRegression model to train on the dataset.

#Create and train the model
model = LinearRegression()
model.fit(x_train, y_train)

#Print the coefficients
print(f'Intercept: {model.intercept_}')
print(f'Coefficient : {model.coef_[0]}')

#The intercept is the value of the house price when the house size is zero (not always meaningful but part of the model), and the coefficient represents how much the house price increase for each additional square foot of size.

#Making Predictions: Let's predict the house prices for hte test data and evaluate the model performance.

#Predict house prices for the test set
y_pred = model.predict(x_test)

#compare predicted and actual values
comparison = pd.DataFrame({'Actual' : [y_test], 'Predicted' : [y_pred]})

#Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

#Visualizing the regression line : Finally, plot the regression line on top of the scatter plot to see how well hte model fits the data.

#Plot the regression line
plt.scatter(x, y, color='blue')
plt.plot(x, model.predict(x), color='red')
plt.title('House Size vs Price with Regression Line')
plt.xlabel('House Size (sq ft)')
plt.ylabel('House price ($)')
plt.show()