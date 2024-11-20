from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load the californai housing dataset
housing = fetch_california_housing()

#create a dataframe
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target
print(df.head())

#MedInc : Median income in block group
#HouseAge : Median house age in block group
#AveRooms : Average number of rooms per household
#AveBedrms: Average number of bedrooms per household
#Population : Population in block group
#AveOccup: Average number of household members
#Latitude: Block group latitude
#Longitude: Block group longitude
#Price: Target variable representing house price(in hundreds of thousands of dollars)
 
#Exploring the data: Let's check some basic statistics of the data and visualize the relationships between a couple of features and house price.
 
#Summary statistics
# print(df.describe())

#Plotting the relationship between Median Income and Price
plt.scatter(df['MedInc'], df['Price'])
plt.title('Median Income vs Price')
plt.xlabel('Median Income')
plt.ylabel('Price')
plt.show()

#Splitting the data: We'll split the dataset into training and testing sets. This ensures we evaluate the model on unseen data to check its performance.

#Split data into features (X) and target (Y)
x = df.drop('Price', axis=1)
y = df['Price']

#Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Training the model: Now we will train a linear regression model onthe training data.
#Train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

#Get the model coefficients
print('Intercept: ', model.intercept_)
print('Coefficients: ', model.coef_)

#Evaluating the model: After training the model, we'll use it to make predictions on the test data and calculate key metrics like Mean Squared Error (MSE) and R-Squared (R square).

#Predict house prices for the test set
y_pred = model.predict(x_test)

#Calculate Mean Squared Error and R-Squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}') #R-squared show how well the model explains the variance  in the data

newDF = pd.DataFrame({'Actual' : y_test,'Predicted': y_pred})
newDF.to_csv('ActualvsPredicted.csv')

#Visualizing predictions: We can plot the predicted values against the actual values to see how well our model is performing.

#Visualize the predicted vs actual prices
plt.scatter(y_test, y_pred)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red") #Line for perfect predictions
plt.show()


