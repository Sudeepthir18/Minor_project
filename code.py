import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data from CSV file
df = pd.read_excel('1729258-1613615-Stock_Price_data_set_(1).xlsx')

# Assuming 'Date' is in datetime format, if not, convert it using: df['Date'] = pd.to_datetime(df['Date'])
# Sort the dataframe by date
df = df.sort_values(by='Date')

# Feature engineering: convert 'Date' to ordinal values for linear regression
df['Date_Ordinal'] = df['Date'].apply(lambda x: x.toordinal())

# Select features and target variable
X = df[['Date_Ordinal']]
y = df['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the predictions against the actual values
plt.scatter(X_test, y_test, c='r', label='Actual')
plt.plot(X_test, y_pred, ":b", linewidth=3, label='Predicted')
plt.xlabel('Date (Ordinal)')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
