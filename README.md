Stock Price Prediction using Linear Regression
Overview
This repository contains a Python script for predicting stock prices using linear regression. The code utilizes the scikit-learn library for machine learning, pandas for data manipulation, and matplotlib for data visualization.

Requirements
Python 3.x
pandas
numpy
scikit-learn
matplotlib

Installation
bash
Copy code
pip install pandas numpy scikit-learn matplotlib
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
Download the dataset

Place the dataset file 1729258-1613615-Stock_Price_data_set_(1).xlsx in the same directory as the script.

Run the script:

bash
Copy code
python stock_price_prediction.py
Description
Loading Data:

The script reads stock price data from the provided Excel file using pandas.
Data Preprocessing:

The 'Date' column is converted to datetime format and the DataFrame is sorted by date.
Feature engineering is performed by converting the 'Date' to ordinal values to use it as a feature for linear regression.
Model Training:

The data is split into training and testing sets using scikit-learn's train_test_split.
A linear regression model is created and trained using the training data.
Prediction and Evaluation:

Predictions are made on the test set, and mean squared error is calculated using scikit-learn's mean_squared_error.
Visualization:

The script uses matplotlib to create a scatter plot comparing actual stock prices with predicted prices.
Results
The Mean Squared Error (MSE) is printed to the console, and a plot is generated to visually compare predicted and actual stock prices.

Acknowledgments
The code is based on a common machine learning workflow and uses popular Python libraries for data analysis and machine learning.
