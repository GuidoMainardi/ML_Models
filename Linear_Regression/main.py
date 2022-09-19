from LinearRegression import LinearRegression
import pandas as pd

data = pd.read_csv('train.csv', index_col=0)
print(data.head())
print(data.corr())
lr = LinearRegression()

lr.fit(data['height'].to_list(), data['weight'].to_list())

print(lr.R_square())