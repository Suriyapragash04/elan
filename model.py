import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("clean.csv",index_col='Unnamed: 0')
y_train = df['Price']
X_train = df.drop('Price',axis=1)
# y_train = df['Price']
from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)
# Make pickle file of our model
pickle.dump(reg_rf, open("model.pkl", "wb"))