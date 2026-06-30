
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns
import math


df = pd.read_csv(r'D:\MACHINE LEARNING\Restraurant rating model\final.csv')
dependent_variable = 'rate'
independent_variable = list(set(df.columns.to_list())-{dependent_variable})
X = df[independent_variable].values  # input features
y = df[dependent_variable].values    # target/output
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 324)

# DECISION TREE
model = tree.DecisionTreeRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(math.sqrt(mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred))

# RANDOM FOREST
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(math.sqrt(mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred))

# XG BOOST
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=0
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(math.sqrt(mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred))
