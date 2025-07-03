''' test_projekt_ML  '''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg

DATA_FILE_DUMMY = 'cars_dummy_no_outliers.csv'
DATA_FILE_DUMMY_ALL = 'cars_dummy_all_no_outliers.csv'

def main():

    df_dummy = load_data_from_csv(DATA_FILE_DUMMY_ALL)

    print(df_dummy.info())

    #sns.pairplot(df_dummy[1:], diag_kind='kde', markers=["o", "s", "D"])

    #for col in df_dummy.columns[7:]:
    #    print(col)
    #    sns.pairplot(df_dummy[df_dummy[col]==1][1:7], diag_kind='auto')
    df_4_ML = df_dummy.iloc[:,1:].copy()

    y = df_4_ML[['mpg']].values.reshape((-1,))
    X = df_4_ML.drop(labels='mpg',axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

    model_lin = LinearRegression()
    model_knn = KNeighborsRegressor(n_neighbors=3)
    model_RF = RandomForestRegressor(n_estimators=100, random_state=0)
    model_xgb = xg.XGBRegressor(objective ='reg:linear',
                  n_estimators = 100, seed = 0)

    model_lin.fit(X_train, y_train)
    model_knn.fit(X_train, y_train)
    model_RF.fit(X_train, y_train)
    model_xgb.fit(X_train, y_train)

    #print('coef_ :', model_lin.coef_)
    print('Model_lin train score', model_lin.score(X_train, y_train))
    print('Model_lin test score', model_lin.score(X_test, y_test))
    print('-----------------------------------------------------')
    print('Model_KNN train score', model_knn.score(X_train, y_train))
    print('Model_KNN test score', model_knn.score(X_test, y_test))
    print('-----------------------------------------------------')
    print('Model_RF train score', model_RF.score(X_train, y_train))
    print('Model_RF test score', model_RF.score(X_test, y_test))
    print('-----------------------------------------------------')
    print('Model_XGB train score', model_xgb.score(X_train, y_train))
    print('Model_XGB test score', model_xgb.score(X_test, y_test))

    return []


def load_data_from_csv(file):
    data = pd.read_csv(file)
    return data

def save_df_data_to_csv(data, file):
    data.to_csv(file, index=False)
    return []

if __name__ == '__main__':
    main()
