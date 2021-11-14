import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
%matplotlib inline

housing = pd.read_csv('Ames_HousePrice.csv', index_col=0)

housing = housing.reset_index() #duplicated index values in csv need to reset
housing = housing.drop('index', axis = 1) # drop original index with duplicates

price = housing['SalePrice'] # Create Y Variable

category = housing.select_dtypes('object') #Select all 'object' data types  which are all categorical

# TEMPORARY STEP fill empty values with 0, will need to run cleaning file on data prior to importing
category = category.fillna('0')
# END TEMP STEP

housing_num = housing.select_dtypes('int64', 'float64') # Select numeric data types

housing_num_PID = housing_num['PID'] # PID index should not be scaled, remove and put back later

## Numeric Colums to convert
# MSSubClass, OverallQual, OverallCond, YearBuilt, YearRemodAdd, MoSold, YrSold
# How to handle MiscVal???
#Leave YearBuilt and YearRemodAdd as numeric to be scaled

housing_num = housing_num.drop(['PID', 'SalePrice', 'MSSubClass', 'OverallQual', \
    'OverallCond', 'MoSold', 'YrSold', 'MiscVal'], axis = 1)

housing_num2cat = housing[['MSSubClass', 'OverallQual', 'OverallCond', \
    'MoSold', 'YrSold', 'MiscVal']]

category = pd.concat([category, housing_num2cat.astype(str)], axis = 1) #Add all categorical features to dataframe to be dummified

cat_dum = pd.get_dummies(category, drop_first = True)

scaler = MinMaxScaler()
scaler.fit(housing_num)
housing_num_scaled = scaler.transform(housing_num)
housing_num_scaled = pd.DataFrame(housing_num_scaled, columns = housing_num.columns)


full_dum_data = pd.concat([housing_num_PID, housing_num, cat_dum], axis = 1) #Concatendate dummified data and numeric data

full_dum_data.to_csv('dum_scaled_data.csv')
