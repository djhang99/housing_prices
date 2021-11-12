import numpy as np
import pandas as pd
%matplotlib inline

housing = pd.read_csv('Ames_HousePrice.csv', index_col=0)

price = housing['SalePrice'] # Create Y Variable

category = housing.select_dtypes('object') #Select all 'object' data types  which are all categorical

# TEMPORARY STEP fill empty values with 0, will need to run cleaning file on data prior to importing
category = category.fillna('0')
# END TEMP STEP

housing_num = housing.select_dtypes('int64', 'float64') # Select numeric data types

## Numeric Colums to convert
# MSSubClass, OverallQual, OverallCond, YearBuilt, YearRemodAdd, MoSold, YrSold
# How to handle MiscVal???

housing_num = housing_num.drop(['SalePrice', 'MSSubClass', 'OverallQual', \
    'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold', 'MiscVal'], axis = 1)

housing_num2cat = housing[['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', \
    'YearRemodAdd', 'MoSold', 'YrSold', 'MiscVal']]

category = pd.concat([category, housing_num2cat], axis = 1) #Add all categorical features to dataframe to be dummified

cat_dum = pd.get_dummies(category, drop_first = True)

full_dum_data = pd.concat([housing_num, cat_dum], axis = 1) #Concatendate dummified data and numeric data
