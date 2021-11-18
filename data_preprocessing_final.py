

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## Cleaning is going to take the data and remove all null values

def cleaning(dataframe):
    ## Replace all null values based on context
    housing = dataframe
    ## LotFrontage -- replace with mean of the column
    housing['LotFrontage'].fillna(value = housing['LotFrontage'].mean(), inplace = True)
    housing['GarageYrBlt'].fillna(value = housing['GarageYrBlt'].mean(), inplace=True)

    ## Replace Nulls where NA just means that the house does not have the feature

    ## Basement metrics
    housing['BsmtQual'].fillna(value = 'No_Basement', inplace=True)
    housing['BsmtCond'].fillna(value = 'No_Basement', inplace=True)
    housing['BsmtExposure'].fillna(value = 'No_Basement', inplace=True)
    housing['BsmtFinType1'].fillna(value = 'No_Basement', inplace=True)
    housing['BsmtFinType2'].fillna(value = 'No_Basement', inplace=True)

    ## Garage Type
    housing['GarageType'].fillna(value = 'No_Garage', inplace=True)

    housing['GarageFinish'].fillna(value = 'No_Garage', inplace=True)
    housing['GarageQual'].fillna(value = 'No_Garage', inplace=True)
    housing['GarageCond'].fillna(value = 'No_Garage', inplace=True)

    ## Replace other nulls where null just means the feature is not there
    housing['FireplaceQu'].fillna(value = 'No_Fireplace', inplace=True)
    housing['PoolQC'].fillna(value = 'No_Pool', inplace=True)
    housing['Fence'].fillna(value = 'No_Fence', inplace=True)
    housing['MiscFeature'].fillna(value = 'No_Misc', inplace=True)
    housing['Alley'].fillna(value = 'No_alley', inplace=True)

    housing['MasVnrType'].fillna(value = 'None', inplace=True)


    ##----------------------------------------------------------------##

    ##Replacing nulls with 0s

    housing['BsmtFinSF1'].fillna(value = 0, inplace=True)
    housing['BsmtFinSF2'].fillna(value = 0, inplace=True)
    housing['BsmtUnfSF'].fillna(value = 0, inplace=True)
    housing['TotalBsmtSF'].fillna(value = 0, inplace=True)
    housing['MasVnrArea'].fillna(value = 0, inplace=True)
    housing['BsmtFullBath'].fillna(value = 0, inplace=True)
    housing['BsmtHalfBath'].fillna(value = 0, inplace=True)
    housing['GarageCars'].fillna(value = 0, inplace=True)
    housing['GarageArea'].fillna(value = 0, inplace=True)


    ## At this point, only one null value remains in the "Electric Column". We will just remove that one row
    housing.dropna(axis = 0, inplace = True)


    ## Create new variables


    ## Add a total baths feature and remove the original columns 
    housing['TotalBath'] = housing['FullBath'] + (housing['HalfBath']*0.5) + housing['BsmtFullBath'] + (housing['BsmtHalfBath']*0.5)
    baths_drop = ['HalfBath', 'FullBath', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBath']
    housing.drop(columns= baths_drop, inplace=True, axis =1)
    
    ## Add ratio for unfinished basement space -- Make sure this is always before "TotalLivArea" is created or columns will be dropped
    housing['Bsmt_Unfin_Ratio'] = housing['BsmtUnfSF'] / housing['TotalBsmtSF']
    housing.drop(columns = 'BsmtUnfSF', axis = 1, inplace = True)

    ## Add a total living area feature and remove original columns 
    housing['TotalLivArea'] = housing['GrLivArea'] + housing['TotalBsmtSF']
    liv_drop = ['GrLivArea', 'TotalBsmtSF']
    housing.drop(columns = liv_drop, axis = 1, inplace = True)

    ## Remove bedrooms from above ground total rooms to avoid multicollinearity
    housing['TotRmsAbvGrd'] = housing['TotRmsAbvGrd'] - housing['BedroomAbvGr']


    ## Remove unnecessary columns 
    cols_to_drop = [
        '1stFlrSF',
        '2ndFlrSF',
        'BsmtFinSF1', 
        'BsmtFinSF2',
        'Street',
        'Alley',
        'Utilities',
        'Condition2',
        'RoofMatl',
        'Heating',
        'Electrical',
        'LowQualFinSF',
        'KitchenAbvGr',
        'GarageCars',
        'GarageCond',
        'PoolArea',
        'PoolQC',
        'MiscFeature',
        'MiscVal',
        'SaleType']

    housing.drop(columns= cols_to_drop, axis=1, inplace=True)

    return housing
    




def dummify_func(housing):
    housing = housing.reset_index() #duplicated index values in csv need to reset
    housing = housing.drop('index', axis = 1) # drop original index with duplicates
    price = housing['SalePrice'] # Create Y Variable
    category = housing.select_dtypes('object') #Select all 'object' data types  which are all categorical
    housing_num = housing.select_dtypes('int64', 'float64') # Select numeric data types
    housing_num_PID = housing_num['PID'] # PID index should not be scaled, remove and put back later
    ## Numeric Colums to convert
    # MSSubClass, OverallQual, OverallCond, YearBuilt, YearRemodAdd, MoSold, YrSold
    # How to handle MiscVal???
    #Leave YearBuilt and YearRemodAdd as numeric to be scaled
    housing_num = housing_num.drop(['PID', 'SalePrice', 'MSSubClass', 'OverallQual', \
    'OverallCond', 'MoSold', 'YrSold'], axis = 1)
    housing_num2cat = housing[['MSSubClass', 'OverallQual', 'OverallCond', \
    'MoSold', 'YrSold']]
    category = pd.concat([category, housing_num2cat.astype(str)], axis = 1) #Add all categorical features to dataframe to be dummified
    cat_dum = pd.get_dummies(category, drop_first = True)
    scaler = MinMaxScaler()
    scaler.fit(housing_num)
    housing_num_scaled = scaler.transform(housing_num)
    housing_num_scaled = pd.DataFrame(housing_num_scaled, columns = housing_num.columns)
    full_dum_data = pd.concat([housing_num_PID, housing_num_scaled, cat_dum], axis = 1) #Concatenate dummified data and numeric data
    return full_dum_data, pd.DataFrame(price)
