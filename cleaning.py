
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="white")
%matplotlib inline 

##Import Data 
housing = pd.read_csv('./data/Ames_Housing_Price_Data.csv', index_col=0)
housing = housing[housing['SaleCondition'] == 'Normal']
real_estate = pd.read_csv('./data/Ames_Real_Estate_data.csv' , index_col = 0)

## Replace all null values based on context

## LotFrontage -- replace with mean of the column 
housing['LotFrontage'].fillna(value = housing['LotFrontage'].mean(), inplace = True)

## Replace Nulls where NA just means that the house does not have the feature 

## Basement metrics
housing['BsmtQual'].fillna(value = 'No_Basement', inplace=True)
housing['BsmtCond'].fillna(value = 'No_Basement', inplace=True)
housing['BsmtExposure'].fillna(value = 'No_Basement', inplace=True)
housing['BsmtFinType1'].fillna(value = 'No_Basement', inplace=True)
housing['BsmtFinType2'].fillna(value = 'No_Basement', inplace=True)

## Garage Type 
housing['GarageType'].fillna(value = 'No_Garage', inplace=True)
housing['GarageYrBlt'].fillna(value = 'No_Garage', inplace=True)
housing['GarageFinish'].fillna(value = 'No_Garage', inplace=True)
housing['GarageQual'].fillna(value = 'No_Garage', inplace=True)
housing['GarageCond'].fillna(value = 'No_Garage', inplace=True)

## Replace other nulls where null just means the feature is not there 
housing['FireplaceQu'].fillna(value = 'No_Fireplace', inplace=True)
housing['PoolQC'].fillna(value = 'No_Pool', inplace=True)
housing['Fence'].fillna(value = 'No_Fence', inplace=True)
housing['MiscFeature'].fillna(value = 'No_Misc', inplace=True)
housing['Alley'].fillna(value = 'No_alley', inplace=True)

##----------------------------------------------------------------##

##Replacing nulls with 0s

housing['BsmtFinSF1'].fillna(value = 0, inplace=True)
housing['BsmtFinSF2'].fillna(value = 0, inplace=True)
housing['BsmtUnfSF'].fillna(value = 0, inplace=True)
housing['MasVnrType'].fillna(value = 0, inplace=True)
housing['TotalBsmtSF'].fillna(value = 0, inplace=True)
housing['MasVnrArea'].fillna(value = 0, inplace=True)
housing['BsmtFullBath'].fillna(value = 0, inplace=True)
housing['BsmtHalfBath'].fillna(value = 0, inplace=True)
housing['GarageCars'].fillna(value = 0, inplace=True)
housing['GarageArea'].fillna(value = 0, inplace=True)


## At this point, only one null value remains in the "Electric Column". We will just remove that one row 
housing.dropna(axis = 0, inplace = True)