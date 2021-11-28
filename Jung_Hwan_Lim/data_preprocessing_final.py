import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split


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
    baths_drop = ['HalfBath', 'FullBath', 'BsmtFullBath', 'BsmtHalfBath']
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

    # Grouping all the different irregular lotshapes together as 'IR'
    housing.loc[housing.LotShape == 'IR1', 'LotShape'] = 'IR'
    housing.loc[housing.LotShape == 'IR2', 'LotShape'] = 'IR'
    housing.loc[housing.LotShape == 'IR3', 'LotShape'] = 'IR'

    # Grouping all the rare roofstyles together as 'Other'
    housing.loc[housing.RoofStyle == 'Gambrel', 'RoofStyle'] = 'Other'
    housing.loc[housing.RoofStyle == 'Flat', 'RoofStyle'] = 'Other'
    housing.loc[housing.RoofStyle == 'Mansard', 'RoofStyle'] = 'Other'
    housing.loc[housing.RoofStyle == 'Shed', 'RoofStyle'] = 'Other'

    housing['Bsmt_Unfin_Ratio'].fillna(value = 0, inplace = True)

    housing = housing[~housing.PID.isin([904300150, 535383070, 905426030, 528142130])]

    housing = housing[housing['SaleCondition'] == 'Normal']

    housing = housing.drop_duplicates(subset=['PID'], keep='first', ignore_index=False)


    return housing


def dummify(dataframe):
    dataframe = dataframe.reset_index() #duplicated index values in csv need to reset
    dataframe = dataframe.drop('index', axis = 1) # drop original index with duplicates
    ## Grab objects from dataframe
    category = dataframe.select_dtypes('object')

    ##Grab data that is numerical but should be treated as object (i.e. should be dummified)
    ## Merge it back into category to be dummified
    housing_num2cat = dataframe[['MSSubClass', 'OverallQual', 'OverallCond', \
                                'MoSold', 'YrSold']]

    category = pd.concat([category, housing_num2cat.astype(str)], axis = 1)

    ## Dummify variables
    cat_dum = pd.get_dummies(category, drop_first = True)

    ## Drop objects out of dataframe
    drop = list(category.columns)
    for column in housing_num2cat.columns:
        drop.append(column)
    dataframe.drop(columns = drop, axis = 1, inplace = True)

    ## Merge dummified back into original dataframe
    dataframe = pd.concat([dataframe, cat_dum], axis = 1)
    return dataframe



def scale_data(dataframe, scaler):
    ## Choose the columns that need to be scaled
    dataframe = dataframe.reset_index() #duplicated index values in csv need to reset
    dataframe = dataframe.drop('index', axis = 1) # drop original index with duplicates
    dataframe_num = dataframe[['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                            'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
                            'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                            'ScreenPorch', 'Bsmt_Unfin_Ratio', 'TotalLivArea', 'TotalBath']] # Select columns that were used to train the scaler

    ## Drop the original columns from the main dataframe
    dataframe.drop(columns = dataframe_num.columns, axis = 1, inplace =True)

    ## Scale the new columns and make dataframe
    num_scaled = scaler.transform(dataframe_num)
    dataframe_num_scaled = pd.DataFrame(num_scaled, columns = dataframe_num.columns)

    ## Merge back into old dataframe
    dataframe = pd.concat([dataframe, dataframe_num_scaled], axis=1)

    ## Separate out target
    target = np.log(dataframe['SalePrice'])
    dataframe.drop('SalePrice', axis=1, inplace = True)

    return dataframe, target



def ord_encoding(dataframe):

    dataframe = dataframe.reset_index() #duplicated index values in csv need to reset
    dataframe = dataframe.drop('index', axis = 1) # drop original index with duplicates
    ## Grab objects from dataframe
    category = dataframe.select_dtypes('object')

    ##Grab data that is numerical but should be treated as object (i.e. should be dummified)
    ## Merge it back into category to be dummified
    housing_num2cat = dataframe[['MSSubClass', 'OverallQual', 'OverallCond', \
                                'MoSold', 'YrSold']]

    category = pd.concat([category, housing_num2cat.astype(str)], axis = 1)

    ## Ord_Encode variables
    oe = OrdinalEncoder()
    cat_ord_enc = oe.fit_transform(category)
    cat_ord_enc_df = pd.DataFrame(cat_ord_enc, columns = category.columns)

    ## Drop objects out of dataframe
    drop = list(category.columns)
    for column in housing_num2cat.columns:
        drop.append(column)
    dataframe.drop(columns = drop, axis = 1, inplace = True)

    ## Merge dummified back into original dataframe
    dataframe = pd.concat([dataframe, cat_ord_enc_df], axis = 1)

    return dataframe

def initiate_data(housing):

    ## Do initial cleaning and set up
    ## For linear model it will dummify the variables
    ## For the tree model datasets it will use ordinal encoding
    housing = cleaning(housing)
    housing_linear = dummify(housing)
    housing_tree = ord_encoding(housing)

    ## Separate out training and testing data
    ## Separate out train and test data for linear model
    train_data_linear, test_data_linear = train_test_split(housing_linear, test_size=0.2, random_state = 0)
    train_data_linear = train_data_linear.copy()
    test_data_linear = test_data_linear.copy()

    ## Separate out train and test data for tree model
    train_data_tree, test_data_tree = train_test_split(housing_tree, test_size=0.2, random_state = 0)
    train_data_tree = train_data_tree.copy()
    test_data_tree = test_data_tree.copy()


    ## Set up the scaler using only training data to be passed into the scaling function

    ## First need to train the scaler
    scale_trainer = train_data_linear.copy()
    scale_trainer = scale_trainer.reset_index() #duplicated index values in csv need to reset
    scale_trainer = scale_trainer.drop('index', axis = 1) # drop original index with duplicates
    scale_trainer_num = scale_trainer.select_dtypes(['int64', 'float64']) # Select numeric data types
    scale_trainer_num = scale_trainer_num.drop(['PID', 'SalePrice'], axis = 1) ## Drop PID and saleprice since they should not be scaled

    ## Set up and train the scaler
    scaler = MinMaxScaler()
    scaler.fit(scale_trainer_num)


    ## Take the dataframes that have already been dummified and pass them into the scale_data() func
    ## this will scale the numerical data and separate the inputs from the target

    ## Set up linear training and testing data
    train_data_linear, train_target_linear = scale_data(train_data_linear, scaler)
    test_data_linear, test_target_linear = scale_data(test_data_linear, scaler)

    ## Set up tree training and testing data
    train_data_tree, train_target_tree = scale_data(train_data_tree, scaler)
    test_data_tree, test_target_tree = scale_data(test_data_tree, scaler)

    ## Return all data

    return train_data_linear, train_target_linear, test_data_linear, test_target_linear, train_data_tree, train_target_tree, test_data_tree, test_target_tree