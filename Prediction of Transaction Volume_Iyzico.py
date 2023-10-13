##############################################################
# Prediction with Time Series
# Forecasting of POS Transaction Volume on Iyzico
##############################################################

##############################################################
# 1. Business Problem
##############################################################
# Iyzico is a financial technology company facilitating online shopping experience for both shoppers and seller.
# Iyzico provides payment process platform for e-commerce companies, marketplaces and individual users.
# It is expected to make an estimation of the total transaction volume on merchant_id based on time for the last three months of 2020

# Dataset story
# Variables:
# transaction_date: Date of sales data, between 01.01.2018 to 31.12.2020
# merchant_id : ID's of business member
# Total_Transaction : Number of transactions
# Category : Category names of member companies
# Total_Paid : Total paid amount


##########################################
# TASK 1 - EXPLORATORY DATA ANALYSIS (EDA)
##########################################

# Importing libraries
##############################################
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
warnings.filterwarnings('ignore')

df = pd.read_csv("Datasets/iyzico_data.csv")
df.head()

# Check DataFrame Information
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)

# Converting the type of transaction_date to date
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df.info()

# Showing the start and end dates of the dataset
df["transaction_date"].min(), df["transaction_date"].max()

# Showing the merchant numbers
df[["merchant_id"]].nunique()

# Showing the total transaction by merchant
df.groupby("merchant_id").agg({"Total_Transaction": "sum"})

# Showing the total transaction in a bar chart
df.groupby(["merchant_id"]).agg({"Total_Transaction": "sum"}).plot.bar()
plt.show(block=True)

# Renaming first column
df.rename(columns={"Unnamed: 0": "index"}, inplace=True)

#####################################################
# TASK 2 - FEATURE ENGINEERING
#####################################################
# Creating New Date Features
##############################
def create_date_features(df):
    df['month'] = df.transaction_date.dt.month
    df['day_of_month'] = df.transaction_date.dt.day
    df['day_of_year'] = df.transaction_date.dt.dayofyear
    df['week_of_year'] = df.transaction_date.dt.weekofyear
    df['day_of_week'] = df.transaction_date.dt.dayofweek
    df['year'] = df.transaction_date.dt.year
    df["is_wknd"] = df.transaction_date.dt.weekday // 4
    df['is_month_start'] = df.transaction_date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.transaction_date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)
df.head()

# Creating Lag/Shifted Features and Adding Random Noise
#################################################################

df.sort_values(by=['merchant_id','transaction_date'], axis=0, inplace=True) #--> To add lag/shifted values
# it is importan to sort values by merchant and date

# Defining two function, random_noise and lag_features
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['Total_Transaction_lag' + str(lag)] = dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [30, 61, 92, 98, 105, 112, 119, 126, 182, 364, 546, 728])

check_df(df) # --> These features are oriented in 3 months period because prediction period is 3 months.

# Define a function to identify Rolling Mean Features
#####################################################
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["merchant_id"])['Total_Transaction']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [365, 546])

# Define a function to identify Exponentially Weighted Mean Features
#####################################################################
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['Transaction_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [30,61, 92, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
check_df(df)


#######################################
# TASK 3 - MODELLING
#######################################

# One-Hot Encoding
########################
df = pd.get_dummies(df, columns=['merchant_id', 'day_of_week', 'month', "year"])

check_df(df)

# Converting sales to log(1+sales) --> The purpose of "log" is to standardize the dependent variable and shorten the train time
###################################
df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values) #--> "1" is a method ussed to prevent some possible errors.

check_df(df)


# Custom Cost Function
###############################
# Symmetric Mean Absolute Percent Error (SMAPE) is an alternative to Mean Absolute Percent Error (MAPE)
# when there are zero or near-zero demand for items. SMAPE self-limits to an error rate of 200%,
# reducing the influence of these low volume items.

# Define the smape and lgbm_smape function
###############################################3
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

# Time-Based Validation Sets
#############################

# Train set until the beginning of 2020.10
train = df.loc[(df["transaction_date"] < "2019-10-01"), :]

# Validation set for the last 3 months of 2019
val = df.loc[(df["transaction_date"] >= "2019-10-01") & (df["transaction_date"] < "2019-12-01"), :]

cols = [col for col in train.columns if col not in ['transaction_date', "Total_Paid", "Total_Transaction", "year"]]

Y_train = train['Total_Transaction']
X_train = train[cols]

Y_val = val['Total_Transaction']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape


# Time Series Model with LightGBM
#####################################

# LightGBM parameters
lgb_params = {'num_leaves': 10, # -->max number of leaves on a tree
              'learning_rate': 0.02,# --> shrinkage_rate
              'feature_fraction': 0.8, # --> same as Random Subspace inf RF. Random number of variables to consider in each iteration
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200, #--> if the error does not decrease, stop modeling.
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  #early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,)
                  #verbose_eval=100)
#-- > LightGBM 4.0.0 does not support early_stopping_rounds and verbose_eval in train() function

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))
#--> Out[32]: 25.95862998145779


# Feature İmportance
######################################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show(block=True)
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)


# Final Model
#####################################
train = df.loc[(df["transaction_date"] < "2020-10-01"), :]
Y_train = train['Total_Transaction']
cols = [col for col in train.columns if col not in ['transaction_date', "Total_Paid", "Total_Transaction", "year"]]
X_train = train[cols]

test = df.loc[(df["transaction_date"] >= "2020-10-01"), :]
X_test = test[cols]
Y_test = test['Total_Transaction']

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_test), np.expm1(Y_test))
#--> Out[32]: 33.35761410556619

# Submission File
#####################################
test.head()

submission = pd.DataFrame({"transaction_date":test.transaction_date,"Total_Transaction":y_pred_test})

submission.to_csv("iyzico_transaction_forecast.csv", index=False)

# Submission File / Prediction Visualization
#############################################
prediction_last_3_months = pd.DataFrame({"transaction_date":test.transaction_date,"Total_Transaction":y_pred_test})
prediction_last_3_months.set_index("transaction_date").Total_Transaction.plot(color = "green", figsize = (20,9),legend=True);
plt.show(block=True)