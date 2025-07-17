# Credit Score Assigning ML model


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

wallet_raw_data = pd.read_json(r"E:\Vaishnavi\practiced\vs code\Self_Practice\internship projects\wallet_credit_scorer\user-wallet-transactions.json")

# Data Preprocessing

# have checked for duplicate and missing values none found

wallet_transactions = wallet_raw_data[["userWallet","timestamp","action","actionData"]].copy()

wallet_transactions['amount'] = wallet_transactions["actionData"].apply(lambda x: x.get('amount',0) if isinstance (x, dict) and 'amount' in x else 0)
wallet_transactions['amount'] = wallet_transactions['amount'].astype(float)
pd.set_option('display.float_format', '{:.0f}'.format)

wallet_transactions = wallet_transactions.drop(columns=["actionData"])

wallet_transactions["date"] = pd.to_datetime(wallet_transactions["timestamp"]).dt.date

# Feature Engineering


grouped = wallet_transactions.groupby('userWallet')

wallet_featured_data = grouped.agg(
    total_transactions=('action', 'count'),
    total_amount=('amount', 'sum'),
    avg_amount=('amount', 'mean'),
    first_txn=('timestamp', 'min'),
    last_txn=('timestamp', 'max'),
    active_days=('date', 'nunique')
).reset_index()


action_counts = wallet_transactions.pivot_table(
    index='userWallet',
    columns='action',                                       # each action as a new column
    aggfunc='size',
    fill_value=0
).reset_index()


wallet_featured_data = wallet_featured_data.merge(action_counts, on='userWallet', how='left')

wallet_featured_data['duration_days'] = (
    pd.to_datetime(wallet_featured_data['last_txn']) - pd.to_datetime(wallet_featured_data['first_txn']
)).dt.days                                                                            # to get number (if 'x days' is ouput get 'x') only                               

wallet_featured_data['repayment_ratio'] = wallet_featured_data.apply(
    lambda row: row['repay'] / row['borrow'] if row['borrow'] > 0 else 0, axis=1)

wallet_featured_data['unique_actions'] = wallet_featured_data[[
    'borrow', 'deposit', 'repay', 'redeemunderlying', 'liquidationcall']].gt(0).sum(axis=1)



# Create proxy target variable


from sklearn.preprocessing import MinMaxScaler                                          # needed scaled feature to use in formula for proxy target variable
scaler = MinMaxScaler()

features = ['repayment_ratio', 'liquidationcall', 'deposit', 'active_days', 'unique_actions']
scaled_data = scaler.fit_transform(wallet_featured_data[features])
wallet_featured_data[['repayratio_scaled', 'liquidationcall_scaled', 'deposit_scaled', 'activedays_scaled', 'uniqueactions_scaled']] = scaled_data

# sum of all the constant in formula should be 1
wallet_featured_data['normalised_credit_score'] = (                                        
    0.4 * wallet_featured_data['repayratio_scaled'] +
    0.2 * (1 - wallet_featured_data['liquidationcall_scaled']) +                           #  minimum liquidation is better
    0.15 * wallet_featured_data['deposit_scaled'] +
    0.15 * wallet_featured_data['activedays_scaled'] +
    0.1 * wallet_featured_data['uniqueactions_scaled']
) * 1000                                                                                   # to scale between 0–1000
                                                     


# Model Building

x = wallet_featured_data[['total_transactions', 'total_amount', 'avg_amount', 'borrow', 'deposit','active_days']]
y = wallet_featured_data['normalised_credit_score']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Before beuilding this model different sample model have build using different algorithm and calculated r2, MAE, RMSE. 
# Finally choose the model that is most realiable one with high r2 and low MAE and RMSE

model = RandomForestRegressor(random_state=0)
model.fit(x_train, y_train)

score = model.predict(x)                                                    # credit score column

wallet_featured_data["credit_score"] = pd.DataFrame(score)

wallet_featured_data.to_csv("wallet_score.csv", index=False)                 # saving file with all feture including target feeture

print(f"File saved succesfully at location:{os.getcwd()}")

# Note: main .ipynb file does contain all code in detail including each step during data cleaning and different algorithm models with calculated r2 for train and test data, MAE and RMSE. CAn be submitted on demand.
