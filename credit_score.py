#!/usr/bin/env python
# coding: utf-8

# # Credit Score Assigning ML model

# # Data Preprocessing

# In[ ]:


import pandas as pd
import numpy as np


# In[3]:


wallet_raw_data = pd.read_json(r"E:\Vaishnavi\practiced\vs code\Self_Practice\internship projects\wallet_credit_scorer\user-wallet-transactions.json")
wallet_raw_data.sample(3)


# In[4]:


wallet_raw_data.shape


# In[5]:


wallet_raw_data.info()


# In[6]:


wallet_raw_data.isnull().sum()


# In[7]:


wallet_transactions = wallet_raw_data[["userWallet","timestamp","action","actionData"]].copy()
wallet_transactions.head()


# In[8]:


wallet_transactions['amount'] = wallet_transactions["actionData"].apply(lambda x: x.get('amount',0) if isinstance (x, dict) and 'amount' in x else 0)
wallet_transactions.amount.head()


# In[9]:


wallet_transactions['amount'] = wallet_transactions['amount'].astype(float)
pd.set_option('display.float_format', '{:.0f}'.format)


# In[10]:


wallet_transactions = wallet_transactions.drop(columns=["actionData"])


# In[11]:


wallet_transactions["date"] = pd.to_datetime(wallet_transactions["timestamp"]).dt.date


# In[12]:


wallet_transactions.head()


# In[13]:


wallet_transactions.isnull().sum()


# In[14]:


wallet_transactions.info()


# # Feature Engineering

# In[15]:


# modify dataset to feed to model

grouped = wallet_transactions.groupby('userWallet')

wallet_featured_data = grouped.agg(
    total_transactions=('action', 'count'),
    total_amount=('amount', 'sum'),
    avg_amount=('amount', 'mean'),
    first_txn=('timestamp', 'min'),
    last_txn=('timestamp', 'max'),
    active_days=('date', 'nunique')
).reset_index()


# In[16]:


action_counts = wallet_transactions.pivot_table(
    index='userWallet',
    columns='action',                                       # each action as a new column
    aggfunc='size',
    fill_value=0
).reset_index()


# In[17]:


wallet_featured_data = wallet_featured_data.merge(action_counts, on='userWallet', how='left')


# In[18]:


wallet_featured_data['duration_days'] = (
    pd.to_datetime(wallet_featured_data['last_txn']) - pd.to_datetime(wallet_featured_data['first_txn'])
).dt.days                                                                                  # to get number (if 'x days' is ouput get 'x') only


# In[19]:


# repay/borrow ratio - describe does user returns everytime he borrows

wallet_featured_data['repayment_ratio'] = wallet_featured_data.apply(
    lambda row: row['repay'] / row['borrow'] if row['borrow'] > 0 else 0, axis=1
)

wallet_featured_data['unique_actions'] = wallet_featured_data[[
    'borrow', 'deposit', 'repay', 'redeemunderlying', 'liquidationcall'
]].gt(0).sum(axis=1)


# In[20]:


wallet_featured_data.head()


# # Create proxy target variable 

# In[21]:


# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

features = ['repayment_ratio', 'liquidationcall', 'deposit', 'active_days', 'unique_actions']

scaled_data = scaler.fit_transform(wallet_featured_data[features])


wallet_featured_data[['repayratio_scaled', 'liquidationcall_scaled', 'deposit_scaled', 'activedays_scaled', 'uniqueactions_scaled']] = scaled_data


# In[22]:


wallet_featured_data['normalised_credit_score'] = (
    0.4 * wallet_featured_data['repayratio_scaled'] +
    0.2 * (1 - wallet_featured_data['liquidationcall_scaled']) +                           #  minimum liquidation is better
    0.15 * wallet_featured_data['deposit_scaled'] +
    0.15 * wallet_featured_data['activedays_scaled'] +
    0.1 * wallet_featured_data['uniqueactions_scaled']
) * 1000                                                                                   # to scale between 0–1000


#  Value to use in this formula are randomly selected considring the fetures. Final sum of all these should be 1.
#  The less liquidation the better hence used 1 - liq_scale. 

# In[23]:


wallet_featured_data['normalised_credit_score'].head()


# Now the target variable is contineous, so we will use regression model

# # Model Building
# Here the data do not have dependent variable. Hence we use a proxy deoendent variable for prediction.

# In[29]:


x = wallet_featured_data[['total_transactions', 'total_amount', 'avg_amount', 'borrow', 'deposit','active_days']]
y = wallet_featured_data['normalised_credit_score']


# In[30]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[47]:


# Check different models

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
}

# Loop and evaluate
for name, model in models.items():
    print(f"\n🔹 Model: {name}")
    model.fit(x_train, y_train)
    
    # Predict
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # R² Scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    
    # Print results
    print(f"{name}:=>  Train R²: {train_r2:.4f} , Test  R²: {test_r2:.4f}, MAE = {mae}, RMSE = {rmse}")
    


# Choose the one with best performance for a given data and requirement.

# In[ ]:


# use randomforest to generate credit_score

model = RandomForestRegressor(random_state=0)
model.fit(x_train, y_train)

score = model.predict(x)


# # Build column for credit score and save a file

# In[52]:


wallet_featured_data["credit_score"] = pd.DataFrame(score)


# In[ ]:


wallet_featured_data.to_csv("wallet_score.csv", index=False)

