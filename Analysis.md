# Analysis

This file contains the EDA and analysis of wallet transaction of Defi with Aave2 protocal.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
```


```python
wallet_score = pd.read_csv(r"E:\Vaishnavi\practiced\vs code\Self_Practice\internship projects\wallet_credit_scorer\wallet_score.csv")
```

# EDA


```python
wallet_score.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userWallet</th>
      <th>total_transactions</th>
      <th>total_amount</th>
      <th>avg_amount</th>
      <th>first_txn</th>
      <th>last_txn</th>
      <th>active_days</th>
      <th>borrow</th>
      <th>deposit</th>
      <th>liquidationcall</th>
      <th>...</th>
      <th>duration_days</th>
      <th>repayment_ratio</th>
      <th>unique_actions</th>
      <th>repayratio_scaled</th>
      <th>liquidationcall_scaled</th>
      <th>deposit_scaled</th>
      <th>activedays_scaled</th>
      <th>uniqueactions_scaled</th>
      <th>normalised_credit_score</th>
      <th>credit_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0x00000000001accfa9cef68cf5371a23025b6d4b6</td>
      <td>1</td>
      <td>2.000000e+09</td>
      <td>2.000000e+09</td>
      <td>2021-08-17 05:29:26</td>
      <td>2021-08-17 05:29:26</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.001957</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>200.293542</td>
      <td>200.293542</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0x000000000051d07a4fb3bd10121a343d85818da6</td>
      <td>1</td>
      <td>1.450000e+20</td>
      <td>1.450000e+20</td>
      <td>2021-05-20 15:36:53</td>
      <td>2021-05-20 15:36:53</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.001957</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>200.293542</td>
      <td>200.293542</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0x000000000096026fb41fc39f9875d164bd82e2dc</td>
      <td>2</td>
      <td>5.000000e+15</td>
      <td>2.500000e+15</td>
      <td>2021-07-24 09:28:33</td>
      <td>2021-07-31 23:15:18</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.003914</td>
      <td>0.009009</td>
      <td>0.00</td>
      <td>201.938436</td>
      <td>201.938436</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0x0000000000e189dd664b9ab08a33c4839953852c</td>
      <td>17</td>
      <td>4.835297e+18</td>
      <td>2.844292e+17</td>
      <td>2021-04-19 15:23:17</td>
      <td>2021-08-26 23:15:16</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>129</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.063063</td>
      <td>0.00</td>
      <td>209.459459</td>
      <td>209.634249</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0x0000000002032370b971dabd36d72f3e5a7bf1ee</td>
      <td>399</td>
      <td>1.735192e+23</td>
      <td>4.348853e+20</td>
      <td>2021-04-21 21:28:30</td>
      <td>2021-09-01 18:15:24</td>
      <td>104</td>
      <td>15</td>
      <td>250</td>
      <td>0</td>
      <td>...</td>
      <td>132</td>
      <td>0.266667</td>
      <td>4</td>
      <td>0.004103</td>
      <td>0.0</td>
      <td>0.489237</td>
      <td>0.927928</td>
      <td>0.75</td>
      <td>489.215733</td>
      <td>507.852307</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
wallet_score.shape
```




    (3497, 22)




```python
# quick summery 
wallet_score.info()                           
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3497 entries, 0 to 3496
    Data columns (total 22 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   userWallet               3497 non-null   object 
     1   total_transactions       3497 non-null   int64  
     2   total_amount             3497 non-null   float64
     3   avg_amount               3497 non-null   float64
     4   first_txn                3497 non-null   object 
     5   last_txn                 3497 non-null   object 
     6   active_days              3497 non-null   int64  
     7   borrow                   3497 non-null   int64  
     8   deposit                  3497 non-null   int64  
     9   liquidationcall          3497 non-null   int64  
     10  redeemunderlying         3497 non-null   int64  
     11  repay                    3497 non-null   int64  
     12  duration_days            3497 non-null   int64  
     13  repayment_ratio          3497 non-null   float64
     14  unique_actions           3497 non-null   int64  
     15  repayratio_scaled        3497 non-null   float64
     16  liquidationcall_scaled   3497 non-null   float64
     17  deposit_scaled           3497 non-null   float64
     18  activedays_scaled        3497 non-null   float64
     19  uniqueactions_scaled     3497 non-null   float64
     20  normalised_credit_score  3497 non-null   float64
     21  credit_score             3497 non-null   float64
    dtypes: float64(10), int64(9), object(3)
    memory usage: 601.2+ KB
    


```python
# check null values
wallet_score.isnull().sum()
```




    userWallet                 0
    total_transactions         0
    total_amount               0
    avg_amount                 0
    first_txn                  0
    last_txn                   0
    active_days                0
    borrow                     0
    deposit                    0
    liquidationcall            0
    redeemunderlying           0
    repay                      0
    duration_days              0
    repayment_ratio            0
    unique_actions             0
    repayratio_scaled          0
    liquidationcall_scaled     0
    deposit_scaled             0
    activedays_scaled          0
    uniqueactions_scaled       0
    normalised_credit_score    0
    credit_score               0
    dtype: int64




```python
# check for duplicates
duplicate_record = wallet_score.duplicated().sum()
duplicate_record
```




    0



- Data is of type object, int64 and float64
- No null values found
- No duplicates found


```python
wallet_score.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_transactions</th>
      <th>total_amount</th>
      <th>avg_amount</th>
      <th>active_days</th>
      <th>borrow</th>
      <th>deposit</th>
      <th>liquidationcall</th>
      <th>redeemunderlying</th>
      <th>repay</th>
      <th>duration_days</th>
      <th>repayment_ratio</th>
      <th>unique_actions</th>
      <th>repayratio_scaled</th>
      <th>liquidationcall_scaled</th>
      <th>deposit_scaled</th>
      <th>activedays_scaled</th>
      <th>uniqueactions_scaled</th>
      <th>normalised_credit_score</th>
      <th>credit_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3497.000000</td>
      <td>3.497000e+03</td>
      <td>3.497000e+03</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
      <td>3497.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>28.595939</td>
      <td>1.871546e+23</td>
      <td>2.676293e+21</td>
      <td>6.563912</td>
      <td>4.885902</td>
      <td>10.811553</td>
      <td>0.070918</td>
      <td>9.237918</td>
      <td>3.589648</td>
      <td>21.593080</td>
      <td>0.357850</td>
      <td>2.261939</td>
      <td>0.005505</td>
      <td>0.002728</td>
      <td>0.021158</td>
      <td>0.050125</td>
      <td>0.315485</td>
      <td>243.897544</td>
      <td>244.032518</td>
    </tr>
    <tr>
      <th>std</th>
      <td>250.732075</td>
      <td>4.074822e+24</td>
      <td>4.548414e+22</td>
      <td>12.529142</td>
      <td>15.133573</td>
      <td>29.868997</td>
      <td>0.692889</td>
      <td>242.518307</td>
      <td>13.374770</td>
      <td>33.174518</td>
      <td>1.341881</td>
      <td>1.331933</td>
      <td>0.020644</td>
      <td>0.026650</td>
      <td>0.058452</td>
      <td>0.112875</td>
      <td>0.332983</td>
      <td>52.719607</td>
      <td>52.197849</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2.900000e+01</td>
      <td>2.900000e+01</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>184.115693</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>9.499274e+13</td>
      <td>3.661972e+12</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.001957</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>200.293542</td>
      <td>200.293542</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>1.620000e+18</td>
      <td>1.000000e+18</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.003914</td>
      <td>0.009009</td>
      <td>0.250000</td>
      <td>225.293542</td>
      <td>225.293542</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.000000</td>
      <td>1.604850e+21</td>
      <td>8.631680e+19</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>32.000000</td>
      <td>0.530612</td>
      <td>4.000000</td>
      <td>0.008163</td>
      <td>0.000000</td>
      <td>0.011742</td>
      <td>0.036036</td>
      <td>0.750000</td>
      <td>284.443633</td>
      <td>283.243137</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14265.000000</td>
      <td>1.756775e+26</td>
      <td>2.057674e+24</td>
      <td>112.000000</td>
      <td>200.000000</td>
      <td>511.000000</td>
      <td>26.000000</td>
      <td>14265.000000</td>
      <td>291.000000</td>
      <td>153.000000</td>
      <td>65.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>819.451790</td>
      <td>690.243142</td>
    </tr>
  </tbody>
</table>
</div>



- To calculated proxy target valued scaled fetures are used.
- Data shows variability in the user engagement. Some user are highly active and some were active for few days only. It indicated the skewness of data. Which means it have outliers.
- 50% of user were have 0 repayment ratio indicating high credit risk.
- Most of the user have credit score 200 to 300 , that shows they are less creditworthy.

# Credit Score Distribution


```python
score_bins = list(range(0, 1100, 100))
score_labels = [f"{i}-{i+99}" for i in range(0, 1000, 100)]
wallet_score['credit_score_range'] = pd.cut(wallet_score['credit_score'], bins=score_bins, labels=score_labels, include_lowest=True)

# Plot score distribution
score_counts = wallet_score['credit_score_range'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
score_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Credit Score Distribution')
plt.xlabel('Credit Score Range')
plt.ylabel('Number of Wallets')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


    
![png](Analysis_files/Analysis_14_0.png)
    


Low credit score range(0-300):
- Large number of wallet fall in this category
- Users are not engaged for  long time.
- Less number of transactions
- Can include spam, fraud users or new user.
- Show less creditworthiness

High credit score range(300-600):
- Very few user wallet fall in this range
- These users are actively using wallet for transaction, shows continuity in engagement
- Users that have been using services for longertime, can be considered as loyal.
- These users are more loyal and creditworthy.

Medium credit score range(600-1000):
- mixed behaviour
- Can include those who are improving in their engagment


# Corelation


```python
# Droping the unneccessary columns
wallets = wallet_score.drop(columns = ['repayratio_scaled', 'liquidationcall_scaled', 'deposit_scaled', 'activedays_scaled', 'uniqueactions_scaled','credit_score_range'])

```


```python
# selecting only neumeric columns for corelation

numeric_wallets = wallets.select_dtypes(include=['int64', 'float64'])
```


```python
# feature realted to credit_score
numeric_wallets.corr()['credit_score'].sort_values(ascending=False)

```




    credit_score               1.000000
    normalised_credit_score    0.994126
    unique_actions             0.883297
    active_days                0.812875
    duration_days              0.713118
    deposit                    0.691993
    borrow                     0.630029
    repay                      0.600982
    repayment_ratio            0.432759
    total_transactions         0.207275
    liquidationcall            0.076230
    total_amount               0.073742
    redeemunderlying           0.056391
    avg_amount                 0.035517
    Name: credit_score, dtype: float64



Corelation is a measure of relation between two features
+1: strong positve correlation
0: No correlation
-1: strong negaitve correlation

Observation:

- Strong Corelation: normalised_credit_score, unique_actions, active_days, duration_days 
- Moderate Corelation: deposit, borrow, repay, repayment_ratio
- Weak corelation: total_transactions, liquidationcall, total_amount, redeemunderlying, avg_amount 



```python
score_corr = numeric_wallets.corr()[['credit_score']].drop('credit_score').sort_values(by='credit_score', ascending=True)

# Plot heatmap
plt.figure(figsize=(4, len(score_corr)*0.5))
sns.heatmap(score_corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title("Correlation with Credit Score")
plt.show()

```


    
![png](Analysis_files/Analysis_21_0.png)
    



Insight: 

- feature like normalised_credit_score, unique_actions, active_days, duration_days have strong corelation with credit score,hence are most important for credit score prdiction
- User wallets with high unique_actions and active days have high credit score
- liquidationcall, total_amount, redeemunderlying, avg_amount have almost zero corelation with credit score. Change in these feature will not cause any measure change in credit score.

# Distribution of data


```python
strong_cor_column = wallets[['normalised_credit_score','unique_actions', 'active_days', 'duration_days']]
```


```python
# subplot of histograms
plt.figure(figsize=(12, 8))

for i, col in enumerate(strong_cor_column):
    plt.subplot(2, 2, i + 1)
    sns.histplot(wallets[col], kde=True, bins=30, color='skyblue')
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```


    
![png](Analysis_files/Analysis_25_0.png)
    


unique_actions feature is uniformaly distibution which means their is no bias or outlier. unique actions done by users are well distributed around the mean. User engagement is consitant.

normalised_credit_score, active_days and duration_days these features datapoints are right skewed/positive skew which means
- few user have high engagement (shown as peack on left)
- large number of user have very less engagment crepresented as a tail on right side
- user engagement was not uniform and have an outliers 


# Check for Outlier


```python
# subplot of boxplot

plt.figure(figsize=(12, 8))

for x, col in enumerate(strong_cor_column):
    plt.subplot(2, 2, x + 1)
    sns.boxplot(x = wallets[col], color='orange')
    plt.title(f'{col} Boxplot')
    plt.xlabel(col)

plt.tight_layout()
plt.show()
```


    
![png](Analysis_files/Analysis_28_0.png)
    


- unique_actions feature have no outlier, most of the user actions are uniforme

- normalised_credit_score, active_days and duration_days have outlier, user engagement is not contineous.
