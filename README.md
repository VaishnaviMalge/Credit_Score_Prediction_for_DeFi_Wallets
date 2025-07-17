
## Credit Score Prediction for DeFi Wallets (Aave v2 Protocol)

>problem statement: https://zerufinance.notion.site/Problem-statement-1-2300a8e4815880bc86b5ddc97b3d8cfd

### Objective

To build a ML model that predicts the credit score for each wallet to know the credit worthiness of a user.


### Methodology(For model building)

1 .  **EDA**: Checked for missing values, duplicates, datatypes. Used different plot to know user behaviour at different credit score and features that are highly correlated with credit score    
2 .  **Feature Engineering**: Created new feature that can be useful to calculate the proxy target variable and further modeling    
3 .  **Predicting Target Variables**: Predicted Target Variables to train models.    
4 .  **Model selection**: As the target variables are continuos experimented with different regression algorithm and choose the most reliable one(Random Forest) with high R^2 score and less errors    
5 .  **Calculating Credit Score**: Using model predicted credit scores. Used a whole x independent variable cause we need to create a whole new column for credit score.    
6 .  **Save file**: Saved the file with all features including credit score. 
7 .   **Analysis**: Performed an analysis to find the user behaviour for different ranges of credit score.

### Why Regression
 The target variable is contineaous
 Supervised learning can give more accurate prediction than Unsupervised learning
 Regression can give credit score between 0 - 1000 for each row. This will be helpful to decide creditworthiness in detailed way
 

### Architecture

-   Raw wallet transaction data→ Cleaned Data    
-   Cleaned Data→ Feature Engineering    
-   Feature Engineering→ Modeling(Regression)
-   Modeling(Regression) → Credit Score Prediction    
-   Credit Score Prediction→ Saving File with Credit Score as new column    
-   Saving File with Credit Score as new column→ Analysis    
-   Analysis → Score Distribution Graph    
-   Score Distribution Graph → Insights
    

### Tools Used

-   Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
-   VS code for development and visualization
-   Stackedit for markdown
    

### Improvements:

Here are some suggested improvement that couldn’t be performed due to time limitation:

1 .   **Time aware data Splitting**
Data split can be done according to time to make the model more reliable for real world scenarios. This ensures the model is trained on only historical data and reduces risk of leakage.

2 .    **Unsupervised clustering for Wallet segmentation**  
Unsupervised clustering can be performed (k-means) to create segments for wallets to show different behaviour groups based on the transaction patterns.

3 .    **Hyperparameter tuning**
Best hyperparameter can be found using GridsearchCV or RandmizedSearchCV these can be used to get more accurate prediction. 
Note: In provided .py file only main coding is given with a finalised model only. detail.ipynb file which include datacleaning whole process(.describe(), .inf(), etc.) and different sample models along with calculation of r2 MAE and RMSE for each value can be provided on demand.
