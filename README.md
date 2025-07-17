
## Credit Score Prediction for DeFi Wallets (Aave v2 Protocol)

>
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
