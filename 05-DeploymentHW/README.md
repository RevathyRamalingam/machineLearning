CREATING a supervised ML MODEL
CREATING a supervised ML MODEL
DATA CLEANING
1.Data cleaning and Analysis is the first important step in building the model.
Replace nullvalues with 0 or mean of the data.
Exploratory DataAnalysis EDA
    Checking missing values
    Looking at the distribution of the target variable (churn)
    Looking at numerical and categorical variables
RISK RATIO 
   By calculating the mean of target variable of a single feature against the global mean of target variable, we can see how much it is correlated with target value .
   Risk ratio 0-0.2 or 0to -0.2 representlow risk
   Medium risk 0.2-0.5 or -0.2 to -0.5
   High risk 0.5-1or -0.5 to -1
   By examing this we can check the model with and without eliminating the feature .
MUTUAL INFORMATION
    Mutual information score tell us how two features are inter related. How eliminating one feature can impact the other one.A Correlation matrix can help us identify the correlation between the features.
    
For numerical values, scaling can be done and categorical(text) values should be oneHotEncoded.
FEATURE ENGINEERING
Adding new features derived from existing features such as age of cars derived from car manufacture date can improve model performance in car prediction usecase.
REGULARIZATION
It is like a chain for a dog,the chain neither makes the dog run too fast(overfitting) and still allowing it to explore(fits the data).
Lower the regularization value ,the better is the model. So, we must always choose smallest regularization value.
This is hyperparameter in model tuning.
METRICS TO EVALUATE MODEL
1.RootMeanSquareError(RMSE)
RMSE value is a way to evaluate linear regression models where target value is a number.Lesser RMSE value indicates a good model.

y_true = [100, 200, 300]
y_pred = [90, 210, 310]
RMSE =10
RMSE =10 indicates model is off by 10(on an average)

2.MODEL ACCURACY

Model accuracy gives the  percentage of correct preditions over actual correct values. 


This shows how ML models can be deployed in Docker 

It uses uvicorn to package all the python dependencies such as scikit-learn, fastapi
The pipeline(DictVectorizer+Model) is saved and loaded via pickle
There is a python script that loads the pipeline and predicts the lead conversion probability