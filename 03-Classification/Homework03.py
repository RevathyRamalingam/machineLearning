# %%
import numpy as np
import pandas as pd
print(pd.__version__)

# %%
df=pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv')
df.head().T


# %% [markdown]
# columns are classified as  numerical and categorical based on their column values
# Q1.Retail is the most frequent industry type of the leads

# %%
print(df.isnull().sum())
print(df.dtypes)
categorical_columns=list(df.columns[df.dtypes=='object'])
numerical_columns=list(df.columns[df.dtypes!='object'])


df[categorical_columns] = df[categorical_columns].fillna('NA')
df[numerical_columns] = df[numerical_columns].fillna(0)
df['industry'].mode()



# %% [markdown]
# Q2.Correlation between columns
# 
# interaction_count and lead_score  0.011374
# number_of_courses_viewed and lead_score  0.011529
# number_of_courses_viewed and interaction_count  -0.050187
# annual_income and interaction_count -0.015510

# %%
from sklearn.model_selection import train_test_split
df_fulltrain,df_test=train_test_split(df,test_size=0.2,random_state=42)
df_train,df_val=train_test_split(df_fulltrain,test_size=0.25,random_state=42)
sum_of_df = len(df_test)+len(df_train)+len(df_val)
print(sum_of_df,len(df))
assert len(df) == sum_of_df


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
print(len(df_train),len(df_val),len(df_test))

y_train = df_train['converted'].values
y_val = df_val['converted'].values
y_test= df_test['converted'].values

del df_train['converted']
del df_val['converted']
del df_test['converted']
if 'converted' in numerical_columns:
    numerical_columns.remove('converted')

df_train[numerical_columns].corr()




# %% [markdown]
# Q3.Mutual information Ans: Lead_source

# %%

from sklearn.metrics import mutual_info_score

def mutual_info_score_series(series):
    return mutual_info_score(series,y_train)

mi=df_train[categorical_columns].apply(mutual_info_score_series)
mi.sort_values(ascending=False)

# %%
from IPython.display import display
for c in categorical_columns:
    display(df_fulltrain[c].unique())

# %% [markdown]
# Q4. logistic regression  Ans 0.74

# %%

print(categorical_columns)
print(numerical_columns)
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical_columns + numerical_columns].to_dict(orient = 'records')
print(train_dict)
X_train=dv.fit_transform(train_dict)

val_dict = df_val[categorical_columns + numerical_columns].to_dict(orient = 'records')
X_val=dv.transform(val_dict)

test_dict = df_test[categorical_columns + numerical_columns].to_dict(orient = 'records')
X_test=dv.transform(test_dict)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
Accuracy_score = 0
def buildLogitic_Regression(c):
    model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
    #print(X_train.shape)
    model.fit(X_train,y_train)
    y_pred= model.predict(X_val)
    Accuracy_score =accuracy_score(y_val,y_pred)
    return model,round(Accuracy_score,3)
 
model,Acc=buildLogitic_Regression(1)
print(Acc)
y_fulltrain = np.concatenate([y_train,y_val])
X_fulltrain_dict = train_dict+val_dict
X_fulltrain = dv.fit_transform(X_fulltrain_dict)
print(X_fulltrain.shape)
model.fit(X_fulltrain,y_fulltrain)
y_pred_n = model.predict(X_test)
print(round(accuracy_score(y_test,y_pred_n),2))

# %% [markdown]
# Q5.Feature elimination
# Which of following feature has the smallest difference?
# 
# 'industry'
# 'employment_status'
# 'lead_score'
# 
# Ans :industry

# %%

def calculate_diff_in_Accuracies(X_dataset,X_vald):
    
    model.fit(X_dataset,y_train)
    y_pred = model.predict(X_vald)
    score = accuracy_score(y_val,y_pred)
    return Accuracy_score-score



# %%
totalcolumns = categorical_columns+numerical_columns
differences = {}
for c in totalcolumns:
    df_new = df_train.copy()
    df_val_new =df_val.copy().drop(c,axis=1)
    df_new = df_new.drop(c,axis=1)
    print(c)
    print(df_new.T)
    xnew_dict = df_new.to_dict(orient ='records')
    xval_dic =df_val_new.to_dict(orient ='records')
    X_new_train =dv.fit_transform(xnew_dict)
    x_vald = dv.transform(xval_dic)
    differences[c]=round(calculate_diff_in_Accuracies(X_new_train,x_vald),3)
differences

# %% [markdown]
# Q6.Regularizeed logistic regression
# 
# Ans : all reularizations have same accuracy 0.7
# So smallest c =0.01 is chosen

# %%
for c in [0.01,0.1,1,10,100]:
    model,Accuracy =buildLogitic_Regression(c)
    print(c,Accuracy)


