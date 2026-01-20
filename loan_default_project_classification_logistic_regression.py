import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

loan_df=pd.read_csv("/Users/SatvikMishra/Desktop/python_proj/Loan Default Prediction - Student Files/data/vehicle_loans_feature_engineered_myself.csv")
print(loan_df.info())

category_col=['MANUFACTURER_ID','STATE_ID','DISBURSAL_MONTH','DISBURSED_CAT','PERFORM_CNS_SCORE_DESCRIPTION','EMPLOYMENT_TYPE']
print(loan_df[category_col].dtypes)

for col in loan_df.columns:
    if loan_df[col].dtype=='object':
        print(f"Column: {col}")
    else:
        continue

loan_df[category_col]=loan_df[category_col].astype('category')
print(loan_df[category_col].dtypes)

summary_cols=['STATE_ID','LTV','DISBURSED_CAT','PERFORM_CNS_SCORE','DISBURSAL_MONTH','LOAN_DEFAULT']
loan_df_summary=loan_df[summary_cols]
print(loan_df_summary.shape)

print(loan_df_summary.columns)
print(loan_df_summary.info())

x=loan_df_summary.drop(['LOAN_DEFAULT'],axis=1)
y=loan_df_summary['LOAN_DEFAULT']
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=90)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(X_train.info())
print(X_test.info())
print(Y_train.describe())
print(Y_test.info())
#using pd.get_dummies to convert categorical variables to dummy/indicator variables
loan_data_dum=pd.get_dummies(loan_df_summary,prefix='_',drop_first=True)
print(loan_data_dum.info())
print(loan_data_dum.head())
print(loan_data_dum['LOAN_DEFAULT'].value_counts())
print(loan_data_dum['__60k-75k'].value_counts())

x=loan_data_dum.drop(['LOAN_DEFAULT'],axis=1)
y=loan_data_dum['LOAN_DEFAULT']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=90)
print(x_train.shape)
print(x_test.shape)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))


Logistic_model=LogisticRegression(max_iter=10000)
Logistic_model.fit(x_train,y_train)
prediction=Logistic_model.predict(x_test)
print(prediction)

print(Logistic_model.score(x_test,y_test))


