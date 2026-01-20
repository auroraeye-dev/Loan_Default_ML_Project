import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
loan_df=pd.read_csv("data/vehicle_loans.csv",index_col= 'UNIQUEID')
print(loan_df.head())
print(loan_df.info())
print(loan_df.describe())
print(loan_df.shape)
print("number of rows : ", loan_df.shape[0])
print("number of columns :", loan_df.shape[1])

loan_df['DISBURSAL_DATE']= pd.to_datetime(loan_df['DISBURSAL_DATE'], format= '%d-%m-%Y')
print("maximum disbursal date is :", loan_df['DISBURSAL_DATE'].max())
print("minimum disbursal date is :", loan_df['DISBURSAL_DATE'].min())
print("timespan of max and min disbursal date is :", loan_df['DISBURSAL_DATE'].max()- loan_df['DISBURSAL_DATE'].min())

print(loan_df['LOAN_DEFAULT'].value_counts())
print(loan_df['LOAN_DEFAULT'].value_counts(normalize=True))
sns.countplot(x='LOAN_DEFAULT', data=loan_df)
plt.show()

print(loan_df.isnull().any())
print(loan_df.isnull().sum())
print(loan_df['EMPLOYMENT_TYPE'].value_counts())
sns.countplot(x='EMPLOYMENT_TYPE', data=loan_df)
plt.show()

loan_df=loan_df.fillna(value={'EMPLOYMENT_TYPE': 'Missing'})
print(loan_df['EMPLOYMENT_TYPE'].value_counts())
print(loan_df[['DISBURSAL_DATE','DATE_OF_BIRTH']].sample(10))

loan_df['DATE_OF_BIRTH']= pd.to_datetime(loan_df['DATE_OF_BIRTH'], format= '%d-%m-%Y')
print(loan_df['DATE_OF_BIRTH'])
loan_df['AGE']= pd.to_datetime(loan_df['DISBURSAL_DATE'], format='%d-%m-%Y').dt.year - loan_df['DATE_OF_BIRTH'].dt.year
print(loan_df['AGE'].head())

'''or can do by 
loan_df['AGE']= loan_df['AGE']// np.timedelta64(1,'Y')
'''
loan_df['DISBURSAL_DATE']= pd.to_datetime(loan_df['DISBURSAL_DATE'], format= '%d-%m-%Y')
loan_df['DISBURSAL_MONTH']= loan_df['DISBURSAL_DATE'].dt.month
print(loan_df[['DISBURSAL_DATE','DISBURSAL_MONTH', 'AGE']])

loan_df['DISBURSAL_MONTH']=loan_df['DISBURSAL_MONTH'].astype('int')
print(loan_df['DISBURSAL_MONTH'].dtype)

print(loan_df['DISBURSAL_MONTH'].value_counts())
loan_df= loan_df.drop(columns=['DATE_OF_BIRTH','DISBURSAL_DATE'])

print(loan_df[['CREDIT_HISTORY_LENGTH','AVERAGE_ACCT_AGE']].sample(10))

def calc_months(x):
    if isinstance(x, list):
        if len(x) == 2:
            years, months = x
        elif len(x) == 1:
            years, months = x[0], 0
        else:
            return np.nan
        return int(years) * 12 + int(months)
    return np.nan

    
loan_df['AVERAGE_ACCT_AGE_MONTHS']= loan_df['AVERAGE_ACCT_AGE'].str.findall('\d+')
print(loan_df['AVERAGE_ACCT_AGE_MONTHS'].head())
loan_df['AVERAGE_ACCT_AGE_MONTHS']= loan_df['AVERAGE_ACCT_AGE_MONTHS'].map(calc_months)
print(loan_df['AVERAGE_ACCT_AGE_MONTHS'].head())


loan_df['CREDIT_HISTORY_LENGTH']=loan_df['CREDIT_HISTORY_LENGTH'].str.findall('\d+')
loan_df['CREDIT_HISTORY_LENGTH_MONTHS']= loan_df['CREDIT_HISTORY_LENGTH'].map(calc_months)
print(loan_df['CREDIT_HISTORY_LENGTH_MONTHS'].head())
age_columns= ['AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH']
for col in age_columns:
    
    loan_df[col]=loan_df[col].str.findall('\d+')
    loan_df[col + '_MONTHS']= loan_df[col].map(calc_months)
    loan_df= loan_df.drop(columns=[col])
    print(loan_df[col + '_MONTHS'].head())

print(loan_df['PERFORM_CNS_SCORE_DESCRIPTION'].value_counts())

loan_df.to_csv('/Users/SatvikMishra/Desktop/python_proj/Loan Default Prediction - Student Files/data/vehicle_loans_clean_myself.csv', index_label='UNIQUEID')

