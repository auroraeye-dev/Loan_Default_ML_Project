import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

loan_df=pd.read_csv("/Users/SatvikMishra/Desktop/python_proj/Loan Default Prediction - Student Files/data/vehicle_loans_clean_myself.csv",index_col= 'UNIQUEID')
print(loan_df.info())

print(loan_df.nunique())
print(loan_df['MOBILENO_AVL_FLAG'].value_counts())

loan_df=loan_df.drop(['MOBILENO_AVL_FLAG'], axis=1)
loan_df=loan_df.drop(['SUPPLIER_ID','CURRENT_PINCODE_ID','EMPLOYEE_CODE_ID','BRANCH_ID'], axis=1)
print(loan_df.info())
print(loan_df[['STATE_ID','MANUFACTURER_ID']].value_counts().sample(10))
print(loan_df[['STATE_ID']].isnull().value_counts())
print(loan_df[['MANUFACTURER_ID']].isnull().value_counts())
print(loan_df['STATE_ID'].nunique())
print(loan_df['MANUFACTURER_ID'].nunique())
print(loan_df['STATE_ID'].value_counts(normalize=True))
print(loan_df['MANUFACTURER_ID'].value_counts(normalize=True))

sns.countplot(x='STATE_ID',data= loan_df)
plt.show()
sns.countplot(x='MANUFACTURER_ID',data= loan_df)
plt.show()

print(loan_df.groupby('MANUFACTURER_ID'))
print(loan_df.groupby('MANUFACTURER_ID').max())
print(loan_df.groupby('MANUFACTURER_ID')['LOAN_DEFAULT'].value_counts(normalize=True).unstack(level=-1))
sns.countplot(x='MANUFACTURER_ID',data=loan_df, hue='LOAN_DEFAULT')
plt.show()

REEDA=['EMPLOYMENT_TYPE','STATE_ID','PERFORM_CNS_SCORE_DESCRIPTION','DISBURSAL_MONTH']
for i in REEDA :
    print(i)
    print(loan_df.groupby(i))
    print(loan_df.groupby(i).max())
    print(loan_df.groupby(i)['LOAN_DEFAULT'].value_counts(normalize=True).unstack(level=0))
    sns.countplot(x=i,hue='LOAN_DEFAULT',data=loan_df)
    plt.show()
    print("-----------------------------------")
    
for j in REEDA:
    if i==j:
        continue
    else:
        print("Relationship between ", i ,"&", j)
        print(loan_df.groupby(i)[j].value_counts(normalize=True).unstack(level=0))
        sns.countplot(x=j,hue=i,data=loan_df)
        plt.show()
        print("***********************************")
print(len(REEDA))

def relations(i):
           print(loan_df[i].head())
           print(loan_df.groupby(i)['LOAN_DEFAULT'].value_counts(normalize=True).unstack(level=0))
           '''sns.countplot(x=i,hue='LOAN_DEFAULT',data=loan_df)
           plt.show()'''

print(relations('EMPLOYMENT_TYPE'))

print(loan_df['AGE'].describe())
sns.boxplot(y='AGE', data=loan_df)
plt.show()

sns.displot(loan_df['AGE'], kde=True)
plt.show()

print(loan_df['AGE'].groupby(loan_df['LOAN_DEFAULT']).describe())
sns.boxplot(x='LOAN_DEFAULT', y='AGE', data=loan_df,orient='v')
plt.show()

def relational_summary(col):
          print(loan_df[col].describe())
          print(loan_df[col].groupby(loan_df['LOAN_DEFAULT']).describe())
          sns.boxplot(x='LOAN_DEFAULT', y=col, data=loan_df,orient='h')
          plt.show()
          sns.displot(loan_df[col], kde=True)
          plt.show()
          sns.displot(loan_df,x=col, y= 'LOAN_DEFAULT',hue='LOAN_DEFAULT',kind='kde')
          plt.show()
        

print(relational_summary('DISBURSED_AMOUNT'))

print(relational_summary('AADHAR_FLAG'))
loan_df.to_csv("/Users/SatvikMishra/Desktop/python_proj/Loan Default Prediction - Student Files/data/vehicle_loans_eda_myself.csv")

