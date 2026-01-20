import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
loan_df=pd.read_csv(r"/Users/SatvikMishra/Desktop/python_proj/Loan Default Prediction - Student Files/data/vehicle_loans_eda_myself.csv")


def relational_summary(col):
          print(loan_df[col].value_counts())
          print(loan_df[col].value_counts(normalize=True))
          print(loan_df.groupby(col)['LOAN_DEFAULT'].value_counts().unstack(level=-1))
          print(loan_df.groupby(col)['LOAN_DEFAULT'].value_counts(normalize=True).unstack(level=-1))
          '''sns.catplot(data=loan_df,kind='count',x=col,hue='LOAN_DEFAULT')
          plt.show()'''

    
relational_summary('DISBURSED_AMOUNT')
print(loan_df['DISBURSED_AMOUNT'].idxmax())
print(loan_df.loc[loan_df['DISBURSED_AMOUNT'].idxmax()])
print(loan_df['DISBURSED_AMOUNT'].idxmin())
print(loan_df.loc[loan_df['DISBURSED_AMOUNT'].idxmin()])

#creating buckets 
disbursed_buckets=[13000,30000,45000,60000,750000,900000,1050000]
disbursed_labels=['13k-30k','30k-45k','45k-60k','60k-75k','75k-90k','90k-105M']
loan_df['DISBURSED_CAT']=pd.cut(loan_df['DISBURSED_AMOUNT'],bins=disbursed_buckets,labels=disbursed_labels)
print(loan_df[['DISBURSED_AMOUNT','DISBURSED_CAT']])
relational_summary('DISBURSED_CAT')

loan_df['DISBURSAL_DIFFERENCE']=-(loan_df['DISBURSED_AMOUNT']-loan_df['ASSET_COST'])
print(loan_df[['DISBURSED_AMOUNT','ASSET_COST','DISBURSAL_DIFFERENCE']].head())

def graph_summary(col):
        print(loan_df[col].describe())
        print(loan_df[col].value_counts())
        sns.boxplot(data=loan_df,x=col)
        plt.show()
        sns.histplot(data=loan_df,x=col,kde=True)
        plt.show()
        sns.displot(data=loan_df,x=col,hue='LOAN_DEFAULT',kde=True)
        plt.show()
        print(loan_df.groupby(col)['LOAN_DEFAULT'].value_counts(normalize=True).unstack(level=-1))

print(graph_summary('PRI_NO_OF_ACCTS'))
graph_summary('PRI_CURRENT_BALANCE')
graph_summary('SEC_NO_OF_ACCTS')
graph_summary('SEC_CURRENT_BALANCE')
graph_summary('PRIMARY_INSTAL_AMT')
graph_summary('SEC_INSTAL_AMT') 

loan_df['TOTAL_ACCTS']=loan_df['PRI_NO_OF_ACCTS']+loan_df['SEC_NO_OF_ACCTS']
print(loan_df[['PRI_NO_OF_ACCTS','SEC_NO_OF_ACCTS','TOTAL_ACCTS']].head(10))

Primary=['PRI_ACTIVE_ACCTS','PRI_OVERDUE_ACCTS','PRI_CURRENT_BALANCE','PRI_SANCTIONED_AMOUNT','PRI_DISBURSED_AMOUNT','PRIMARY_INSTAL_AMT']
Secondary=['SEC_ACTIVE_ACCTS','SEC_OVERDUE_ACCTS','SEC_CURRENT_BALANCE','SEC_SANCTIONED_AMOUNT','SEC_DISBURSED_AMOUNT','SEC_INSTAL_AMT']
Total=['TOTAL_ACTIVE_ACCTS','TOTAL_OVERDUE_ACCTS','TOTAL_CURRENT_BALANCE','TOTAL_SANCTIONED_AMOUNT','TOTAL_DISBURSED_AMOUNT','TOTAL_INSTAL_AMT']
def total_columns(i):
        for i in range(len(Primary)):
                loan_df[Total[i]]=loan_df[Primary[i]]+loan_df[Secondary[i]]
                
total_columns(0)
total_columns(1)
total_columns(2)
total_columns(3)
total_columns(4)
total_columns(5)
print(loan_df['TOTAL_DISBURSED_AMOUNT'].head())
print(loan_df['TOTAL_SANCTIONED_AMOUNT'].head())
print(loan_df['TOTAL_CURRENT_BALANCE'].head())
print(loan_df['TOTAL_INSTAL_AMT'].head(9))
print(loan_df[['TOTAL_OVERDUE_ACCTS', 'TOTAL_ACTIVE_ACCTS']].head())

loan_df=loan_df.drop(['PRI_NO_OF_ACCTS','SEC_NO_OF_ACCTS','PRI_ACTIVE_ACCTS','SEC_ACTIVE_ACCTS','PRI_OVERDUE_ACCTS','SEC_OVERDUE_ACCTS','PRI_CURRENT_BALANCE','SEC_CURRENT_BALANCE','PRI_SANCTIONED_AMOUNT','SEC_SANCTIONED_AMOUNT','PRI_DISBURSED_AMOUNT','SEC_DISBURSED_AMOUNT','PRIMARY_INSTAL_AMT','SEC_INSTAL_AMT'],axis=1)
print(loan_df.columns)

print(loan_df['TOTAL_OVERDUE_ACCTS'].value_counts(normalize=True))

loan_df['OVERDUE_PCT']=loan_df['TOTAL_OVERDUE_ACCTS']/loan_df['TOTAL_ACCTS']
print(loan_df[['TOTAL_OVERDUE_ACCTS','TOTAL_ACCTS','OVERDUE_PCT']].sample(10))

print(loan_df['OVERDUE_PCT'].isnull().value_counts())
loan_df['OVERDUE_PCT']=loan_df['OVERDUE_PCT'].fillna(0)
print(loan_df['OVERDUE_PCT'].isnull().value_counts())
print(loan_df['OVERDUE_PCT'].value_counts())
print(loan_df.info())

numeric_col=['DISBURSED_AMOUNT', 'ASSET_COST', 'LTV',
        'NEW_ACCTS_IN_LAST_SIX_MONTHS',
       'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS',
       'NO_OF_INQUIRIES',  'AGE',
        'AVERAGE_ACCT_AGE_MONTHS',
        'PERFORM_CNS_SCORE',
       'CREDIT_HISTORY_LENGTH_MONTHS', 'DISBURSAL_DIFFERENCE',
       'OVERDUE_PCT',
       'TOTAL_ACCTS', 'TOTAL_ACTIVE_ACCTS', 'TOTAL_OVERDUE_ACCTS',
       'TOTAL_CURRENT_BALANCE', 'TOTAL_SANCTIONED_AMOUNT',
       'TOTAL_DISBURSED_AMOUNT', 'TOTAL_INSTAL_AMT']

'''loan_df[['LTV','ASSET_COST','AGE']].boxplot()
plt.show()'''

mm_scaler=MinMaxScaler() #minmax scaler object
loan_df[numeric_col]=mm_scaler.fit_transform(loan_df[numeric_col])
loan_df[['LTV','ASSET_COST','AGE']].boxplot()
plt.show()


loan_df.to_csv(r"/Users/SatvikMishra/Desktop/python_proj/Loan Default Prediction - Student Files/data/vehicle_loans_feature_engineered_myself.csv")