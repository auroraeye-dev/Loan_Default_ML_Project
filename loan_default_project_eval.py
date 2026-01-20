import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, roc_curve, auc, precision_score, ConfusionMatrixDisplay

loan_df=pd.read_csv("/Users/SatvikMishra/Desktop/python_proj/Loan Default Prediction - Student Files/data/vehicle_loans_feat.csv",index_col='UNIQUEID')

category_cols=['MANUFACTURER_ID','STATE_ID','DISBURSAL_MONTH','DISBURSED_CAT','PERFORM_CNS_SCORE_DESCRIPTION','EMPLOYMENT_TYPE']
loan_df[category_cols]=loan_df[category_cols].astype('category')
small_loan_df=loan_df[['STATE_ID','LTV','DISBURSED_CAT','PERFORM_CNS_SCORE','DISBURSAL_MONTH','LOAN_DEFAULT']]
print(small_loan_df.info())
small_loan_df_dumm=pd.get_dummies(small_loan_df,prefix='_',drop_first=True)
print(small_loan_df_dumm.info())
print(small_loan_df_dumm.head())

Logistic_model=LogisticRegression()
x=small_loan_df_dumm.drop(['LOAN_DEFAULT'],axis=1)
y=small_loan_df_dumm['LOAN_DEFAULT']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=90)
Logistic_model.fit(x_train,y_train)
prediction=Logistic_model.predict(x_test)
print(prediction)

print(Logistic_model.score(x_test,y_test))

conf_matrix=confusion_matrix(y_test,prediction)
print(conf_matrix)
tn=conf_matrix[0][0]
tp=conf_matrix[1][1]
fn=conf_matrix[1][0]
fp=conf_matrix[0][1]
print("True Negative:",tn)
print("True Positive:",tp)  
print("False Negative:",fn)
print("False Positive:",fp)

conf=ConfusionMatrixDisplay(conf_matrix).plot()
conf.plot()
plt.show()

print("precision score of the model is : {}".format(precision_score(y_test,prediction)))
print("recall score of the model is : {}".format(recall_score(y_test,prediction)))
print("f1 score of the model is : {}".format(f1_score(y_test,prediction)))
print("accuracy score of the model is : {}".format(accuracy_score(y_test,prediction)))
probabilities=Logistic_model.predict_proba(x_test)
print(probabilities)
print(probabilities.shape)
print(len(x_test))
print(probabilities[:,0])
print(probabilities[:,1])
probabilities_df=pd.DataFrame()
probabilities_df['probs 0']=probabilities[:,0]
probabilities_df['probs 1']=probabilities[:,1]
print(probabilities_df.head())
print(probabilities_df.describe())
fpr, tpr, threshold =roc_curve(y_test,probabilities[:,1],pos_label=1)
print(fpr)
print(tpr)
print(threshold)
roc_df=pd.DataFrame({'FPR':fpr,'TPR':tpr,'Threshold':threshold})
print(roc_df.head())
print(roc_df.describe())
roc_auc=auc(fpr,tpr)
print("Area under the curve is : {}".format(roc_auc))

def roc_curve_fig(fpr,tpr,roc_auc):
    plt.title("receiver operating characteristic")
    plt.plot(fpr,tpr,label="AUC = %0.2f"%roc_auc)
    plt.legend(loc="lower right")
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()
roc_curve_fig(fpr,tpr,roc_auc)

result_df=pd.DataFrame()
result_df['True_class']=y_test
result_df['predicted_class']=list(prediction)
result_df['default_prob']= probabilities[:,1] 


print(result_df.groupby('True_class')['predicted_class'].value_counts(normalize=True))

default_probability_f=result_df[result_df['True_class']==0]['default_prob']
print(default_probability_f.describe())
default_probability_t=result_df[result_df['True_class']==1]['default_prob']
print(default_probability_t.describe())

sns.displot(default_probability_f,kde=True)
sns.displot(default_probability_t,kde=True)
plt.legend()
plt.show()


