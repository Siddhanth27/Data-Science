import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
advertise = pd.read_csv("C:/users/veeru/advertising.csv", sep = ",")

c1 = advertise.drop('daily_time', axis = 1)
c1.head(11)
c1.describe()
c1.isna().sum()

# To drop NaN values
df = advertise.dropna()

# Imputating the missing values           


mean_value = c1.Timestamp.mean()
mean_value
c1.timestamp = c1.timestamp.fillna(mean_value)
c1.timestamp.isna().sum()

# For Median imputation try this
# median_value = claimants.CLMAGE.median()
# claimants1['CLMAGE'] = claimants1['CLMAGE'].fillna(median_value)


# For Mode - for Discrete variables
mode_area_income = c1.area_income.mode()
mode_area_income
c1.area_income = c1.area_income.fillna((mode_area_income)[0])
c1.area_income.isna().sum()

mode_daily_internet_usage = c1['daily_internet_usage'].mode()
mode_internet
c1['daily_internet_usage'] = c1['daily_internet_usage'].fillna((mode_internet)[0])
c1.daily_internet_usage.isna().sum()

mode_SB = c1['ad_topic_line'].mode()
mode_SB
c1['ad_topic_line'] = c1['ad_topic_line'].fillna((mode_SB)[0])
c1.ad_topic_line.isna().sum()

# Alternate approach
########## Median Imputation for all the columns ############
c1.fillna(c1.median(), inplace=True)
c1.isna().sum()

c1.timestamp.median()
c1.ad_topic_line.median()
#############################################################

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('Age ~ timestamp + clicked_on_ad + ad_topic_line + daily_internet_usage + country', data = c1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(c1.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(c1.Age, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
c1["pred"] = np.zeros(1340)
# taking threshold value and above the prob value will be treated as correct value 
c1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(c1["pred"], c1["Age"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(c1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('Age ~ timestamp + clicked_on_ad + ad_topic_line + daily_internet_usage + country', data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(402)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Age'])
confusion_matrix

accuracy_test = (131 + 155)/(402) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Age"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Age"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(938)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['ATTORNEY'])
confusion_matrx

accuracy_train = (321 + 356)/(938)
print(accuracy_train)
