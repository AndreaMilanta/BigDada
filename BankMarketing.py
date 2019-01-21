#------------ IMPORTS ----------------------------------#
#-------------------------------------------------------#
import numpy as np
import pandas as pd
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import seaborn as sns
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt

# splitting and crossvalidation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


#                                                       #
#------------ CONSTANTS --------------------------------#
#-------------------------------------------------------#
# BankMarketingURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
# BankMarketingName = "bank-full.csv"
BankMarketingName = "bank.csv"

#                                                       #
#------------ DATASET IMPORT ---------------------------#
#-------------------------------------------------------#
#
# import dataset from URL and open it as dataframe
# bmzip = urlopen(BankMarketingURL)
# bmcsv = ZipFile(BytesIO(bmzip.read())).extract(BankMarketingName)
# df = pd.read_csv(bmcsv, sep=';');
df = pd.read_csv(BankMarketingName, sep=';');

#
#----------- ENUM to INTEGER ---------------------------#
#-------------------------------------------------------#
#
# function which converts month string to int
def month2int(month):
    if month == "jan":      return 1
    elif month == "feb":    return 2
    elif month == "mar":    return 3
    elif month == "apr":    return 4
    elif month == "may":    return 5
    elif month == "jun":    return 6
    elif month == "jul":    return 7
    elif month == "aug":    return 8
    elif month == "sep":    return 9
    elif month == "oct":    return 10
    elif month == "nov":    return 11
    else:                   return 12

# function which categorizes age into four group
def age2group(age):
    if age <= 29:   return "young"    # [18-29] - usually living alone, without family and stable income
    elif age <= 44: return "adult"    # [30-44] - starting family and incresing salary
    elif age <= 65: return "middleage"    # [45-60] - working, stable salary
    else:           return "senior"    # [61-99] - pension age

# function which categorizes balance into four group
def balance2group(balance):
    if balance <= 0 or balance == "balance":
        return "no"
    elif balance <= 1000:
        return "low"
    elif balance <= 5000:
        return "medium"
    else:
        return "high"

# Months
df["month"] = df["month"].apply(month2int).astype(np.int64) 
    
# convert bynary values (yes/no) to bools (yes -> 1)
df["housing"] = df["housing"].apply(lambda x: 0 if x == 'no' else 1).astype(np.int64)
df["subscribed"] = df["y"].apply(lambda x: 0 if x == 'no' else 1).astype(np.int64)
df["loan"] = df["loan"].apply(lambda x: 0 if x == 'no' else 1).astype(np.int64)
df["default"] = df["default"].apply(lambda x: 0 if x == 'no' else 1).astype(np.int64)

# discretization
df["age"] = df["age"].apply(age2group)
df["balance"] = df["balance"].apply(balance2group)

# convert target class to binary values and drop original
df["subscribed"] = df["y"].apply(lambda x: 0 if x == 'no' else 1).astype(np.int64)
df.drop("y", axis=1, inplace=True);

df.drop(["pdays","previous","poutcome","campaign","day","contact","duration"], axis=1, inplace=True)

# df = df.apply(LabelEncoder().fit_transform)
# print(df['subscribed'].value_counts()/len(df))


#                                                       #
#------------ FEATURE ANALYSIS -------------------------#
#-------------------------------------------------------#
#
# correlation matrix
crmtx = df.corr();
# print(crmtx);
# sns.heatmap(crmtx,annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
# fig=plt.gcf()
# fig.set_size_inches(18,15)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.show()
# fig.savefig('correlation matrix.svg')     Save correlation matrix figure

# housing loan - subscription correlation
crs_housing = pd.crosstab(df['subscribed'], df['housing']).apply(lambda x: x/x.sum() * 100)


#                                                       #
#------------ CLASSIFICATION X-VALIDATION --------------#
#-------------------------------------------------------#
#
#setup train
x_train = df.drop("subscribed", axis=1).apply(LabelEncoder().fit_transform)
y_train = df["subscribed"] 

# set fixed crossvalidation param
shuffle = StratifiedKFold(n_splits=10, shuffle=True, random_state=125)

# Logistic Regression - CROSS VALIDATION
lr_clf = LogisticRegression(solver='liblinear')
lr_pred = cross_val_predict(lr_clf, x_train, y_train, cv=shuffle)
lr_acc = accuracy_score(y_train, lr_pred)
lr_prec = precision_score(y_train, lr_pred)
lr_recall = recall_score(y_train, lr_pred)
lr_f1 = f1_score(y_train, lr_pred)

# Decision Tree
dt_clf = tree.DecisionTreeClassifier()
dt_pred = cross_val_predict(dt_clf, x_train, y_train, cv=shuffle)
dt_acc = accuracy_score(y_train, dt_pred)
dt_prec = precision_score(y_train, dt_pred)
dt_recall = recall_score(y_train, dt_pred)
dt_f1 = f1_score(y_train, dt_pred)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=18)
rf_pred = cross_val_predict(rf_clf, x_train, y_train, cv=shuffle)
rf_acc = accuracy_score(y_train, rf_pred)
rf_prec = precision_score(y_train, rf_pred)
rf_recall = recall_score(y_train, rf_pred)
rf_f1 = f1_score(y_train, rf_pred)


#                                                       #
#------------ RESULTS PRESENTATION ---------------------#
#-------------------------------------------------------#
#
# Values
print("Logistic Regression:\n\taccuracy: %2.2f \n\tprecision: %2.2f\n\trecall: %2.2f\n\tf1: %2.2f" % (lr_acc, lr_prec, lr_recall, lr_f1))
print("Decision Tree:\n\taccuracy: %2.2f \n\tprecision: %2.2f\n\trecall: %2.2f\n\tf1: %2.2f" % (dt_acc, dt_prec, dt_recall, dt_f1))
print("Random Forest:\n\taccuracy: %2.2f \n\tprecision: %2.2f\n\trecall: %2.2f\n\tf1: %2.2f" % (rf_acc, rf_prec, rf_recall, rf_f1))

# feature importance
dt_clf.fit(x_train, y_train)
importances = dt_clf.feature_importances_
feature_names = x_train.columns
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")       
for f in range(x_train.shape[1]):
    print("\t%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# Plot the feature importances of the forest
def feature_importance_graph(indices, importances, feature_names):
    plt.figure(figsize=(12,6))
    plt.title("Determining Feature importances \n with DecisionTreeClassifier", fontsize=18)
    plt.barh(range(len(indices)), importances[indices], color='#31B173',  align="center")
    plt.yticks(range(len(indices)), feature_names[indices], rotation='horizontal',fontsize=14)
    plt.ylim([-1, len(indices)])
    # plt.axhline(y=1.85, xmin=0.21, xmax=0.952, color='k', linewidth=3, linestyle='--')
feature_importance_graph(indices, importances, feature_names)
plt.show()

# Logistic Regression confusion matrix
lr_conf = confusion_matrix(y_train, lr_pred)
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(lr_conf, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.title("Logistic Regression Confusion Matrix", fontsize=20)
plt.subplots_adjust(left=0.22, right=0.99, bottom=0.05, top=0.9)
ax.set_yticks(np.arange(lr_conf.shape[0]) + 0.5, minor=False)
ax.set_xticklabels("")
ax.set_yticklabels(['Refused T. Deposits', 'Accepted T. Deposits'], fontsize=16, rotation=360)
# plt.show()


# Decision Tree confusion matrix
dt_conf = confusion_matrix(y_train, dt_pred)
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(dt_conf, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.title("Decision Tree Confusion Matrix", fontsize=20)
plt.subplots_adjust(left=0.22, right=0.99, bottom=0.05, top=0.9)
ax.set_yticks(np.arange(dt_conf.shape[0]) + 0.5, minor=False)
ax.set_xticklabels("")
ax.set_yticklabels(['Refused T. Deposits', 'Accepted T. Deposits'], fontsize=16, rotation=360)
# plt.show()


# Logistic Regression confusion matrix
rf_conf = confusion_matrix(y_train, rf_pred)
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(rf_conf, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.title("Random Forest Confusion Matrix", fontsize=20)
plt.subplots_adjust(left=0.22, right=0.99, bottom=0.05, top=0.9)
ax.set_yticks(np.arange(rf_conf.shape[0]) + 0.5, minor=False)
ax.set_xticklabels("")
ax.set_yticklabels(['Refused T. Deposits', 'Accepted T. Deposits'], fontsize=16, rotation=360)
# plt.show()


#                                                       #
#------------ TRAIN TEST SPLIT -------------------------#
#-------------------------------------------------------#
#
# # get train and test dataset stratified
# stratifyfeature = "subscribed"      # feature used for stratification
# splitters = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=425)
# for train_set, test_set in splitters.split(df, df[stratifyfeature]):
#     train = df.loc[train_set]
#     test = df.loc[test_set]

# y_train = train["subscribed"]
# y_test = test["subscribed"]

# x_train = train.drop("subscribed", axis=1).apply(LabelEncoder().fit_transform)
# x_test = test.drop("subscribed", axis=1).apply(LabelEncoder().fit_transform)


#                                                       #
#------------ CLASSIFICATION STANDARD ------------------#
#-------------------------------------------------------#
#
# # logistic regression - STANDARD
# LRclassifier = LogisticRegression()
# LRclassifier.fit(x_train,y_train)
# LRtrain_score = LRclassifier.score(x_train, y_train)
# LRtest_score = LRclassifier.score(x_test, y_test)

# print("train score: " + str(LRtrain_score))
# print("test score: " + str(LRtest_score))
