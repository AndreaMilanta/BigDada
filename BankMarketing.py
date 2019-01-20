#------------ IMPORTS ----------------------------------#
#-------------------------------------------------------#
import numpy as np
import pandas as pd
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

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
BankMarketingURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
# BankMarketingName = "bank-full.csv"
BankMarketingName = "bank.csv"

#                                                       #
#------------ DATASET IMPORT ---------------------------#
#-------------------------------------------------------#
#
# import dataset from URL and open it as dataframe
bmzip = urlopen(BankMarketingURL)
bmcsv = ZipFile(BytesIO(bmzip.read())).extract(BankMarketingName)
df = pd.read_csv(bmcsv, sep=';');

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

#                                                       #
#------------ FEATURE ANALYSIS -------------------------#
#-------------------------------------------------------#
#
# correlation matrix
crmtx = df.corr();
sns.heatmap(crmtx,annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
fig.savefig('correlation matrix.svg')

# housing loan - subscription correlation
crs_housing = pd.crosstab(df['subscribed'], df['housing']).apply(lambda x: x/x.sum() * 100)


#                                                       #
#------------ TRAIN TEST SPLIT -------------------------#
#-------------------------------------------------------#
#
# get train and test dataset stratified
stratifyfeature = "subscribed"      # feature used for stratification
splitters = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=425)
for train_set, test_set in splitters.split(df, df[stratifyfeature]):
    train = df.loc[train_set]
    test = df.loc[test_set]

y_train = train["subscribed"]
y_test = test["subscribed"]

x_train = train.drop("subscribed", axis=1).apply(LabelEncoder().fit_transform)
x_test = test.drop("subscribed", axis=1).apply(LabelEncoder().fit_transform)

# print(y_train.ftypes)

#                                                       #
#------------ CLASSIFICATION ---------------------------#
#-------------------------------------------------------#
#
# logistic regression - STANDARD
LRclassifier = LogisticRegression()
LRclassifier.fit(x_train,y_train)
LRtrain_score = LRclassifier.score(x_train, y_train)
LRtest_score = LRclassifier.score(x_test, y_test)

# Logistic Regression - CROSS VALIDATION
LRscores = cross_val_score(LRclassifier, x_train, y_train, cv=3)
LRmean = LRscores.mean()

print("train score: " + str(LRtrain_score))
print("test score: " + str(LRtest_score))
print("crossvalidation score: " + str(LRmean))

# dataset balance yes-no
# print(df['subscribed'].value_counts())
