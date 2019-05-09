
# coding: utf-8

# # Compas Analysis
# 
# Analaysis of the COMPAS Recidivism Risk Scores.
# 
# ### Submitted By : Jayalakshmi Vaidyanathan | Prachi Sharma | Sarah Fernandes | Vikita Nayak

# ## 1. Introduction

# COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) is a popular commercial algorithm used by judges and parole officers for scoring criminal defendant’s likelihood of reoffending (recidivism).It has been used to assess more than 1 million offenders since it was developed. This software predicts a defendant’s risk of committing a misdemeanor or felony within 2 years of assessment from 137 features about an individual and the individual’s past criminal record.
# 

# #### Data Collection Method

# When most defendants are booked in jail, they respond to a COMPAS questionnaire. Their answers are fed into the COMPAS software to generate several scores including predictions of “Risk of Recidivism” and “Risk of Violent Recidivism.”
# Through a public records request, ProPublica obtained two years worth of COMPAS scores from the Broward County Sheriff’s Office in Florida
# 7000 individuals arrested in Broward County, Florida between 2013 and 2014.
# 
# 
# Dataset Link: https://www.kaggle.com/danofer/compass#compas-scores-raw.csv
# 

# ### Load Packages

# In[710]:

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn import svm
from operator import itemgetter
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import sklearn as sk

from pandas.tools.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

get_ipython().magic('matplotlib inline')
plt.style.use('ggplot') # emulate R's pretty plotting

# print all the outputs in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings('ignore')
from datetime import date
# print numpy arrays with precision 4
np.set_printoptions(precision=4)


# ### Loading the Dataset
# 
# At first we select all the fields

# In[711]:

df = pd.read_csv('compas-scores-two-years.csv')
df.head()


# ## 2. Data Cleaning
# 
# ### Extracting Useful Fields
# 
# Not all of the rows are useable for the first round of analysis. There are a number of columns that have more than 50% missing data, hence we take them out from analysis. 

# In[712]:

def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(df)


# In[713]:

def missing_data(data):   
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False) 
    record_missing = pd.DataFrame(percent[percent > 50]).reset_index().rename(columns ={'index': 'feature',0: 'missing_fraction'})
    to_drop = list(record_missing['feature'])
    print(to_drop)
    data = data.drop(to_drop, axis = 1)
    return data

new_df = missing_data(df)
new_df.columns


# Removing insignificant columns like Id, Name and Date related fields 

# In[714]:

df = new_df[['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count',
             'priors_count','days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',  'c_days_from_compas', 
             'c_charge_degree', 'c_charge_desc', 'is_recid', 'is_violent_recid', 'type_of_assessment', 'decile_score.1', 
             'score_text', 'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'in_custody', 'out_custody', 
             'priors_count.1', 'start', 'end', 'event', 'two_year_recid']]
df.head()


# ## 3. Data Exploratory Analysis

# ### 3.1 Distribution of the COMPAS decile scores among whites and blacks.

# In[715]:

df_rico =  df[df['race']== 'African-American']


# In[716]:

df_rican =  df[df['race'] != 'African-American']


# In[717]:

#fig, ax = pyplot.subplots(figsize=a4_dims)
fig, ax =plt.subplots(1,2, figsize=(15,5))
sns.countplot(x='decile_score', data=df_rico, ax=ax[0])
sns.countplot(x='decile_score', data=df_rican, ax=ax[1])
ax[0].set_title('Black Defendant Decile Score')
ax[1].set_title('White Defendant Decile Score')
fig.show()


# ### Analysis : These histograms show that scores for white defendants were skewed toward lower-risk categories, while black defendants were evenly distributed across scores.

# In[718]:

fig, ax =plt.subplots(1,2, figsize=(15,5))
sns.countplot(x='v_decile_score', data=df_rico, ax=ax[0])
sns.countplot(x='v_decile_score', data=df_rican, ax=ax[1])
ax[0].set_title('Black Defendant Violent Decile Score')
ax[1].set_title('White Defendant Violent Decile Score')
fig.show()


# ### Analysis The histograms for COMPAS’s violent risk score also show a disparity in score distribution between white and black defendants

# We can drop 'type_of_assessment' and 'v_type_of_assessment' considering they have only one unique value. This doesn't add much value to our analysis.

# ## 4. Featurization of Fields

# Binarizing the 'sex' column by assigning 0 if it is Male and 1 if it is Female

# In[719]:

df['sex'] = df['sex'].apply(lambda x: 0 if x == 'Male' else 1)


# Converting the race categories in to numbers ranging from 0 to 5. For example, assigning 0 to African-American race category

# In[720]:

df.race = df.race.apply(lambda x:0 if x == 'African-American'                        else 1 if x == 'Caucasian'                       else 2 if x == 'Hispanic'                       else 3 if x == 'Other'                       else 4 if x == 'Asian'                       else 5)
                       


# Binarizing the 'c_charge_degree' column which can contains two types of values: F and M. We are assigning 0 to F and 1 to M

# In[721]:

df['c_charge_degree'] = df['c_charge_degree'].apply(lambda x: 0 if x == 'F' else 1)


# The v_score_text column & score_text columns are numerized by assigning 0 to Low, 0.5 to Medium and 1 to High scores.

# In[722]:

df['v_score_text'] = df['v_score_text'].apply(lambda x: 0 if x == 'Low' 
                                                          else 0.5 if x == 'Medium'
                                                          else 1)


# In[723]:

df['score_text'] = df['score_text'].apply(lambda x: 0 if x == 'Low' 
                                                      else 0.5 if x == 'Medium'
                                                      else 1)


# #### Calculated Columns for storing Jail Duration and Custody Duration

# Converting the below listed columns in to datetime format

# In[724]:

df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
df['in_custody'] = pd.to_datetime(df['in_custody'])
df['out_custody'] = pd.to_datetime(df['out_custody'])


# Extracting time out of these columns to get jail duration and custody duration to get insights if duration has an impact on the prediction

# In[725]:

df['jail_dur'] = df['c_jail_out'].sub(df['c_jail_in'], axis=0).astype('timedelta64[D]')


# In[726]:

df['custody_dur'] = df['out_custody'].sub(df['in_custody'], axis=0).astype('timedelta64[D]')


# In[727]:

df = df[['age','sex', 'race', 'juv_fel_count', 'decile_score','juv_misd_count', 'juv_other_count', 
         'priors_count', 'days_b_screening_arrest', 'c_days_from_compas', 'c_charge_degree', 'is_recid',
         'is_violent_recid', 'score_text','v_decile_score', 'v_score_text', 'start', 'end', 'event',
         'two_year_recid', 'jail_dur', 'custody_dur']]


# Checking for null values

# In[728]:

df.isna().sum()


# Filling the null values with 0

# In[729]:

df.days_b_screening_arrest.fillna(0,inplace=True)


# In[730]:

df.c_days_from_compas.fillna(0,inplace=True)


# In[731]:

df.jail_dur.fillna(0,inplace=True)


# In[732]:

df.custody_dur.fillna(0,inplace=True)


# In[733]:

df['length_of_sentence'] = df.end - df.start


# In[734]:

df.drop(columns={'start','end'}, inplace=True)


# Finally, checking the dataframe on which different machine learning models can be applied

# In[735]:

df.columns


# Finding the co-relation between the features

# In[736]:

df.corr()


# In[737]:

scatter_matrix(df,alpha=0.5, figsize=(30,32));


# ### Splitting the dataset with all features

# In[738]:

df_allfeatures = df


# In[739]:

df.columns


# In[740]:

X = df_allfeatures.drop('two_year_recid',axis=1)


# In[741]:

y = df_allfeatures.two_year_recid


# In[742]:

Xall_train, Xall_test, yall_train, yall_test = train_test_split(X,y, test_size=0.2,random_state=42)


# ###  Dataset with 10 features

# In[743]:

Xten_train = Xall_train[['is_recid', 'length_of_sentence', 'event', 'is_violent_recid',
       'decile_score', 'score_text', 'priors_count', 'custody_dur',
       'v_decile_score','age']]
Xten_test = Xall_test[['is_recid', 'length_of_sentence', 'event', 'is_violent_recid',
       'decile_score', 'score_text', 'priors_count', 'custody_dur',
       'v_decile_score','age']]


# In[744]:

yten_train = yall_train
yten_test = yall_test


# ## 5. Applying Machine Learning Models  to predict if they will recommite the crime 

# ### 5.1.a Logistic Regression 

# In[838]:

from sklearn.linear_model import LogisticRegression


# In[839]:

#Fitting the model
LogReg = LogisticRegression()
LogReg.fit(Xall_train, yall_train)


# In[840]:

#Predicting y values
pred_label = LogReg.predict(Xall_test)


# In[841]:

from sklearn.metrics import confusion_matrix


# In[842]:

#Construct confusion matrix
confusion_m = confusion_matrix(yall_test, pred_label)
sns.set()
mat = confusion_matrix(yall_test, pred_label)
sns.heatmap(mat,square=True, annot=True,fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[843]:

# Accuracy Of Prediction
print("Accuracy:" + str(LogReg.score(Xall_test, yall_test)))


# In[851]:

y_test_probability_predictions = LogReg.predict_proba(Xall_test)[:, 1]


# In[852]:

# Cross validation score of Prediction
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score1 = roc_auc_score(yall_test,pred_label)
print("ROC AUC: {0}".format(roc_auc_score1))
false_positive_rate, true_positive_rate, threshold = roc_curve(yall_test, y_test_probability_predictions)


# In[855]:

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.title('Receiver Operating Characteristic')
plt.ylabel('True Positive Rate', labelpad=13)
plt.xlabel('False Positive Rate', labelpad=13);


# ### Logistics regression on best 10 features

# In[752]:

#Fitting the model
LogReg = LogisticRegression()
LogReg.fit(Xten_train, yten_train)


# In[753]:

#Predicting y values
pred_label = LogReg.predict(Xten_test)


# In[754]:

from sklearn.metrics import confusion_matrix


# In[755]:

#Contruct confusion matrix
confusion_m = confusion_matrix(yten_test, pred_label)
sns.set()
mat = confusion_matrix(yten_test, pred_label)
sns.heatmap(mat,square=True, annot=True,fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[756]:

# Accuracy Of Prediction
print("Accuracy:" + str(LogReg.score(Xten_test, yten_test)))


# In[757]:

# Cross validation score of Prediction
from sklearn.metrics import roc_auc_score
roc_auc_score1 = roc_auc_score(yten_test,pred_label)
print("ROC AUC: {0}".format(roc_auc_score1))


# ### Naive Bayes 

# In[758]:

from sklearn import naive_bayes as naive_b


# In[759]:

#Fitting the model
naive_b = GaussianNB()
naive_b.fit(Xall_train,yall_train)


# In[760]:

#Predicting y values
pred_label = naive_b.predict(Xall_test)


# In[761]:

from sklearn.metrics import confusion_matrix


# In[762]:

#Construct confusion matrix
confusion_m = confusion_matrix(yall_test, pred_label)
sns.set()
mat = confusion_matrix(yall_test, pred_label)
sns.heatmap(mat,square=True, annot=True,fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[763]:

# Accuracy Of Prediction
print("Accuracy:" + str(naive_b.score(Xall_test, yall_test)))


# In[764]:

# Cross validation score of Prediction
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
roc_auc_score1 = roc_auc_score(yall_test,pred_label)
print("ROC AUC: {0}".format(roc_auc_score1))


# ### Naive bayes on best 10 features

# In[765]:

#Fitting the model
naive_b = GaussianNB()
naive_b.fit(Xten_train,yten_train)


# In[766]:

#Predicting y values
pred_label = naive_b.predict(Xten_test)


# In[767]:

from sklearn.metrics import confusion_matrix


# In[768]:

#Construct confusion matrix
confusion_m = confusion_matrix(yten_test, pred_label)
sns.set()
mat = confusion_matrix(yten_test, pred_label)
sns.heatmap(mat,square=True, annot=True,fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[769]:

# Accuracy Of Prediction
print("Accuracy:" + str(naive_b.score(Xten_test, yten_test)))


# In[770]:

# Cross validation score of Prediction
from sklearn.metrics import roc_auc_score
roc_auc_score1 = roc_auc_score(yten_test,pred_label)
print("ROC AUC: {0}".format(roc_auc_score1))


# ### SVM Classifier Linear

# In[771]:

#Fitting the model
model_svm = svm.SVC(kernel='linear')
model_svm.fit(Xall_train, yall_train)


# In[772]:

# predicting y values
y_pred = model_svm.predict(Xall_test)


# In[773]:

#Construct confusion matrix
confusion_m = confusion_matrix(yten_test, pred_label)
sns.set()
mat = confusion_matrix(yten_test, pred_label)
sns.heatmap(mat,square=True, annot=True,fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[774]:

# Accuracy Of Prediction
print("Accuracy:" + str(accuracy_score(yall_test,y_pred)))


# In[775]:

# Cross validation score of Prediction
from sklearn.metrics import roc_auc_score
roc_auc_score = roc_auc_score(yall_test, y_pred)
print("ROC AUC: {0}".format(roc_auc_score))


# ### SVM RBF

# In[776]:

#Fitting the model
model_svm = svm.SVC(kernel='rbf')
model_svm.fit(Xall_train, yall_train)


# In[777]:

# predicting y values
y_pred = model_svm.predict(Xall_test)


# In[778]:

#Construct confusion matrix
confusion_m = confusion_matrix(yten_test, pred_label)
sns.set()
mat = confusion_matrix(yten_test, pred_label)
sns.heatmap(mat,square=True, annot=True,fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[779]:

# Accuracy Of Prediction
print("Accuracy:" + str(accuracy_score(yall_test,y_pred)))


# In[780]:

# Cross validation score of Prediction
from sklearn.metrics import roc_auc_score
roc_auc_score = roc_auc_score(yall_test, y_pred)
print("ROC AUC: {0}".format(roc_auc_score))


# #### Analysis
# 
# #### The second model we selected was a Support Vector Machine with radial basis function kerneI because of its generally robust performance on nonlinear classification tasks

# ### SVC Classfier on best 10 Features 

# In[781]:

#Fitting the model
model_svm = svm.SVC(kernel='linear')
model_svm.fit(Xten_train, yten_train)


# In[782]:

# predict out of sample
y_pred = model_svm.predict(Xten_test)


# In[783]:

#Construct confusion matrix
confusion_m = confusion_matrix(yten_test, pred_label)
sns.set()
mat = confusion_matrix(yten_test, pred_label)
sns.heatmap(mat,square=True, annot=True,fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[784]:

# Accuracy Of Prediction
print("Accuracy:" + str(accuracy_score(yten_test,y_pred)))


# In[785]:

# Cross validation score of Prediction
from sklearn.metrics import roc_auc_score
roc_auc_score = roc_auc_score(yten_test, y_pred)
print("ROC AUC: {0}".format(roc_auc_score))


# ### Random Forest Classifier 

# In[786]:

#Fitting the model
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10, random_state=42)
rf.fit(Xall_train, yall_train)


# In[787]:

rf.score(Xall_test, yall_test)


# In[788]:

feature_viewer = {}
for col, score in zip(X, rf.feature_importances_):
    feature_viewer[col] = score
pd.Series(feature_viewer)



# In[789]:

sum(rf.feature_importances_)


# In[790]:

sns.set()
mat = confusion_matrix(yall_test, rf.predict(Xall_test))


# In[791]:

sns.heatmap(mat,square=True, annot=True,fmt='d', cbar=False)
plt.xlabel('True label- Crime recommited in two years')
plt.ylabel('Predicted label - Crime recommited in two years');


# In[792]:

# Accuracy Of Prediction
print("Accuracy:" + str(accuracy_score(yall_test,y_pred)))


# In[793]:

# Cross validation score of Prediction
from sklearn.metrics import roc_auc_score
roc_auc_score = roc_auc_score(yall_test,rf.predict(Xall_test))
print("ROC AUC: {0}".format(roc_auc_score))


# ### Feature viewer for Random forest

# In[794]:

#Create dataframe for feature importance
out=pd.DataFrame.from_dict(feature_viewer, orient='index')

out.columns=(['feature_score'])
top_f=out.sort_values("feature_score",ascending=False)
top_f


# In[795]:

#Graph with feature importance score
ax = top_f.plot(kind='bar', figsize=(15, 10), legend=True, fontsize=12, color= 'purple')
Features = [top_f.index]
v2 = [top_f.columns]
ax.set_xlabel('Features', fontsize=12)
ax.set_ylabel('feature_score', fontsize=12)
plt.show()


# #### Feature Analysis :  the most influential features are is_recid, length_of_sentence, is_violent_recid,decile_score and event. The least influential were juvenile_felony_count (possibly) because it contained so few non-zero values) and charge degree, which is interesting because one might expect that a more serious charge degree to indicate a higher chance of recidivism. This indicates that criminal record is generally weighed more heavily than demographic information.

# In[796]:

#Creating a new df to re apply random forest classifier with top 10 imp features
top_10=out.sort_values("feature_score",ascending=False).head(10)
top_10


# In[797]:

#Graph with top 10 best features
ax = top_10.plot(kind='bar', figsize=(15, 10), legend=True, fontsize=12, color= 'purple')
Features = [top_10.index]
v2 = [top_10.columns]
ax.set_xlabel('Features', fontsize=12)
ax.set_ylabel('feature_score', fontsize=12)
plt.show()


# In[798]:

df_topten = df[['is_recid','length_of_sentence','event','is_violent_recid','decile_score','score_text','priors_count','custody_dur','v_decile_score','age','two_year_recid']]


# In[799]:

df_topten.head()


# ### Random Forest Classifier best 10 features

# In[800]:

#Fitting the model
rf1 = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10, random_state = 42)
rf1.fit(Xten_train, yten_train)


# In[801]:

sns.set()
mat1 = confusion_matrix(yten_test, rf1.predict(Xten_test))


# In[802]:

#Construct confusion matrix
sns.heatmap(mat1,square=True, annot=True,fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[803]:

# Accuracy Of Prediction
rf1.score(Xten_test, yten_test)


# In[804]:

# Cross validation score of Prediction
from sklearn.metrics import roc_auc_score
roc_auc_score1 = roc_auc_score(yten_test,rf1.predict(Xten_test))
print("ROC AUC: {0}".format(roc_auc_score1))


# ## KNN Classifier

# In[805]:

from sklearn import metrics
#Fitting the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xall_train, yall_train)


# In[806]:

#Predicting y values
pred_label = knn.predict(Xall_test)


# In[807]:

#Construct confusion matrix
confusion_m = confusion_matrix(yall_test, pred_label)
sns.set()
mat = confusion_matrix(yall_test, pred_label)
sns.heatmap(mat,square=True, annot=True,fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[808]:

# Accuracy Of Prediction
pred_label = knn.predict(Xall_test)
print("Accuracy:" + str(knn.score(Xall_test, yall_test)))


# In[809]:

# Cross validation score of Prediction
from sklearn.metrics import roc_auc_score
roc_auc_score2 = roc_auc_score(yall_test,pred_label)
print("ROC AUC: {0}".format(roc_auc_score2))


# ### KNN for best  10 features

# In[810]:

from sklearn import metrics
#Fitting the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xten_train, yten_train)


# In[811]:

pred_label = knn.predict(Xten_test)


# In[812]:

from sklearn.metrics import confusion_matrix


# In[813]:

#Construct confusion matrix
confusion_m = confusion_matrix(yten_test, pred_label)
sns.set()
mat = confusion_matrix(yten_test, pred_label)
sns.heatmap(mat,square=True, annot=True,fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[814]:

# Accuracy Of Prediction
pred_label = knn.predict(Xten_test)
print("Accuracy:" + str(knn.score(Xten_test, yten_test)))


# In[815]:

from sklearn.metrics import roc_auc_score
roc_auc_score1 = roc_auc_score(yten_test,pred_label)
print("ROC AUC: {0}".format(roc_auc_score1))


# ### XG BOOST

# In[816]:

#Fitting the model
model = XGBClassifier()
model.fit(Xall_train, yall_train)


# In[817]:

y_pred3 = model.predict(Xall_test)
predictions = [round(value) for value in y_pred3]


# In[818]:

sns.set()
mat3 = confusion_matrix(yall_test, y_pred3)
sns.heatmap(mat3,square=True, annot=True,fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[819]:

# Accuracy Of Prediction
accuracy = accuracy_score(yall_test, predictions)
print("Accuracy:" + str(accuracy))    


# In[820]:

# Cross validation score of Prediction
from sklearn.metrics import roc_auc_score
roc_auc_score3 = roc_auc_score(yall_test,y_pred3)
print("ROC AUC: {0}".format(roc_auc_score3))


# ## XGBoost for best 10 features

# In[821]:

#Fitting the model
model = XGBClassifier()
model.fit(Xten_train, yten_train)


# In[822]:

y_pred3 = model.predict(Xten_test)
predictions = [round(value) for value in y_pred3]


# In[823]:

sns.set()
mat3 = confusion_matrix(yten_test, y_pred3)
sns.heatmap(mat3,square=True, annot=True,fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[824]:

# Accuracy Of Prediction
accuracy = accuracy_score(yten_test, predictions)
print("Accuracy:" + str(accuracy)) 


# In[825]:

# Cross validation score of Prediction
from sklearn.metrics import roc_auc_score
roc_auc_score3 = roc_auc_score(yten_test,y_pred3)
print("ROC AUC: {0}".format(roc_auc_score3))


# ### Accuracy & ROC plots for all models

# In[826]:

import pandas as pd

df_result = {'Model': ['Logistic Regression','Naïve Bayes','SVM','Random Forest','Xgboost','KNN'],
             'AllFeature_Accuracy': [0.9882, 0.979, 0.988,0.986, 0.993, 0.87],
             'AllFeature_ROC_AUC': [0.9886, 0.981, 0.989,0.993, 0.9939, 0.88],
             'TopFeature_Accuracy': [0.9902, 0.977, 0.99,0.993, 0.993, 0.87],
             'TopFeature_ROC_AUC': [0.991, 0.98, 0.99,0.993, 0.9939, 0.88]}

df_result = pd.DataFrame(df_result)


# #### Normalization

# In[827]:

norm_AllFeature_Accuracy = df_result.AllFeature_Accuracy / df_result.AllFeature_Accuracy[0]


# In[828]:

norm_AllFeature_ROC_AUC = df_result.AllFeature_ROC_AUC / df_result.AllFeature_ROC_AUC[0]


# In[829]:

norm_TopFeature_Accuracy = df_result.TopFeature_Accuracy / df_result.TopFeature_Accuracy[0]


# In[830]:

norm_TopFeature_ROC_AUC = df_result.TopFeature_ROC_AUC / df_result.TopFeature_ROC_AUC[0]


# #### Accuracy plot  For all Models with top 10 and all features

# In[831]:

fig = plt.figure(figsize=(10,5))
x = df_result.Model
plt.xlabel('Models')
plt.ylabel('Feature Accuracy')
plt.plot(x, norm_AllFeature_Accuracy, '-b', label='All Features')
plt.plot(x, norm_TopFeature_Accuracy, '-r', label='Top Features')
plt.legend(loc='best')
plt.show(100)


# #### ROC-AUC plot for all models with top 10 and all features

# In[832]:

fig = plt.figure(figsize=(10,5))
x = df_result.Model
plt.xlabel('Models')
plt.ylabel('Feature ROC-AUC')
plt.plot(x, norm_AllFeature_ROC_AUC, '-b', label='All Features')
plt.plot(x, norm_TopFeature_ROC_AUC, '-r', label='Top Features')
plt.legend(loc='best')
plt.show(100)


# In[833]:

df_confusion_matrix = {'Model': ['Logistic Regression','Naïve Bayes','SVM','SVM with PCA','Random Forest','KNN','XG Boost'],
             'Total_Correct_Prediction_all_features': [1426,1414,1411,1425,1433,1264,1433],
             'Total_False_Prediction_all_features': [17,29,32,18,10,179,10],
             'Total_Correct_Prediction_top_features': [1429,1411,1411,1423,1433,1264,1433],
             'Total_False_Prediction_top_features': [14,32,32,20,10,179,10]}

df_confusion_matrix = pd.DataFrame(df_confusion_matrix)


# In[834]:

df_confusion_matrix.head()


# In[835]:

fig = plt.figure(figsize=(10,5))
x = df_confusion_matrix.Model
plt.xlabel('Models')
plt.ylabel('Total_Correct_Prediction_all_features')
plt.plot(x, df_confusion_matrix.Total_Correct_Prediction_all_features, '-b', label='All Features')
plt.plot(x, df_confusion_matrix.Total_Correct_Prediction_top_features, '-r', label='Top Features')
plt.legend(loc='best')
plt.show(100)


# In[836]:

sns.factorplot(x='Total_Correct_Prediction_all_features',y='Model',data=df_confusion_matrix, aspect =3)


# In[837]:

fig = plt.figure(figsize=(15,5))
X = df_confusion_matrix.Model
Y = df_confusion_matrix.Total_Correct_Prediction_all_features
Z = df_confusion_matrix.Total_Correct_Prediction_top_features
_X = np.arange(len(X)) 
plt.bar(_X - 0.2, Y, 0.4, label='All Features') 
plt.bar(_X + 0.2, Z, 0.4, label='Top Features') 
plt.legend(loc='best')
plt.xticks(_X, X) # set labels manually plt.show()


# In[ ]:



