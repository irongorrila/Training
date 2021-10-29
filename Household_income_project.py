###################################################################
''' House Hold Income Project '''
###################################################################
'''
1. Identify the output variable.                                                    (done)
2. Understand the type of data.                                                     (done)
3. Check if there are any biases in your dataset.                                   (done)
4. Check whether all members of the house have the same poverty level.              (done)
5. Check if there is a house without a family head.                                 (done)
6. Set poverty level of the members and the head of the house within a family.
7. Count how many null values are existing in columns.                              (done)
8. Remove null value rows of the target variable.                                   (done)
9. Predict the accuracy using random forest classifier.                             (done)
10.Check the accuracy using random forest with cross validation                     (done)

'''
####################################################################

import pandas as pd
import numpy as np

####################################################################

" Importing Data"

df_train= pd.read_csv('hh_train.csv')
df_test= pd.read_csv('hh_test.csv')

####################################################################
" Exploratory Analysis / Understanding the data "

df_train.shape    # 9557 rows /143 cols
df_test.shape     # 23856 rows / 142 cols

# Checking to see if there are any duplicate Ids in training & test dataset
(set(df_train['Id'])).intersection(set(df_test['Id']))

# returned a empty set which means no duplicate entries

# finding the column which is  missing from the test data ( target column )
train_features= set(df_train.columns)
test_features= set(df_test.columns)

# Finding the output variable
print('The output variable or the predicted variable is: {} '.format(train_features.difference(test_features)))

print(df_train.dtypes)
df_train.isnull().sum()
df_train.describe()
df_train.dependency.head()
df_train.edjefe.head(); 
df_train.edjefa.head()
df_train.Target.isnull().sum()
df_train= df_train.round(decimals=3)

# Checking for families with no adults who can be counted as HoF
hh_no_adult= df_train[df_train['hogar_adul'] == 0]
print('There are {} families where there is no HoF'. format(len(hh_no_adult)))

df_train.Target.value_counts(normalize= True)
# 63% of the training data belongs to category . This indicates a bias

# Chekcing if the poverty levels of all members are same within a family
level_checker= df_train.groupby('idhogar')['Target'].agg([max, min])
L= level_checker.query('max != min')
# 85 families where there is a mismatch
# Identify the family id where this is happening
L_arr= np.array(L.index)

# replacing poverty levels as per HoF
for i in L_arr:
    new_target= df_train[(df_train['idhogar']==i) & (df_train['parentesco1']==1)]['Target']
    df_train.loc[df_train['idhogar']==i, 'Target'] = new_target

#Checking 
level_checker= df_train.groupby('idhogar')['Target'].agg([max, min])
L= level_checker.query('max != min')


######################
""" observations """
######################

# Variable v2a1 would need nulll value handling. 6860 NA values
# text data in dependency, edjefe, edjefa columns
# There are no NULL values in Target column
# No Null values in Target column
# escolari column has the same info as edjefe / edjefa and we can keep only escolari
# multiple redundant columsn found : v18q, mobilephone, hogar_total etc would need handling in feature engg. phase
# 63% of the training data belongs to category . This indicates a bias

####################################################################
''' Feature Engineering '''

len(df_train.columns) # 143 columns

# replacing dependency column values with sqrt of SQBdependency
df_train.dependency = np.sqrt(df_train.SQBdependency).round(decimals=3)
df_train.dependency.head()

# removing all columns squared columns
col_sq= df_train.columns.str.startswith('SQB')
df_train_nosq= df_train.loc[:,~col_sq]

# removing other redundant / unwanted columns
# Gender based columns being ignored if total count is available
# rez_esc : most values are blank, any fill strategy would result in unwanted approximation
# r4 values dropped since count is more important than gender demographics here
# tamhog,  hogar  values are redundant
#elimbasu5 is only 0 values

drop_cols= ['v18q','agesq', 'mobilephone', 'edjefe', 'edjefa', 'hogar_nin',
            'hogar_adul', 'hogar_mayor', 'hogar_total','tamviv', 'tamhog', 'r4t3', 'r4h3',
            'r4h1','r4h2', 'r4m1', 'r4m2', 'r4m3', 'rez_esc', 'elimbasu5', 'male', 'female']

#saving local copy of changes to dataset
df_train_nosq.drop(drop_cols, axis=1).to_csv('Curated_hh.csv')

df_train_nosq.drop(drop_cols, axis=1).shape
# No. of features reduced to 117

df_trim1= df_train_nosq.drop(drop_cols, axis=1)
print('No. of people living in fully owned house:', len(df_trim1[(df_trim1.tipovivi1 == 1)]))
print('No. of people living in own house paying emi: ', len(df_trim1[df_trim1.tipovivi2 == 1]))
print('No. of people living in rented: ', len(df_trim1[df_trim1.tipovivi3 == 1]))
print('No. of people living in precarious: ', len(df_trim1[df_trim1.tipovivi4 == 1]))
print('No. of people living in other: ', len(df_trim1[df_trim1.tipovivi5 == 1]))

# Subtracting the ones paying emi(961) and paying rent(1736) from the total count of v2a1 column...
# we get the exact missing count 6860. Therfore replacing with 0 for null alues

df_trim1.v2a1.fillna(0, inplace= True)
df_trim1.isnull().sum()
df_trim1.v18q1.fillna(0, inplace=True)
df_trim1.v18q1.isnull().sum()

# negating bias by dropping random 62% of category 4 values

no_bias_df = df_trim1.drop(df_trim1.query('Target == 4').sample(frac= 0.62, random_state=5).index)

# checking the split of the Target column after bias mitigation
no_bias_df.Target.value_counts(normalize= True)
no_bias_df.shape

# Bootstrapping unbiased sample to make up for lost rows
no_bias_add = no_bias_df.sample(frac= 0.9, random_state= 5)
# Joining both data frames to form the training dataset
nb_df= pd.concat([no_bias_df, no_bias_add])
nb_df.Target.value_counts(normalize=True)

# finding non numerical columns
for c in nb_df.columns:
    if nb_df['{}'.format(c)].dtypes == 'float64':
        print(c)
    
nb_df.dropna(inplace=True)
#################################################################
'''Preprocessing the test dataset'''

df_test.dependency= np.sqrt(df_test.SQBdependency).round(decimals=3)
df_test.dependency.isnull().sum()

col_sq_test= df_test.columns.str.startswith('SQB')
df_test_nosq= df_test.loc[:,~col_sq_test]
df_test_new= df_test_nosq.drop(drop_cols, axis=1)

# Handling NA values
df_test_new.isnull().sum()
df_test_new.v2a1.fillna(0, inplace= True)
df_test_new.v18q1.fillna(0, inplace= True)
df_test_new.drop(['Id', 'idhogar'], axis= 1, inplace= True)

df_test_new.iloc[:,75:100].isnull().sum() # meaneduc has 3 null values
df_test_new.meaneduc.fillna(df_test_new.meaneduc.mean().round(decimals=3), inplace= True)

#################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

X= nb_df.drop(['Target', 'Id', 'idhogar'], axis=1)
y= nb_df.Target

# splitting the given training dataset
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=5, stratify= y)


rfc= RandomForestClassifier()

rfc.fit(X_train, y_train)
y_pred= rfc.predict(X_test)

'''Accuracy for given dataset'''
print(classification_report(y_test, y_pred))

'''Accuracy with Cross-Validation'''
cross_val_score(rfc, X_train, y_train, cv= 5, scoring='accuracy').mean()

############################################################
# trying model on unseen data
X_unseen= df_test_new
y_unseen_predict= rfc.predict(df_test_new)

print(cross_val_score(rfc, X_unseen, y_unseen_predict, cv=5, scoring= 'accuracy'))

