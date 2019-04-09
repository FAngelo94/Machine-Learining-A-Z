# Data Preprocessing Template

### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Importing dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values # get all columns except the last
Y = dataset.iloc[:,-1].values # get the depended variables

M = max(X[:,1])
m = min(X[:,1])


### We want remove nan value replacing them with  other values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') # replace nan data with mean of column
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


### Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
'''
 The following code is equal to use ColumnTransformer below, LabelEncoder and OneHotEncoder
 will be deprecated in the next releases:
 
 labelencoder_X = LabelEncoder()
 X[:, 0] = labelencoder_X.fit_transform(X[:,0])
 honehotencoder = OneHotEncoder(categorical_features = [0])
 X = honehotencoder.fit_transform(X).toarray()
'''
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

X = np.array(ct.fit_transform(X), dtype=np.float)

labelencoder_Y = LabelEncoder()     # For depend variables just need label encoder because machine learning
Y = labelencoder_Y.fit_transform(Y) # know that Y is a categorical data without order of importance


### Split data in train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)


### Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

