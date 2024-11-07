import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score 


# opening and reading dataset
df = pd.read_csv('Housing.csv')

df.head()
df.shape
df.describe()
df.info()

# looking for duplicates
df.loc[df.duplicated()]

# original data visualization
'''
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('House Price Distribution Plot')
sns.displot(df.price)

plt.subplot(1,2,2)
sns.boxplot(df.price)
plt.title('House Pricing Spread')

plt.show()
'''

# visualizing categorical data
categorical_list = [x for x in df.columns if df[x].dtype == 'object']
for x in categorical_list:
    print(x)
    
# visualizing frequency of mainroad, guestroom, basement
plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt1 = df['mainroad'].value_counts().plot(kind = 'bar')
plt.title('mainroad Histogram')
plt1.set(xlabel = 'mainroad', ylabel = 'Frequency of mainroad')

plt.subplot(1,3,2)
plt1 = df['guestroom'].value_counts().plot(kind = 'bar')
plt.title('guestroom Histogram')
plt1.set(xlabel = 'guestroom', ylabel = 'Frequency of guestroom')

plt.subplot(1,3,3)
plt1 = df['basement'].value_counts().plot(kind = 'bar')
plt.title('basement Histogram')
plt1.set(xlabel = 'basement', ylabel = 'Frequency of basement')
 

# visualizing frequency of hotwaterheating, aircoinditioning, prefarea, and furnishing status
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt1 = df['hotwaterheating'].value_counts().plot(kind='bar',color='r')
plt.title('hotwaterheating Histogram')
plt1.set(xlabel = 'hotwaterheating', ylabel='Frequency of hotwaterheating')

plt.subplot(1, 3, 2)
plt1 = df['airconditioning'].value_counts().plot(kind='bar',color='r')
plt.title('airconditioning Histogram')
plt1.set(xlabel = 'airconditioning', ylabel='Frequency of airconditioning')

plt.subplot(1, 3, 3)
plt1 = df['prefarea'].value_counts().plot(kind='bar',color='r')
plt.title('prefarea Histogram')
plt1.set(xlabel = 'prefarea', ylabel='Frequency of prefarea')

plt.subplot(2, 2, 3)
plt1 = df['furnishingstatus'].value_counts().plot(kind='bar',color='r')
plt.title('furnishingstatus Histogram')
plt1.set(xlabel = 'furnishingstatus', ylabel='Frequency of furnishingstatus')

# visualizing mainroad vs price and guestroom vs price
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Mainroad vs Price')
sns.boxplot(x=df.mainroad, y=df.price, palette=("cubehelix"))

plt.subplot(1,2,2)
plt.title('guestroom vs Price')
sns.boxplot(x=df.guestroom, y=df.price, palette=("PuBuGn"))
'''
after looking at boxplot, can see that there is a correlation 
b/w being on a mainroad/having a guestroom and price
'''

# visualizing basement vs price and hotwaterheating vs price
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title('basement vs Price')
sns.boxplot(x=df.basement, y=df.price, palette=("cubehelix"))

plt.subplot(1,2,2)
plt.title('hotwaterheating vs Price')
sns.boxplot(x=df.hotwaterheating, y=df.price, palette=("PuBuGn"))
'''
having basement and hot water has slight correlation
'''

# visualizing airconiditioning, prefarea, and furnishingstatus vs price
plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.title('airconditioning vs Price')
sns.boxplot(x=df.airconditioning, y=df.price, palette=("cubehelix"))

plt.subplot(1,3,2)
plt.title('prefarea vs Price')
sns.boxplot(x=df.prefarea, y=df.price, palette=("PuBuGn"))

plt.subplot(1,3,3)
plt.title('furnishingstatus vs Price')
sns.boxplot(x=df.furnishingstatus, y=df.price, palette=("PuBuGn"))
plt.xticks(rotation=45)
'''
having airconditioning and prefarea have positive corealtion 
with Price of the house. 

Also a furnished house would have higher price.

'''
# visualizing numerical data
numerical_list = [x for x in df.columns if df[x].dtype in ('int64', 'float64')]
print(numerical_list)

def scatter(x,fig):
    plt.subplot(5,2,fig)
    plt.scatter(df[x],df['price'])
    plt.title(x+' vs Price')
    plt.ylabel('Price')
    plt.xlabel(x)

plt.figure(figsize=(10,20))

scatter('area', 1)
scatter('bedrooms', 2)
scatter('stories', 3)
scatter('stories', 4)
scatter('parking', 5)
plt.tight_layout
'''
can see that area has a strong positive correlation with price
'''
sns.pairplot(df)

# making a correlation matrix
cor_matrix = df[numerical_list].corr()
plt.figure(figsize=(12,8))
sns.heatmap(cor_matrix, annot= True, cmap='coolwarm', linewidths = 0.5)
plt.title('Correlation Heatmap')

# making dummy variables
def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first= True).astype(int)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df

# applying dummies function to the df
df = dummies('mainroad', df)
df = dummies('guestroom', df)
df = dummies('hotwaterheating', df)
df = dummies('basement', df)
df = dummies('airconditioning', df)
df = dummies('prefarea', df)
df = dummies('furnishingstatus', df)

df.tail()
df.shape


np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.75, test_size = 0.25, random_state = 100)

scaler = MinMaxScaler()
df_train[numerical_list] = scaler.fit_transform(df_train[numerical_list])

df_train.head()

# dividing data into X and y varaibles
y_train = df_train.pop('price')
X_train = df_train

rfe = RFE(estimator=LinearRegression(), n_features_to_select = 10)
rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns, rfe.support_, rfe.ranking_))

print(X_train.columns[rfe.support_])

X_train_rfe = X_train[X_train.columns[rfe.support_]]
X_train_rfe.head()

def build_model(X,y):
    X = sm.add_constant(X) # constant for intercept
    lm = sm.OLS(y,X).fit() # fitting the model
    print(lm.summary()) # model summary
    return X

# function to check variance inflation factor for each independent varaible in the dataset
def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['vif'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = 'VIF', ascending = False)
    return(vif)

# First Model
X_train_new = build_model(X_train_rfe, y_train)

# removing bedrooms since it has high p value (over 0.05)
X_train_new = X_train_new.drop(['bedrooms'], axis = 1)
X_train_new = build_model(X_train_new, y_train)

checkVIF(X_train_new)
'''
dropping yes dummies since they have very high VIF scores
'''
# second model
X_train_new = X_train_new.drop(['yes'], axis = 1)
X_train_new = build_model(X_train_new, y_train)

checkVIF(X_train_new)

# verifying model via residual analysis
lm = sm.OLS(y_train, X_train_new).fit()
y_train_price = lm.predict(X_train_new)

# plotting histogram of the error terms
fig = plt.figure()
sns.displot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                 
plt.xlabel('Errors', fontsize = 18)

## predicting on the test set

# test set numerical values scaled
df_test[numerical_list] = scaler.fit_transform(df_test[numerical_list])

# dividing into X and y
y_test = df_test.pop('price')
X_test = df_test

# selecting the choosen features from the train set
X_train_new = X_train_new.drop('const', axis = 1)

# creating X_test_new dataframe by droppping variables from X_test
X_test_new = X_test[X_train_new.columns]

# adding a constant
X_test_new = sm.add_constant(X_test_new)

# making predictions
y_pred = lm.predict(X_test_new)
r2_score(y_test, y_pred)

# evaluation of the model
fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 20)
plt.xlabel('y_test', fontsize = 18)
plt.ylabel('y_pred', fontsize = 16)
plt.show()

print(lm.summary())