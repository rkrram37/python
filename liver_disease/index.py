import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv('cirrhosis.csv', index_col='ID')
df.head()


df.info()
df.describe()

df.isna().sum()
df.select_dtypes(include=(['int64', 'float64'])).isna().sum()


df_num_col = df.select_dtypes(include=(['int64', 'float64'])).columns
for c in df_num_col:
    df[c].fillna(df[c].median(), inplace=True)

df.select_dtypes(include=(['int64', 'float64'])).isna().sum()

df.select_dtypes(include=('object')).isna().sum()

df_cat_col = df.select_dtypes(include=('object')).columns
for c in df_cat_col:
    df[c].fillna(df[c].mode().values[0], inplace=True)
    
df.select_dtypes(include=('object')).isna().sum()

df['Stage'].value_counts()

#df['Stage'] = np.where(df['Stage'] == 4,1,0)



plt.figure(figsize=(21.2,10))

plt.subplot(1,2,2)
sns.countplot(x=df['Stage'], hue=df['Sex'], palette='Blues', alpha=0.9)
sns.despine(top=True, right=True, bottom=True, left=True)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False)
plt.xlabel('')
plt.title('Disease Stage Across Gender')

plt.subplot(1,2,2)
sns.countplot(x=df['Stage'], hue=df['Ascites'], palette='Purples', alpha=0.9)
sns.despine(top=True, right=True, bottom=True, left=True)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False)
plt.xlabel('')
plt.title('Ascites proportion across Stages')

plt.subplot(1,2,2)
sns.countplot(x=df['Stage'], hue=df['Drug'], palette='Blues', alpha=0.9)
sns.despine(top=True, right=True, bottom=True, left=True)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False)
plt.xlabel('')
plt.title('Medications prescribed across Stages');


plt.subplot(1,2,2)
sns.countplot(x=df['Stage'], hue=df['Hepatomegaly'], palette='Purples', alpha=0.9)
sns.despine(top=True, right=True, bottom=True, left=True)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False)
plt.xlabel('')
plt.title('Hepatomegaly');


plt.subplot(1,2,2)
sns.countplot(x=df['Stage'], hue=df['Spiders'], palette='Blues', alpha=0.9)
sns.despine(top=True, right=True, bottom=True, left=True)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False)
plt.xlabel('')
plt.title('Presence of Spiders across stages');


plt.subplot(1,2,2)
sns.countplot(x=df['Stage'], hue=df['Edema'], palette='Purples', alpha=0.9)
sns.despine(top=True, right=True, bottom=True, left=True)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False)
plt.xlabel('')
plt.title('Edema');


    # replacing catagorical data with intigers.
    df['Sex'] = df['Sex'].replace({'M':0, 'F':1})                                # Male : 0 , Female :1
    df['Ascites'] = df['Ascites'].replace({'N':0, 'Y':1})                        # N : 0, Y : 1   
    df['Drug'] = df['Drug'].replace({'D-penicillamine':0, 'Placebo':1})          # D-penicillamine : 0, Placebo : 1   
    df['Hepatomegaly'] = df['Hepatomegaly'].replace({'N':0, 'Y':1})              # N : 0, Y : 1
    df['Spiders'] = df['Spiders'].replace({'N':0, 'Y':1})                        # N : 0, Y : 1
    df['Edema'] = df['Edema'].replace({'N':0, 'Y':1, 'S':-1})                    # N : 0, Y : 1, S : -1
    df['Status'] = df['Status'].replace({'C':0, 'CL':1, 'D':-1})                 # 'C':0, 'CL':1, 'D':-1

cdf = df[['Status','Drug','Age','Sex','Ascites','Hepatomegaly','Spiders','Edema','Bilirubin','Cholesterol','Albumin','Copper','Alk_Phos','SGOT','Tryglicerides','Platelets','Prothrombin','Stage']]

x = cdf.iloc[:, :17]
y = cdf.iloc[:, -1]

from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



linearRegression = LinearRegression()
linearRegression.fit(x, y)
y_pred = linearRegression.predict([[-1,0,21464,1,1,1,1,1,14.5,261.0,2.6,156.0,1718.0,137.95,172.0,190.0,12.2]])

SVM = SVR()
SVM.fit(x, y)
y_pred = SVM.predict([[0,0,20617,1,0,1,1,0,1.1,302.0,4.14,54.0,7394.8,113.52,88.0,221.0,10.6]])
print(y_pred)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=44)
model.fit(x, y)
predictions = model.predict([[0,1,14060,1,0,0,0,0,0.7,132.0,3.6,17.0,423.0,49.6,56.0,265.0,11.0]])

round(linearRegression.score(x,y), 4)
round(model.score(x,y), 4)
round(SVM.score(x,y), 4)


file=open('my_model.pkl','wb')
pickle.dump(SVM,file,protocol=2)