#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import pandas as pd


# In[3]:


df = pd.read_csv('air pollution.csv',na_values='=')
df


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.head(6)


# In[7]:


df.columns


# In[8]:


data2=df.copy()


# In[9]:


data2=data2.fillna(data2.mean())


# In[10]:


data2.head()


# In[11]:


data2.info()


# In[14]:


#mapping
dist=(data2['City'])
distset=set(dist)
dd=list(distset)
dict0fWords= {dd[i] : i for i in range(0, len(dd))}
data2['City']=data2['City'].map(dict0fWords)


# In[15]:


dist=(data2['AQI_Bucket'])
distset=set(dist)
dd=list(distset)
dict0fWords= {dd[i] : i for i in range(0, len(dd))}
data2['AQI_Bucket']=data2['AQI_Bucket'].map(dict0fWords)


# In[16]:


data2['AQI_Bucket']=data2['AQI_Bucket'].fillna(data2['AQI_Bucket'].mean())


# In[17]:


data2


# In[18]:


data2.isnull().sum()


# In[21]:


print(data2)


# In[22]:


data2.columns


# In[32]:


features=data2[['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3','Benzene', 'Toluene', 'Xylene']]
labels=data2['AQI']


# In[33]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest=train_test_split(features,labels,test_size=0.2,random_state=2)


# In[34]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
regr=RandomForestRegressor(max_depth=2,random_state=0)
regr.fit(Xtrain,Ytrain)


# In[35]:


print(regr.predict(Xtest))


# In[36]:


y_pred=regr.predict(Xtest)


# In[37]:


from sklearn.metrics import r2_score
r2_score(Ytest,y_pred)


# In[ ]:





# In[ ]:




