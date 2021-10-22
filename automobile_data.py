#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pickle
import pandas as pd
# In[1]:


df = pd.read_csv('imports-85.csv')


# In[2]:


df.columns = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "doors", "body", "wheels","engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine", "cylindres", "engine-size", "fuel-system", "bore", "stroke", "compression", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]


# In[128]:


df_to_reg = df[['wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']]


# In[129]:


indexes = []
for index, row in df_to_reg.iterrows():
    if row['price'] == '?':
        indexes.append(index)
df_to_reg = df_to_reg.drop(indexes)


# In[130]:


indexes = []
for index, row in df_to_reg.iterrows():
    if row['peak-rpm'] == '?':
        indexes.append(index)
df_to_reg = df_to_reg.drop(indexes)


# In[131]:


indexes = []
for index, row in df_to_reg.iterrows():
    if row['horsepower'] == '?':
        indexes.append(index)
df_to_reg = df_to_reg.drop(indexes)


# In[132]:


indexes = []
for index, row in df_to_reg.iterrows():
    if row['stroke'] == '?' or row['bore'] == '?':
        indexes.append(index)
df_to_reg = df_to_reg.drop(indexes)


# In[133]:


for c in df_to_reg.columns:
    df_to_reg[c] = pd.to_numeric(df_to_reg[c], downcast='float')


# In[112]:


"""scales = []
for c in df_to_reg.columns:
    
    df_to_reg[c] = pd.to_numeric(df_to_reg[c], downcast='float')
    l = df_to_reg[c].values
    top = l.max()
    scales.append(1/top)
    for i in range(len(l)):
        l[i] = l[i]/top
    """


# In[134]:


X = df_to_reg.iloc[:,:-1].values


# In[135]:


y = df_to_reg.iloc[:,-1:].values


# In[136]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=0)


# In[137]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[138]:


coefs = regressor.coef_[0]


# In[139]:


import matplotlib.pyplot as plt
x = np.arange(len(regressor.coef_[0]))
#plt.bar(x, coefs)
#plt.show()


# In[140]:


y_pred = regressor.predict(X_test)


# In[165]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[145]:


sample = [1.012e+02, 1.768e+02, 6.480e+01, 5.430e+01, 2.710e+03, 1.640e+02,
       3.310e+00, 3.190e+00, 9.000e+00, 1.210e+02, 4.250e+03, 2.100e+01,
       2.800e+01]


# In[155]:


sample = np.array(sample)
sample = sample.reshape(1, -1)
res = regressor.predict(sample)


# In[166]:





# In[168]:


model = pickle.dump(regressor, open('model.pkl', 'wb'))


# In[ ]:




