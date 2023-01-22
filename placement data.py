#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv(r"C:\Users\smara\OneDrive\Desktop\DATA SET\Job_Placement_Data.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.columns


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


le=LabelEncoder()


# In[11]:


data.head()


# In[14]:


data.gender=le.fit_transform(data.gender)
data.ssc_percentage=le.fit_transform(data.ssc_percentage)
data.ssc_board=le.fit_transform(data.ssc_board)
data.hsc_percentage=le.fit_transform(data.hsc_percentage)
data.hsc_board=le.fit_transform(data.hsc_board)
data.hsc_subject=le.fit_transform(data.hsc_subject)
data.degree_percentage=le.fit_transform(data.degree_percentage)
data.undergrad_degree=le.fit_transform(data.undergrad_degree)
data.work_experience=le.fit_transform(data.work_experience)
data.emp_test_percentage=le.fit_transform(data.emp_test_percentage)
data.specialisation=le.fit_transform(data.specialisation)
data.mba_percent=le.fit_transform(data.mba_percent)
data.status=le.fit_transform(data.status)


# In[15]:


data.head()


# In[16]:


data.columns


# In[17]:


y=data['status']
x=data[['gender', 'ssc_percentage', 'ssc_board', 'hsc_percentage', 'hsc_board',
       'hsc_subject', 'degree_percentage', 'undergrad_degree',
       'work_experience', 'emp_test_percentage', 'specialisation',
       'mba_percent']]


# In[18]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[22]:


x_train


# In[23]:


y_train


# In[24]:


x_test


# In[25]:


y_test


# In[26]:


model.fit(x_train,y_train)


# In[27]:


model.intercept_


# In[28]:


model.coef_


# In[32]:


model.score(x_test,y_test)


# In[33]:


from sklearn.linear_model import LogisticRegression
logic=LogisticRegression()


# In[38]:


logic.fit(x_train,y_train)


# In[39]:


logic.intercept_


# In[40]:


logic.coef_


# In[41]:


logic.score(x_test,y_test)


# In[ ]:




