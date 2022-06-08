#!/usr/bin/env python
# coding: utf-8

# THE SPARKS FOUNDATION
# 
# DATA SCIENCE AND BUSINESS ANALYTICS INTERNSHIP
# 
# NAME:UJJWAL WADERA
# 

# TASK-1 PREDICTING THE PERCENTAGE OF STUDENTS BASED ON THE STUDY HOURS USING SUPERVISED MACHINE LEARNING

# In[1]:


#importing important libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# reading the csv file which i made and stored the given data into it.

df=pd.read_excel('supervised_ML.xlsx')


# In[3]:


#shows the top 5 readings

df.head()


# In[4]:


#lets plot the graph and see its nature

get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours VS Scores')
plt.scatter(df.Hours,df.Scores,color='red',marker='+')


# In[5]:


#iloc basically tells the location based on the indexing and here we are 

X=df.iloc[:, :-1].values 
y=df.iloc[:, 1].values


# In[6]:


#Now we will train our model and prepare it for prediction,we will use sklearn's train_test split method

from sklearn.model_selection import train_test_split


# In[25]:


#we obtain these particular values and test size indicates what portion of the data set we are taking for testing after testing

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=50)


# In[26]:


len(X_train)


# In[27]:


len(X_test)


# In[28]:


from sklearn.linear_model import LinearRegression


# In[29]:


clf=LinearRegression()


# In[30]:


#this command makes our model ready for prediction

clf.fit(X_train, y_train)


# In[38]:


#This will show us the best fit line

slope=clf.coef_
inter=clf.intercept_
line=slope*X+inter
plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# In[31]:


# now lets test the model 

y_pred=clf.predict(X_test)


# In[32]:


y_test


# In[33]:


# tells about the percentage precision of prediction

clf.score(X_test,y_test)


# In[34]:


#created a dataframe and stored the actual and the predicted values side by side

Df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})


# In[35]:


Df


# In[36]:


Hour=[[9.25]]

clf.predict(Hour)


# In[37]:


#This command will basically tell you the error between your actual vs predicted value
from sklearn import metrics
from sklearn.metrics import r2_score
metrics.mean_absolute_error(y_test,y_pred)

