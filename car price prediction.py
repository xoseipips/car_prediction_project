#!/usr/bin/env python
# coding: utf-8

#    ### A MULTIPLE LINEAR REGRESSION PROJECT ON A PRICE  DATASET 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# lets import necessary libraries

import pandas as pd  #for linear algebra
import numpy as np   #for dataprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import metrics
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, cross_val_score


# In[2]:


#os.getcwd()
#os.listdir('C:\\Users\\DELL\\Desktop\\september projects\\CarPrice_Assignment1.csv')


# In[3]:


dataset = pd.read_csv('C:\\Users/DELL/Desktop/september projects/CarPrice_Assignment1.csv')
dataset
dataset.head(6)  # lets show the first 6 rows of the dataset.


# In[4]:


dataset.info()


# ### dataset having null or not 

# In[5]:


dataset.isna().sum()


# # Data preprocessing
# ### checking for duplicates

# In[6]:


dataset.nunique()


# In[7]:


dataset.drop_duplicates()
dataset


# In[8]:


# check for skewness and correlations
dataSkew = dataset.corr() 
dataSkew
#sns.heatmap(data = dataset.corr())
#plt.show()


# In[9]:


dataset["fueltype"].value_counts()


# In[10]:


dataset['fueltype'].value_counts()


# ### checking for outliers

# In[16]:


fig, axs = plt.subplots(3, figsize = (5,5) )
plt1 =sns.boxplot(dataset["price"], ax = axs[0])
plt1 =sns.boxplot(dataset["enginesize"], ax = axs[1])
plt1 =sns.boxplot(dataset["curbweight"], ax = axs[2])
#plt1 =sns.boxplot(dataset["fueltype"], ax = axs[4])

plt.tight_layout()


# # exploratory data analytics
#  ### distribution of the target variable

# In[12]:


sns.distplot(dataset['price'])


# ### conclusion is its not normally distributed. its right skewed

# # relationship betweeen target variable and other variables

# In[13]:


sns.pairplot(dataset, x_vars=["enginesize", "horsepower", "curbweight"], 
            y_vars = "price", height = 4, aspect = 1, kind = "scatter")

plt.show()


# # MODELLING THE DATASET
# ###  splitting the dataset into train and test set

# In[14]:


# defining independent variables(features), x and dependent variable (target), y

x = dataset[['enginesize','horsepower','curbweight' ]]
y = dataset['price']

#split the dataset into train and test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 4)

#creating a linear regression model

model1 = LinearRegression()

# train the model using the train data

model1.fit(x_train, y_train)

#make predictions on the test data

predictions = model1.predict(x_test)


#actual value and the predicted value

model_df = pd.DataFrame({'Actual value' : y_test, 
                        "predicted value" : predictions})

print(model_df)

#calcculate and measure the model performance metrics
#this measures the average squared differences between the actual and predicted values
mse = mean_squared_error(y_test, predictions)
#the measure the proportion of the variance on the target variable thats predictable from the independent variable.
r2 = r2_score(y_test, predictions)


#printing the model coefficients
print(list(zip(x, model1.coef_)))
# printing the intercept
print('intercept:', model1.intercept_)
print("mean squared error: ", mse)
print("R-squared: ", r2)


#  the MSE value indicates that the average squared differences between the predicted and the actual is 133224006.65707883
#  the R-squared represents the proportion of the variance in the target variable thats predictable from the independent variable
#  thus a rsquared value of 78.95 indicates that 78.9% of the variance in  the target variable is explained by the independent variables in the model.
#  
#  the intercept says what the target value would become if all the independent variables are reduced to 0.

# In[ ]:





# In[ ]:




