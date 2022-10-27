#!/usr/bin/env python
# coding: utf-8

# In[1]:

## Initialize

# In[37]:


import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os

# Visualisation libraries
import matplotlib.pyplot as plt
# matplotlib inline
import seaborn as sns
sns.set()


import plotly.express as px
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

#Racing Bar Chart
get_ipython().system('pip install bar_chart_race')
import bar_chart_race as bcr
from IPython.display import HTML


for dirname, _, filenames in os.walk('C:\Data\penguins'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




# In[38]:


# Pull in the dataset, take a quick look at first few lines

df = pd.read_csv('C:\Data\penguins\penguins_size.csv')
df.head()


# In[39]:


#learning more: what datatype are the columns?

df.info()


# In[40]:


#learning more: size/shape of df, generic stat overview

print(df.shape)
df.describe(include='all')


# In[41]:


# Covariance & Correlation

# “Covariance” indicates the direction of the linear relationship between variables.
# “Correlation” on the other hand measures both the strength and direction of the linear relationship between two variables. 

print('Covariance:')
df.cov()


# In[42]:


print('Correlation:')
df.corr()


# In[43]:


# how clean is this data? Credit: Will Koehrsen

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(2)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

missing= missing_values_table(df)
missing


# In[44]:


#show what missing values, same as above just in original column order
df.isnull().sum()


# In[45]:


# Handling missing values
from sklearn.impute import SimpleImputer

#setting strategy to 'most frequent' to impute by the mean
imputer = SimpleImputer(strategy='most_frequent')# strategy can also be mean or median 
df.iloc[:,:] = imputer.fit_transform(df)


# In[46]:


# can now see missing values are "smoothed over"

df.isnull().sum()


# In[47]:


# sex column to integers

lb = LabelEncoder()
df["sex"] = lb.fit_transform(df["sex"])
df['sex'][:5]


# In[53]:


# how many species in this df?

df['species'].value_counts()


# In[67]:


df['species'].value_counts().iplot(kind='bar')


# In[55]:


df1 = df[['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm']]
sns.boxplot(data=df1, width=0.5,fliersize=5)


# In[57]:


#quick visual showing correlation beween features
#goal is to pull out the important features that affect classification the most

sns.pairplot(df, hue="species", height=3,diag_kind="hist")


# In[70]:


sns.FacetGrid(df, hue="species", size=8
            ) \
   .map(plt.scatter, "culmen_length_mm", "culmen_depth_mm") \
   .add_legend()


# In[74]:


ax = sns.violinplot(x="species", y="flipper_length_mm", data=df,size=12)


# In[78]:


sns.FacetGrid(df, hue="species", size=6,)    .map(sns.kdeplot, "flipper_length_mm",shade=True)    .add_legend()


# In[66]:


sns.FacetGrid(df, hue="species", size=8)    .map(plt.scatter, "body_mass_g", "flipper_length_mm")    .add_legend()


# In[ ]:




