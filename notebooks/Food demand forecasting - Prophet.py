#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Import library and dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from fbprophet import Prophet
plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings('ignore')

import pickle


# In[19]:


raw = pd.read_csv('dss_mock_1.csv')


# In[20]:


raw.describe()


# In[21]:
# cleaning dataset
# convert 'epoch' date into appropriate datetime
from datetime import datetime
tf = raw.copy()
tf['date_conv'] = raw['date'].apply(lambda x : datetime.fromtimestamp(x).strftime("%Y-%m-%d"))


# In[22]:
tf.drop(['date','is_holiday'], axis=1, inplace=True)


# In[23]:


tf['date_conv'] = pd.DatetimeIndex(tf['date_conv'])
tf = tf.sort_values(by='date_conv').reset_index(drop=True)


# In[24]:


tf.dtypes


# In[25]:
tf.tail()


# In[26]:
cols = list(tf.columns.values)
cols.remove('date_conv')
###############################################################################

# In[27]:
# Model fit
for col in cols:
    subtf = tf[['date_conv',col]]
    subtf = subtf.rename(columns={'date_conv':'ds',col:'y'})
    model = Prophet(interval_width=0.95)
    model.fit(subtf)
    pickle.dump(model, open('model'+'_'+col+'.pickle','wb'))


# In[30]:
# Model predict
results = []
for col in cols:
    model = pickle.load(open('model'+'_'+col+'.pickle', 'rb'))
    f_df = model.make_future_dataframe(periods=7, freq='D')
    result = model.predict(f_df)
    result_tf = result[['ds','yhat']].rename(columns={'yhat':col})
    results.append(result_tf)
tf.predict = pd.concat(results, axis=1)


# In[31]:


# Cleaning predict table
tf.predict = tf.predict.iloc[:,  ~tf.predict.columns.duplicated()]
tf.predict.drop_duplicates(subset='ds', keep="first", inplace=True)


# In[32]:


tf.predict = tf.predict.reset_index(drop=True).rename(columns={'ds':'date'})


# In[33]:


tf.predict.head()


# In[34]:


tf_revised = tf.predict[(tf.predict['date']>'2019-12-01')].reset_index(drop=True)


# In[35]:


tf_revised


# In[ ]:




