#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


news_df = pd.read_csv('train.csv')


# In[3]:


news_df.head()


# In[4]:


news_df.shape


# In[5]:


news_df.isna().sum()


# In[6]:


news_df = news_df.fillna(' ')


# In[7]:


news_df.isna().sum()


# In[8]:


news_df['content'] = news_df['author']+" "+news_df['title']


# In[9]:


news_df


# In[10]:


news_df['content']


# In[11]:


# stemming
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[12]:


news_df['content'] = news_df['content'].apply(stemming)


# In[13]:


news_df['content']


# In[14]:


X = news_df['content'].values
y = news_df['label'].values


# In[15]:


print(X)


# In[16]:


vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)


# In[17]:


print(X)


# In[18]:


print(X.shape)
print(y.shape)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


# In[20]:


X_train.shape


# In[21]:


X_test.shape


# In[22]:


model = LogisticRegression()
model.fit(X_train,y_train)


# In[23]:


train_y_pred = model.predict(X_train)
print("train accurracy :",accuracy_score(train_y_pred,y_train))


# In[24]:


test_y_pred = model.predict(X_test)
print("train accurracy :",accuracy_score(test_y_pred,y_test))


# In[25]:


y = np.array([17223, 17223])
mylabels = ["True", "False"]


# In[26]:


plt.pie(y, labels = mylabels)
plt.show()


# In[27]:


input_data = X_test[25]
prediction = model.predict(input_data)
if prediction[0] == 1:
    print('Fake news')
else:
    print('Real news')


# In[28]:


news_df['content'][24]


# In[29]:


from sklearn.metrics import confusion_matrix

# Evaluate on the test set
conf_matrix = confusion_matrix(y_test, test_y_pred)

# Display the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Extract values from confusion matrix
true_negative, false_positive, false_negative, true_positive = conf_matrix.ravel()

# Calculate precision, recall, and F1-score
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * (precision * recall) / (precision + recall)

# Display precision, recall, and F1-score
print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# In[33]:


get_ipython().system('pip install textstat')


# In[ ]:




