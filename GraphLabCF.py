
# coding: utf-8

# In[2]:

import graphlab
import pandas as pd
bookData = pd.read_excel('/Users/mukhar207/Desktop/DataProgramming/Python/Project/Datasets/FinalD_Mukhar.xlsx', sheetname = 'FinalData')
bookData.head()


# In[3]:

bookData.head()


# In[6]:

newBookDatadict = {'UserID':['count']}
newBookDatadf = bookData.groupby('UserID').agg(newBookDatadict)
newBookDatadf = newBookDatadf.reset_index()
newBookDatadf.columns = ('UserID','UserIDCount')
newBookDatadf = newBookDatadf[(newBookDatadf.UserIDCount>20)]


# In[7]:

joinedBookData = pd.merge(newBookDatadf,bookData, on = "UserID", how = "inner")
joinedBookData.head()


# In[8]:

del joinedBookData['UserIDCount']


# In[9]:

joinedBookData


# In[10]:

from sklearn.cross_validation import train_test_split
splitData = joinedBookData
book_trainData, book_testData = train_test_split(splitData,test_size=0.4, random_state = 123)


# In[11]:

book_trainData.shape


# In[12]:

book_testData.shape


# In[14]:

book_trainData.dtypes


# In[15]:

train_data = graphlab.SFrame(book_trainData)
test_data = graphlab.SFrame(book_testData)


# In[16]:

train_data


# In[17]:

test_data


# In[18]:

popularity_model = graphlab.popularity_recommender.create(train_data, user_id='UserID', item_id='ISBN', target='Rating')


# In[24]:

item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='UserID', item_id='ISBN', 
                                                             target='Rating', similarity_type='cosine')


# In[28]:

#Make Recommendations:
item_collaborative_recomm = item_sim_model.recommend(k=3)
item_collaborative_recomm.print_rows(num_rows=100)


# In[29]:

model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])
graphlab.show_comparison(model_performance,[popularity_model, item_sim_model])


# In[ ]:



