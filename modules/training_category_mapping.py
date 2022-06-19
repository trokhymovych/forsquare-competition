# imports needed and logging
import gzip
import gensim 
import logging

from sklearn.cluster import KMeans
import numpy as np
import pickle

import pandas as pd 

train = pd.read_csv("../data/train.csv")
categories_list = train.groupby("point_of_interest")["categories"].apply(list)

# building documents
documents = []

for cats in categories_list:
    tmp_doc = []
    for el in cats:
        if pd.isna(el):
            continue
        tmp_ = el.split(", ")
        if len(tmp_) > 0:
            tmp_doc += tmp_
    if len(tmp_doc) > 1:
        documents.append(tmp_doc)
        
# build vocabulary and train model
model = gensim.models.Word2Vec(
    documents,
    window=20,
    min_count=1,
    workers=10)


N_CLUSTERS = 50

km = KMeans(
    n_clusters=N_CLUSTERS, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(model.wv.vectors)

cat_mapping_to_id = {}
cat_mapping_to_id["unknown"] = -1

for i in range(N_CLUSTERS):
    for j in np.array(model.wv.index_to_key)[np.where(y_km==i)[0]]:
        cat_mapping_to_id[j] = i
        
cat_vec_mapping = {k: v for k, v in zip(model.wv.index_to_key, model.wv.vectors)}
cat_vec_mapping["__nan__"] = model.wv.vectors.mean(axis=0)

a = {"cat_vec_mapping": cat_vec_mapping,
     "cat_mapping": cat_mapping_to_id}

with open('../data/category_mappings_new_tunning.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)