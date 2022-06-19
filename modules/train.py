from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pandarallel import pandarallel
import multiprocessing

import Levenshtein
import difflib

import gc

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--country",
                    help="display a square of a given number")

args = parser.parse_args()
cc = args.country
print(cc)


pandarallel.initialize(nb_workers=multiprocessing.cpu_count()-1, progress_bar=False)

pd.set_option('display.max_columns', None)

# cc = "GR"

N_TRAIN = 4000000
df = pd.read_csv(f"../data/train_cc/training_dataset_prepared_{cc}.csv")
if len(df) > N_TRAIN:
    df = df.sample(N_TRAIN)
    df = df.reset_index(drop=True)

print(len(df))
train = pd.read_csv("../data/train.csv")


### Creating train index of features

ids_to_leave = set(df.id_1) | set(df.id_2)
train_cc = train[train.id.isin(ids_to_leave)]

train_cc_index = {}
for i, row in tqdm(train_cc.iterrows(), ):
    train_cc_index[row.id] = row
    
    
### Recreating features (it takes some time)

feat_1 = [train_cc_index[i] for i in df["id_1"]]
feat_2 = [train_cc_index[i] for i in df["id_2"]]

df1 = pd.DataFrame(feat_1)
df2 = pd.DataFrame(feat_2)

df1.columns = [f"{c}_1" for c in df1.columns]
df2.columns = [f"{c}_2" for c in df2.columns]

df1 = df1.reset_index(drop=True)
df2 = df2.reset_index(drop=True)

train_df = pd.concat([df1, df2], axis=1)
train_df["feat_incl"] = df["feat_incl"]


### feature creation

# ===============================================================================
# Get manhattan distance
# ===============================================================================
def manhattan(lat1, long1, lat2, long2):
    return np.abs(lat2 - lat1) + np.abs(long2 - long1)

# ===============================================================================
# Get haversine distance
# ===============================================================================
def vectorized_haversine(lats1, lats2, longs1, longs2):
    radius = 6371
    dlat=np.radians(lats2 - lats1)
    dlon=np.radians(longs2 - longs1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lats1)) \
        * np.cos(np.radians(lats2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c
    return d

# ===============================================================================
# Compute distances + Euclidean
# ===============================================================================
def add_lat_lon_distance_features(df):
    lat1 = df['latitude_1']
    lat2 = df['latitude_2']
    lon1 = df['longitude_1']
    lon2 = df['longitude_2']
    df['latdiff'] = (lat1 - lat2)
    df['londiff'] = (lon1 - lon2)
    df['manhattan'] = manhattan(lat1, lon1, lat2, lon2)
    df['euclidean'] = (df['latdiff'] ** 2 + df['londiff'] ** 2) ** 0.5
    df['haversine'] = vectorized_haversine(lat1, lat2, lon1, lon2)
    col_64 = list(df.dtypes[df.dtypes == np.float64].index)
    for col in col_64:
        df[col] = df[col].astype(np.float32)
    return df

# ===============================================================================
# Compute distances for categorical features
# ===============================================================================
def get_distance_cat(df, column):
    
    def get_gesh(tup):
        str1, str2 = tup
        if str1==str1 and str2==str2:
            return difflib.SequenceMatcher(None, str1, str2).ratio()
        else:
            return -1
        
    def get_levens(tup):
        str1, str2 = tup
        if str1==str1 and str2==str2:
            return Levenshtein.distance(str1, str2)
        else:
            return -1
        
    def get_jaros(tup):
        str1, str2 = tup
        if str1==str1 and str2==str2:
            return Levenshtein.jaro_winkler(str1, str2)
        else:
            return -1
    
    for s in [column + '_1', column + '_2']:
        df[s] = df[s].astype(str).str.lower()
    tuple_series = pd.Series([(s1,s2) for s1,s2 in df[[column + '_1', column + '_2']].values])
    df[f"{column}_geshs"] = tuple_series.parallel_apply(get_gesh)
    df[f"{column}_levens"] = tuple_series.apply(get_levens)
    df[f"{column}_jaros"] = tuple_series.apply(get_jaros)

    del tuple_series
    gc.collect()
    
    if column not in ['country', 'phone', 'zip']:
        df[f"{column}_len_1"] = df[column + '_1'].astype(str).apply(len).astype(int)
        df[f"{column}_len_2"] = df[column + '_2'].astype(str).apply(len).astype(int)
        df[f"{column}_nlevens"] = df[f"{column}_levens"] / df[[f"{column}_len_1", f"{column}_len_2"]].max(axis = 1)
    col_64 = list(df.dtypes[df.dtypes == np.float64].index)
    for col in col_64:
        df[col] = df[col].astype(np.float32)
#     df = pd.concat([df, df1], axis = 1)
    return df

# ====================================================
# GETTING CATEGORIES CLUSTERS MATCH
# ====================================================
## Add categorical features:
import pickle
with open('../data/category_precalculated_mappings_4.pickle', 'rb') as handle:
# with open('../input/category-mapping-dict/category_precalculated_mappings_4.pickle', 'rb') as handle:
    cat_mapping = pickle.load(handle)

def get_cats_indx(cats):
    return set([cat_mapping["cat_mapping"].get(c, -1) for c in cats.split(", ")])

def add_cat_match(df):
    cat_1 = df.categories_1.fillna("__nan__").apply(get_cats_indx)
    cat_2 = df.categories_2.fillna("__nan__").apply(get_cats_indx)
    df['cat_match'] = [len(a&b)/len(a|b) for a,b in zip(cat_1, cat_2)]
    return df

# ====================================================
# COMPLEX FUNCTION FOR FEATURES PREPARATION
# ====================================================
def prepare_features_initial(df):
    # Numerical Feature Engineering
    df = add_lat_lon_distance_features(df)
    # Categorical Feature Engineering
    cat_columns = ['name', 'address', 'city', 'state', 'zip', 'country', 'url', 'phone', 'categories']
    pair_cat_columns = [col + '_1' for col in cat_columns] + [col + '_2' for col in cat_columns]
    for col in cat_columns:
        df = get_distance_cat(df, col)
    # Add category match:
    df = add_cat_match(df)
    return df

train_df = prepare_features_initial(train_df)


def generate_train_target(df, dataset):
    poi_dict = {k: v for k, v in zip(df['id'], df['point_of_interest'])}
    poi_1 = dataset["id_1"].map(poi_dict)
    poi_2 = dataset["id_2"].map(poi_dict)
    return (poi_1==poi_2).astype(int)

target = generate_train_target(train, train_df)
train_df["target"] = target


features_to_use = [
       'latitude_1', 'longitude_1',
       'latitude_2', 'longitude_2', 'feat_incl', 'latdiff', 'londiff',
       'manhattan', 'euclidean', 'haversine', 'name_geshs', 'name_levens',
       'name_jaros', 'name_len_1', 'name_len_2', 'name_nlevens', 'cat_match',
       'address_geshs', 'address_levens', 'address_jaros', 'address_len_1',
       'address_len_2', 'address_nlevens', 'city_geshs', 'city_levens',
       'city_jaros', 'city_len_1', 'city_len_2', 'city_nlevens', 'state_geshs',
       'state_levens', 'state_jaros', 'state_len_1', 'state_len_2',
       'state_nlevens', 'zip_geshs', 'zip_levens', 'zip_jaros',
       'country_geshs', 'country_levens', 'country_jaros', 'url_geshs',
       'url_levens', 'url_jaros', 'url_len_1', 'url_len_2', 'url_nlevens',
       'phone_geshs', 'phone_levens', 'phone_jaros', 'categories_geshs',
       'categories_levens', 'categories_jaros', 'categories_len_1',
       'categories_len_2', 'categories_nlevens'
]

# cat_features = ["categories_1", "categories_2"]

# dummy split + shuffling
len_train = len(train_df)
train_dataset = train_df.sample(len_train)
target = train_dataset.target
spli_index = int(len_train*0.8)


train_c = train_dataset[:spli_index]
train_pool = Pool(data=train_c[features_to_use],
                  label=target[:spli_index],)
#                       cat_features = cat_features)

test_t = train_dataset[spli_index:]
test_pool = Pool(data=test_t[features_to_use],
                 label=target[spli_index:],)
#                      cat_features = cat_features)

classes = np.unique(target[:spli_index])
weights = compute_class_weight(class_weight='balanced', classes=classes, y=target[:spli_index])
class_weights = dict(zip(classes, weights))

clf = CatBoostClassifier(iterations=7000, 
                         learning_rate=0.01, 
                         loss_function='Logloss',
                         custom_metric=["AUC", "Accuracy", "Recall", "Precision"],
                         one_hot_max_size = 10,
                         class_weights=class_weights,
                         use_best_model=True,
                         metric_period=500
)
clf.fit(train_pool, eval_set=test_pool,
        verbose=True,
        plot = False,
        metric_period=500
       )

train_df['pred'] = \
    clf.predict_proba(train_df[clf.feature_names_])[:, 1] > 0.96
true_dict = train_cc.groupby("point_of_interest")['id'].apply(list).to_dict()
poi_dict = {k: v for k, v in train_cc[["id", "point_of_interest"]].values}
pred_dict_1_f = train_df[train_df.pred==1].groupby("id_1")["id_2"].apply(list).to_dict()
pred_dict_2_f = train_df[train_df.pred==1].groupby("id_2")["id_1"].apply(list).to_dict()

final_pred_dict = {k: list(set(pred_dict_1_f.get(k, [])) | set(pred_dict_2_f.get(k, []))) 
                 for k in (set(pred_dict_1_f.keys()) | set(pred_dict_2_f.keys()))}

def calculate_metrics(poi_dict, true_dict, df, pred_dict):
    recall = []
    iou = []
    n_true = []
    n_pred = []
    
    for po_id in tqdm(df.id_1.unique()):
        correct_list = true_dict[poi_dict[po_id]]
        pred_list = pred_dict.get(po_id, []) + [po_id]
        recall.append(len(set(correct_list) & set(pred_list)) / len(set(correct_list)))
        iou.append(len(set(correct_list) & set(pred_list)) / len(set(correct_list) | set(pred_list)))
        n_true.append(len(set(correct_list)))
        n_pred.append(len(set(pred_list)))
    res = (np.mean(recall), np.mean(iou), np.mean(n_true), np.mean(n_pred))
    
    res_dict_recall = {k: v for k, v in tqdm(zip(df.id_1.unique(), recall))}
    res_dict_iou = {k: v for k, v in tqdm(zip(df.id_1.unique(), iou))}
    res_dict_n_true = {k: v for k, v in tqdm(zip(df.id_1.unique(), n_true))}
    res_dict_n_pred = {k: v for k, v in tqdm(zip(df.id_1.unique(), n_pred))}
    
    del pred_dict, recall, iou
    gc.collect()
    return res, res_dict_recall, res_dict_iou, res_dict_n_true, res_dict_n_pred

res, res_dict_recall, res_dict_iou, res_dict_n_true, res_dict_n_pred = \
    calculate_metrics(poi_dict, true_dict, train_df, final_pred_dict)
print(cc, res)

clf.save_model(f"model_initial_{cc}.bin")