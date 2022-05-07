import numpy as np
from fuzzywuzzy import fuzz
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.utils.class_weight import compute_class_weight

FEATURE_TO_USE = ['dist', 'fuzz_ration', 'fuzz_partial_ratio',
                   "categories_1", "categories_2",
                   "zip_same", "phone_same", "categories_intersect"]
CAT_FEATURES = ["categories_1", "categories_2",
                "zip_same", "phone_same", "categories_intersect"]


class ModelLevelTwo:
    def __init__(self, pairs):
        self.pairs = pairs
        self.pairs = self.prepare(self.pairs)
        self.clf = self.train()

    def prepare(self, df):
        """ TODO: move it to preparator_pairs"""
        # print("Start preparing")
        df['dist'] = df.apply(lambda row: self.haversine_np(row["latitude_1"], row["longitude_1"],
                                                            row["latitude_2"], row["longitude_2"]),
                                              axis=1)
        df['fuzz_ration'] = df.apply(lambda row: fuzz.ratio(row["name_1"], row["name_2"]), axis=1)
        df['fuzz_partial_ratio'] = df.apply(lambda row: fuzz.partial_ratio(
            row["name_1"], row["name_2"]), axis=1)

        df["phone_same"] = df["phone_2"] == df["phone_1"]
        df["zip_same"] = df["zip_2"] == df["zip_1"]

        df["categories_set_1"] = df["categories_1"].fillna("empty_cat").apply(str).apply(lambda x: set(x.split(", ")))
        df["categories_set_2"] = df["categories_2"].fillna("empty_cat").apply(str).apply(lambda x: set(x.split(", ")))

        df['categories_intersect'] = df.apply(lambda row:
                                              len(row["categories_set_1"] & row["categories_set_2"]),
                                              axis=1)

        df[CAT_FEATURES] = df[CAT_FEATURES].fillna(-1)

        # print("Finish preparing")
        return df

    def train(self):
        print("Start training")
        # todo improve model with new features
        # todo move it to config or whatever

        self.pairs.match = self.pairs.match.astype(int)

        train_pool = Pool(data=self.pairs[FEATURE_TO_USE],
                          label=self.pairs.match.astype(int),
                          cat_features=CAT_FEATURES)

        # todo: review as it is optional step
        classes = np.unique(self.pairs.match)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.pairs.match)
        class_weights = dict(zip(classes, weights))

        clf = CatBoostClassifier(iterations=1000,
                                 learning_rate=0.01,
                                 loss_function='Logloss',
                                 custom_metric=["AUC", "Accuracy"],
                                 one_hot_max_size=100,
                                 # class_weights=class_weights,
                                 metric_period=100
                                 )

        clf.fit(train_pool,
                verbose=True,
                plot=True
                )
        print('CatBoost model is fitted: ' + str(clf.is_fitted()))
        print('CatBoost model parameters:')
        print(clf.get_params())
        return clf

    def predict(self, pairs):
        return self.clf.predict_proba(pairs[FEATURE_TO_USE])[:, 1] > 0.9

    @staticmethod
    def haversine_np(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)

        All args must be of equal length.

        """
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return km