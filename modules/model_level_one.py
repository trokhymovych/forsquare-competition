import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import *
from whoosh import scoring

from tqdm.auto import tqdm
import os


class ModelLevelOne:
    def __init__(self, dataframe, ix_name=None):
        self.dataframe = dataframe

        self.ball_index = None
        self.index_location()

        # todo: field_boost to config + tuning
        self.schema = Schema(name=TEXT(stored=True, field_boost=5),
                             address=TEXT(stored=True, field_boost=1),
                             city=TEXT(stored=True, field_boost=1),
                             state=TEXT(stored=True, field_boost=1),
                             categories=TEXT(stored=True, field_boost=10),
                             idd=ID(stored=True))
        if ix_name:
            self.ix = open_dir(ix_name)
        else:
            self.ix = None

        self.mparser = \
            MultifieldParser(["name", "address", "city", "state"], schema=self.schema, group=OrGroup)
        self.cparser = QueryParser("categories", schema=self.schema, group=OrGroup)

    def index_location(self):
        for column in self.dataframe[["latitude", "longitude"]]:
            rad = np.deg2rad(self.dataframe[column].values)
            self.dataframe[f'{column}_rad'] = rad
        self.ball_index = BallTree(self.dataframe[["latitude_rad", "longitude_rad"]].values, metric='haversine')

    def index_texts(self, ix_name):

        if not os.path.exists(ix_name):
            os.mkdir(ix_name)
        self.ix = create_in(ix_name, self.schema)
        writer = self.ix.writer(procs=8, limitmb=256, multisegment=True)
        for name, address, city, state, categories, idd in tqdm(zip(
                self.dataframe.name, self.dataframe.address, self.dataframe.city,
                self.dataframe.state, self.dataframe.categories, self.dataframe.id)):
            writer.add_document(name=name if not pd.isna(name) else "",
                                address=address if not pd.isna(address) else "",
                                city=city if not pd.isna(city) else "",
                                state=state if not pd.isna(state) else "",
                                categories=categories if not pd.isna(categories) else "",
                                idd=idd)
        writer.commit()

    def get_closest_locations(self, lat, lon, n=1000):

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        distances, indices = self.ball_index.query([(lat_rad, lon_rad)], k=n)
        return self.dataframe.loc[indices[0], "id"], distances

    def get_closest_text(self, name, address="", city="", state="", categories=""):

        name, address, city, state, categories = \
            (self._process_text(t) for t in [name, address, city, state, categories])

        # todo: modify to use address, city, state, categories in query
        query_text = name

        searcher = self.ix.searcher(weighting=scoring.BM25F())
        query_parsed = self.mparser.parse(query_text)
        query_category = self.cparser.parse(categories)
        results = searcher.search(query_parsed | query_category, limit=1000)

        # todo: change this super hardcode logic (when model available -> we can allow wider set)
        results_scores = [r.score for r in results]
        max_score = results_scores[0]
        score_treshhold = max_score / 2
        results_scores = [r for r in results_scores if r > score_treshhold]

        results_ids = [r.fields()["idd"] for r in results if r.score > score_treshhold]

        return results_ids, results_scores

    @staticmethod
    def _process_text(text):
        return "" if pd.isna(text) else text
