import re
from timeit import default_timer as timer
from datetime import timedelta
from urllib.parse import urlparse

import pandas as pd

TRAIN_PATH = "data/train.csv"

class BaseFeatures:
    def __init__(self, file_path: str = TRAIN_PATH) -> None:
        self.train_df = pd.read_csv(file_path)

    def extract_adress_number(self) -> None:
        def extract_number(adress: str) -> int:
            # todo: check what to do with multiple numbers in list
            if adress:
                try:
                    result = re.findall(r'\d+', adress)
                    return int(''.join(result))
                except:
                    return 0
            else:
                return 0
        
        start = timer()
        self.train_df["adress_number"] = self.train_df["address"].apply(extract_number)
        end = timer()
        print("adress_number was genareted by", timedelta(seconds=end-start))

    def parse_url_domain(self):
        def parse_domain(url: str):
            try:
                return urlparse(url).netloc
            except:
                pass

        start = timer()
        self.train_df["domain"] = self.train_df["url"].apply(parse_domain)
        end = timer()
        print("domain was parsed by", timedelta(seconds=end-start))

    def calc_url_features(self):
        def get_www(url):
            try:
                if "www" in url:
                    return 1
                else:
                    return 2
            except:
                return 0
        

        def get_protocol(url):
            try:
                if "https" in url:
                    return 1
                else:
                    return 2
            except:
                return 0

        start = timer()
        self.train_df["is_www"] = self.train_df["url"].apply(get_www)
        self.train_df["protocol"] = self.train_df["url"].apply(get_protocol)
        end = timer()
        print("URL features was calculated by", timedelta(seconds=end-start))




    def prepare(self):
        self.extract_adress_number()
        self.parse_url_domain()
        self.calc_url_features()
        return self.train_df


if __name__ == "__main__":
    features = BaseFeatures()
    features.prepare()
    print(features.train_df.shape)
    print(features.train_df.columns)
