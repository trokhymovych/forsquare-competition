import re
from timeit import default_timer as timer
from datetime import timedelta

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

    def prepare(self):
        self.extract_adress_number()
        return self.train_df


if __name__ == "__main__":
    features = BaseFeatures()
    features.prepare()
    print(features.train_df.shape)
    print(features.train_df.columns)
