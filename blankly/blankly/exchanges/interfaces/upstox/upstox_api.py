import pandas as pd
import json


class UpstoxAPI:
    def get_assets_list(self):
        self.download_assets_list()
        with open('./assets_cache/assets.json', 'r') as f:
            data = json.load(f)
        return data

    def download_assets_list(self):
        csv_file = pd.DataFrame(pd.read_csv("./assets_cache/complete.csv", sep=",", header=0, index_col=False))
        csv_file.to_json("./assets_cache/assets.json", orient="records", date_format="epoch", double_precision=10,
                         force_ascii=True, date_unit="ms", default_handler=None)
