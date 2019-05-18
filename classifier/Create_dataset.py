import os
import pandas as pd
import numpy as np


class CreateDataset:

    def __init__(self):
        d = os.path.dirname(os.path.abspath(__file__))
        self.filename = d + '/data/dataset.csv'
        self.source_data = d + '/data/source_data.csv'
        self.train_data = d + '/data/train.csv'
        self.test_data = d + '/data/test.csv'

    def create(self):

        df = pd.read_csv(self.filename)
        df.head()
        col = ['url_id', 'bm_title', 'bm_brand', 'bm_cat1', 'bm_cat2', 'bm_cat3', 'bm_warranty', 'DK_Entitle', 'DK_Fatitle', 'DK_titlealt', 'DK_titleKey', 'DK_site_firstcat', 'DK_site_secondcat', 'DK_site_thirdcat', 'DK_supply_catFa', 'DK_site_firstcat', 'DK_supply_maincatFa', 'DK_supply_maincatEn', 'DK_brandEn', 'DK_brandFa', 'dkp']
        df = df[col]
        df = df.sample(frac=1).reset_index(drop=True)
        # zz = df.isnull().values.any()
        df = df[df.bm_title.notnull()]
        # zx = df.isnull().values.any()
        msk = np.random.rand(len(df)) <= 0.8
        train = df[msk]
        test = df[~msk]
        df.to_csv(self.source_data, index=False)
        train.to_csv(self.train_data, index=False)
        test.to_csv(self.test_data, index=False)


if __name__ == '__main__':
    create_data_set = CreateDataset()
    create_data_set.create()
