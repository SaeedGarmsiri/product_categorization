# import mysql.connector
import pandas as pd
import numpy as np
from classifier.classify_products import Classifier


class Evaluator:

    def train(self):
        clf = Classifier()
        accuracy = clf.train()
        return accuracy

    def test(self, test_data):
        clf = Classifier()
        predicted = clf.test(test_data)
        return predicted

    @staticmethod
    def read_data(addr):
        df = pd.read_csv(addr)
        df.head()
        col = ['url_id', 'bm_title', 'bm_brand', 'bm_cat1', 'bm_cat2', 'bm_cat3', 'bm_warranty', 'DK_supply_maincatEn', 'DK_site_firstcat', 'dkp']
        df = df[col]
        # df.dropna(inplace=True)
        df.drop(['url_id'], inplace=True, axis=1)
        df['features'] = df['bm_title'] + ' ' + df['bm_brand'] + ' ' + df['bm_cat1'] + ' ' + \
                         df['bm_cat2'] + ' ' + df['bm_cat3'] + ' ' + df['bm_warranty']
        var2 = df.loc[df['DK_supply_maincatEn'] == 'Fresh Food']
        df.drop(var2.index, inplace=True)
        cols = ['dkp', 'features', 'DK_site_firstcat']
        df = df[cols]
        df.replace(np.nan, ' ', inplace=True, regex=True)
        return df
