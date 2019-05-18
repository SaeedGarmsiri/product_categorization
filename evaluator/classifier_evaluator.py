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
        col = ['url_id', 'comp_title', 'comp_brand', 'comp_cat1', 'comp_cat2', 'comp_cat3', 'comp_warranty', 'supply_maincatEn', 'site_firstcat', 'product_id']
        df = df[col]
        # df.dropna(inplace=True)
        df.drop(['url_id'], inplace=True, axis=1)
        df['features'] = df['comp_title'] + ' ' + df['comp_brand'] + ' ' + df['comp_cat1'] + ' ' + \
                         df['comp_cat2'] + ' ' + df['comp_cat3'] + ' ' + df['comp_warranty']
        cols = ['product_id', 'features', 'site_firstcat']
        df = df[cols]
        df.replace(np.nan, ' ', inplace=True, regex=True)
        return df
