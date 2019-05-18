import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import libraries.persian as persian
from sklearn.externals import joblib
import os


class DataManipulator:

    def __init__(self):
        d = os.path.dirname(os.path.abspath(__file__))
        self.count_vect_file = d + '/data/cv.joblib.pkl'
        self.tfidf_transformer_file = d + '/data/tfidf.joblib.pkl'

    @staticmethod
    def read_data(adr):
        df = pd.read_csv(adr)
        df.head()
        df.replace(np.nan, ' ', inplace=True, regex=True)
        col = ['url_id', 'bm_title', 'bm_brand', 'bm_cat1', 'bm_cat2', 'bm_cat3',
               'bm_warranty', 'DK_site_firstcat', 'DK_supply_maincatEn', 'dkp']
        df = df[col]
        df.drop(['url_id'], inplace=True, axis=1)
        df['features'] = df['bm_title'] + ' ' + df['bm_brand'] + ' ' + df['bm_cat1'] + ' ' + \
                         df['bm_cat2'] + ' ' + df['bm_cat3'] + ' ' + df['bm_warranty']
        ff = df.loc[df['DK_supply_maincatEn'] == 'Fresh Food']
        df.drop(ff.index, inplace=True)
        cols = ['dkp', 'features', 'DK_site_firstcat']
        df = df[cols]
        return df

    def split_label(self, adr):

        df = self.read_data(adr)
        x_train = df['features']
        y_train = df['DK_site_firstcat']
        return df, x_train, y_train

    def train_feature_engineering(self, x):
        count_vect = CountVectorizer(encoding='utf-8', stop_words='english', strip_accents='unicode',
                                     lowercase=True, analyzer='word')
        tfidf_transformer = TfidfTransformer(use_idf=True)
        x_counts = count_vect.fit_transform(x)
        joblib.dump(count_vect, self.count_vect_file, compress=9)
        features = tfidf_transformer.fit_transform(x_counts)
        joblib.dump(tfidf_transformer, self.tfidf_transformer_file, compress=9)
        return features

    def test_feature_engineering(self, x):
        count_vect = joblib.load(self.count_vect_file)
        tfidf_transformer = joblib.load(self.tfidf_transformer_file)
        x_counts = count_vect.transform(x)
        features = tfidf_transformer.fit_transform(x_counts)
        return features

    def clean(self, x):
        d = os.path.dirname(os.path.abspath(__file__))
        # words_count_before = x.str.split(expand=True).stack().value_counts()
        # words_count_before.to_csv(d + "/data/train_words_count_before{0}.csv".format(datetime.datetime.now()))
        stopwords = pd.read_csv(d + "/data/stopwords.csv", header=0)
        stop_list = stopwords['word'].tolist()
        x = x.str.split(' ').apply(lambda z: ' '.join(k for k in z if k not in stop_list))
        if x.size > 1:
            for j in range(0, x.size - 1):
                x.iloc[j] = persian.convert_ar_characters(x.iloc[j])
                x.iloc[j] = persian.convert_fa_numbers(x.iloc[j])
        else:
            for j in range(0, x.size):
                x.iloc[j] = persian.convert_ar_characters(x.iloc[j])
                x.iloc[j] = persian.convert_fa_numbers(x.iloc[j])

        # words_count_after = x.str.split(expand=True).stack().value_counts()
        # words_count_after.to_csv(d + "/data/train_words_count_after{0}.csv".format(datetime.datetime.now()))
        return x
