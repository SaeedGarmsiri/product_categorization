import pandas as pd
import numpy as np
from classifier.data_manipulator import DataManipulator
from sklearn.ensemble import RandomForestClassifier
import datetime
from sklearn.externals import joblib
import os


class Classifier:
    def __init__(self):
        d = os.path.dirname(os.path.abspath(__file__))
        pass
        self.accs = [0, 0, 0, 0, 0]
        self.manipulator = DataManipulator()
        self.filename = d + '/data/rf.joblib.pkl'

    def train(self):
        d = os.path.dirname(os.path.abspath(__file__))
        filename = d + '/data/train.csv'
        train, x_train, y_train = self.manipulator.split_label(filename)
        x_train = self.manipulator.cleaner(x_train)
        features_train = self.manipulator.train_feature_engineering(x_train)
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(features_train, y_train)
        joblib.dump(rf, self.filename, compress=9)
        accu_train_rf = np.mean(rf.predict(features_train) == y_train)
        return accu_train_rf

    def test(self, test):
        d = os.path.dirname(os.path.abspath(__file__))
        x_test = pd.DataFrame.from_records(test)
        x_test.head()
        col = ['url_id', 'features']
        x_test = x_test[col]
        urls = x_test['url_id']
        features = x_test['features']
        # x_test = pd.Series({'features': test})
        features = self.manipulator.cleaner(features)
        features_test = self.manipulator.test_feature_engineering(features)
        clf = joblib.load(self.filename)
        rf_predicted = clf.predict(features_test)
        final_log = pd.DataFrame({'url_id': urls, 'predicted': rf_predicted})
        final_log.to_csv(d + "/data/log{0}.csv".format(datetime.datetime.now()), index=False)
        return final_log
