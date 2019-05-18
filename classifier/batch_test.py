import os
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from classifier.classify_products import Classifier, DataManipulator

if __name__ == '__main__':
    d = os.path.dirname(os.path.abspath(__file__))
    filename = d + '/data/test.csv'
    clf_name = d + '/data/rf.joblib.pkl'
    clf = Classifier()
    manipulator = DataManipulator()
    test, x_test, y_test = manipulator.split_label(filename)
    x_test = manipulator.cleaner(x_test)
    features_test = manipulator.test_feature_engineering(x_test)
    clf = joblib.load(clf_name)
    rf_predicted = clf.predict(features_test)
    actual = np.array(y_test)
    accu_test_rf = np.mean(rf_predicted == actual)
    print(precision_score(actual, rf_predicted, average="macro"))
    print(recall_score(actual, rf_predicted, average="macro"))
    print(f1_score(actual, rf_predicted, average="macro"))
    print(accu_test_rf)
