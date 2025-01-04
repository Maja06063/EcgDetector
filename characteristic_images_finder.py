from characteristic_values import CharacteristicValues
from common_finder import CommonFinder
from health_table import HEALTH_IDS_LIST
from scipy.signal import find_peaks
from sklearn.neural_network import MLPClassifier

import logging
import matplotlib.pyplot as plt
import numpy as np

class CharacteristicImagesFinder(CommonFinder):

    def __init__(self, _ifNormalize: bool, _plot_number: int):

        self.nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
        self.ifNormalize = _ifNormalize
        self.plot_number = _plot_number

    def fit(self, x_train, y_train):
        pass

    def predict(self, x_test) -> list:
        pass
