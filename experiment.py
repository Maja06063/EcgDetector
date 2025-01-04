from health_table import HEALTH_IDS_LIST
from characteristic_values import CharacteristicValues
from characteristic_values_finder import CharacteristicValuesFinder
from characteristic_images_finder import CharacteristicImagesFinder

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from scipy.signal import find_peaks, peak_widths

import numpy as np
import matplotlib.pyplot as plt
import wfdb

import logging

#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout

class Experiment:

    EXPERIMENTAL_RABBIT_ID = 100
    FLAG_IMAGE = False
    FLAG_NORMALIZE = False
    FLAG_K = 0

    def get_time_list_for_value_list(self, values_list: list) -> list:

        return list(np.arange(0, len(values_list)) / self.SAMPLING_FREQ)

    def read_dataset(self, records_files) -> None:

        self.dataset = {}
        print("\nWczytywanie pacjentów...\n")

        for record_file_name in records_files:

            record_file = wfdb.rdrecord(record_file_name[:-4]) # Odczyt pliku pacjenta
            patient_id = int(record_file_name[-7:-4]) # Odczyt id pacjenta

            # Odczyt i uzyskanie napięcia od czasu z sygnału EKG pacjenta:
            signals = record_file.p_signal
            signal = [float(row[0]) for row in signals]

            # Zapisanie przebiegu EKG do słownika (klucz to id pacjenta):
            self.dataset[patient_id] = signal
            logging.debug(f"Wczytano pacjenta {patient_id}")

        # Posortowanie słownika po kluczach:
        self.dataset = {key: self.dataset[key] for key in sorted(self.dataset)}

    def run(self) -> None:

        # Lista na wyniki ekstrakcji i klasyfikatory:
        characteristic_values_list = []
        knn = KNeighborsClassifier(n_neighbors=self.FLAG_K)
        cvf = CharacteristicValuesFinder()
        cif = CharacteristicImagesFinder()

        print("\nEkstrakcja cech...\n")

        if self.FLAG_IMAGE:
            cif.fit(self.dataset[self.EXPERIMENTAL_RABBIT_ID], self.EXPERIMENTAL_RABBIT_ID in HEALTH_IDS_LIST)
            characteristic_values_list = cvf.predict(self.dataset)

        else:
            cvf.fit(self.dataset[self.EXPERIMENTAL_RABBIT_ID], self.EXPERIMENTAL_RABBIT_ID in HEALTH_IDS_LIST)
            characteristic_values_list = cvf.predict(self.dataset)

        print("\nEkstrakcja cech zakończona.")
        print("Klasyfikacja na podstawie cech...\n")

        # KNN
        X_train, X_test, y_train, y_test = train_test_split(
            [x.to_list() for x in characteristic_values_list],
            [y.is_sick() for y in characteristic_values_list],
            test_size=0.2
        )

        # Wytrenuj model
        knn.fit(X_train, y_train)

        # Dokonaj predykcji na zbiorze testowym
        y_pred = knn.predict(X_test)

        # Oceń model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Dokładność: {accuracy:.2f}")
