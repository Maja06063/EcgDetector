from characteristic_images_finder import CharacteristicImagesFinder
from characteristic_values_finder import CharacteristicValuesFinder
from health_table import HEALTH_IDS_LIST
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier

import logging
import numpy as np
import wfdb

#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout

class Experiment:

    EXPERIMENTAL_RABBIT_ID = 100
    FLAG_IMAGE = False
    FLAG_NORMALIZE = False
    FLAG_K = 0

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
            logging.info(f"Wczytano pacjenta {patient_id}")

        # Posortowanie słownika po kluczach:
        self.dataset = {key: self.dataset[key] for key in sorted(self.dataset)}

    def run(self, plot_number: int) -> None:

        # Lista na wyniki ekstrakcji i klasyfikatory:
        characteristic_values_list = []
        knn = KNeighborsClassifier(n_neighbors=self.FLAG_K)
        cvf = CharacteristicValuesFinder(self.FLAG_NORMALIZE, plot_number)
        cif = CharacteristicImagesFinder(self.FLAG_NORMALIZE, plot_number)

        print("\nEkstrakcja cech...\n")

        if self.FLAG_IMAGE:
            cif.fit(self.dataset[self.EXPERIMENTAL_RABBIT_ID], self.EXPERIMENTAL_RABBIT_ID in HEALTH_IDS_LIST)
            characteristic_values_list = cif.predict(self.dataset)

        else:
            cvf.fit(self.dataset[self.EXPERIMENTAL_RABBIT_ID], self.EXPERIMENTAL_RABBIT_ID in HEALTH_IDS_LIST)
            characteristic_values_list = cvf.predict(self.dataset)

        print("\nEkstrakcja cech zakończona.")
        print("Klasyfikacja na podstawie cech...\n")

        # Przydzielenie pacjentów do chorych i zdrowych za pomocą KNN:
        kfold = KFold(n_splits=10, shuffle=True) # Walidacja krzyżowa.
        scores = cross_val_score(
            knn,
            [x.to_list() for x in characteristic_values_list],
            [y.is_sick() for y in characteristic_values_list],
            cv=kfold,
            scoring='accuracy'
        )

        print(f"Wyniki walidacji krzyżowej: {scores}")
        print(f"Średnia dokładność: {np.mean(scores):.2f}")
        print(f"Odchylenie standardowe: {np.std(scores):.2f}")
