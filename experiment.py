from characteristic_images_finder import CharacteristicImagesFinder
from characteristic_values_finder import CharacteristicValuesFinder
from health_table import HEALTH_IDS_LIST
from reference import Reference
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier

import logging
import numpy as np
import wfdb

class Experiment:
    """
    Klasa Experiment zawiera metody służące do wczytania danych z plików bazy MIT-BIH oraz
    uruchomienia protokołu eksperymentu.
    """
    EXPERIMENTAL_RABBIT_ID = 100
    FLAG_IMAGE = False
    FLAG_REFERENCE = False
    FLAG_NORMALIZE = False
    FLAG_K = 0

    def read_dataset(self, records_files: list) -> None:
        """
        Metoda read_dataset służy do wczytania danych z podanych blików bazy w formacie wfdb.

        Parametry:
        1. records_files - lista zawierająca nazwy plików do wczytania.

        Funkcja nie zwraca żadnych wartości.
        """
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
        """
        Metoda run służy do uruchomienia protokołu eksperymentalnego.

        Parametry:
        1. plot_number - numer wykresów, który wyświetlić podczas trwania algorytmów (0 - brak).

        Funkcja nie zwraca żadnych wartości.
        """
        # Lista na wyniki ekstrakcji i klasyfikatory:
        characteristic_values_list = []
        knn = KNeighborsClassifier(n_neighbors=self.FLAG_K)
        cvf = CharacteristicValuesFinder(self.FLAG_NORMALIZE, plot_number)
        cif = CharacteristicImagesFinder(self.FLAG_NORMALIZE, plot_number)
        ref = Reference(self.FLAG_NORMALIZE, plot_number)

        print("\nEkstrakcja cech...\n")

        # Przetwarzanie z obrazu z danych 2-wymiarowych:
        if self.FLAG_IMAGE:
            cif.fit(
                self.dataset[self.EXPERIMENTAL_RABBIT_ID],
                self.EXPERIMENTAL_RABBIT_ID in HEALTH_IDS_LIST
            )
            characteristic_values_list = cif.predict(self.dataset)

        # Algorytm referencyjny:
        elif self.FLAG_REFERENCE:
            ref.fit(
                self.dataset[self.EXPERIMENTAL_RABBIT_ID],
                self.EXPERIMENTAL_RABBIT_ID in HEALTH_IDS_LIST
            )
            characteristic_values_list = ref.predict(self.dataset)

        # Przetwarzanie z danych 1-wymiarowych (algorytm podstawowy:)
        else:
            cvf.fit(
                self.dataset[self.EXPERIMENTAL_RABBIT_ID],
                self.EXPERIMENTAL_RABBIT_ID in HEALTH_IDS_LIST
            )
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

        #TODO T-studenta
