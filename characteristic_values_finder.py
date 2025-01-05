from characteristic_values import CharacteristicValues
from common_finder import CommonFinder
from health_table import HEALTH_IDS_LIST
from scipy.signal import find_peaks
from sklearn.neural_network import MLPClassifier

import logging
import matplotlib.pyplot as plt
import numpy as np

class CharacteristicValuesFinder(CommonFinder):
    """
    Klasa CharacteristicValuesFinder to najbardziej podstawowa wersja algorytmu ekstrakcji cech z
    danych EKG. Posiada metody fit i predict zgodne z sklearn.
    """
    def __init__(self, _ifNormalize: bool, _plot_number: int):
        """
        Konstruktor służący do zapisania wartości parametrów oraz zbudowania sieci neuronowej.

        Parametry:
        1. _ifNormalize - czy normalizować odcinki okresów EKG (aby były od 0 do 1),
        2. _plot_number - nr wykresów do wyświetlenia (0 - brak).

        Funkcja nie zwraca żadnych wartości.
        """
        self.nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
        self.ifNormalize = _ifNormalize
        self.plot_number = _plot_number

    def fit(self, x_train: dict, y_train: bool):
        """
        Metoda fit służy do nauki ekstraktora, w jaki sposób ma wyznaczać cechy odcinków EKG
        pacjentów.

        Parametry:
        1. x_train - pacjent treningowy,
        2. y_train - stan zdrowia pacjenta (najlepiej podać zdrowego, z równymi przebiegami).

        Zwraca:
        1. self.nn - nauczona siec neuronowa.
        """
        if not y_train:
            logging.warning("Uczenie ekstrakcji powinno przebiegać na zdrowym pacjencie")

        parts = []
        labels = []

        # Podział sygnału na okresy
        peaks, _ = find_peaks(x_train, prominence=1)
        periods = [x_train[peaks[i]:peaks[i+1]] for i in range(len(peaks)-1)]

        # WYKRES NR 2 - Wykryte peaki:
        if self.plot_number == 2:
            self.peak_plot(x_train, peaks)

        for period in periods:

            period_len = len(period)
            if period_len < self.PART_LEN:
                continue

            qrs_part = period[0:int(0.05*period_len)]
            t_part = period[int(0.05*period_len):int(0.4*period_len)]
            p_part = period[int(0.67*period_len):int(0.85*period_len)]

            parts.append(np.interp(range(self.PART_LEN), range(len(qrs_part)), qrs_part))
            labels.append(self.QRS)

            parts.append(np.interp(range(self.PART_LEN), range(len(t_part)), t_part))
            labels.append(self.T)

            parts.append(np.interp(range(self.PART_LEN), range(len(p_part)), p_part))
            labels.append(self.P)

        return self.nn.fit(parts, labels)

    def predict(self, x_test: dict) -> list:
        """
        Metoda predict służy do przeprowadzenie ekstrakcji cech z pacjentów.

        Parametry:
        1. x_test - słownik pacjentów (klucz to id, wartość to przebieg EKG).

        Zwraca:
        1. characteristic_values_list - lista wyekstraktowanych cech ze wszytkich pacjentów.
        """
        characteristic_values_list = []

        for signal_id, voltage_signal in x_test.items():

            patient_values = CharacteristicValues(signal_id)
            patient_values.sick = not signal_id in HEALTH_IDS_LIST

            # Usuwanie składowej stałej sygnału:
            mean_value = np.mean(voltage_signal)
            voltage_signal -= mean_value

            # FFT:
            patient_values.F_max, patient_values.F_width = self.calculate_fft(voltage_signal, signal_id)

            # Podział sygnału na okresy
            peaks, _ = find_peaks(voltage_signal, prominence=1)
            periods = [voltage_signal[peaks[i]:peaks[i+1]] for i in range(len(peaks)-1)]
            periods_num = len(periods)

            for period in periods:

                period_len = len(period)
                if period_len < self.PART_LEN:
                    period = self.add_padding(period, i)

                # Normalizacja okresu danych:
                if self.ifNormalize:
                    min_val = np.min(period)
                    period = np.subtract(period, min_val)
                    max_val = np.max(period)
                    if max_val != 0:
                        period = np.divide(period, max_val)

                parts = []
                for i in range(0, period_len, self.PART_LEN):

                    parts.append(self.add_padding(period, i))

                res = self.nn.predict_proba(parts)
                predicted_parts = np.argmax(res, axis=0)

                # Znajdź peaki i ich właściwości
                qrs_part_begin = predicted_parts[0] * self.PART_LEN

                max_peak_height, max_peak_width = self.analyze_peaks(period, qrs_part_begin)

                patient_values.A_QRS += max_peak_height
                patient_values.T_QRS += max_peak_width * 2

                t_part_begin = predicted_parts[1] * self.PART_LEN

                max_peak_height, max_peak_width = self.analyze_peaks(period, t_part_begin)
                
                patient_values.A_T += max_peak_height
                patient_values.T_T += max_peak_width

                p_part_begin = predicted_parts[2] * self.PART_LEN

                max_peak_height, max_peak_width = self.analyze_peaks(period, p_part_begin)

                patient_values.A_P += max_peak_height
                patient_values.T_P += max_peak_width

                # WYKRES NR 3 - Części okresu:
                if self.plot_number == 3:
                    self.peak_plot(period, [qrs_part_begin, t_part_begin, p_part_begin])

            patient_values.A_P /= periods_num
            patient_values.T_P /= periods_num
            patient_values.A_QRS /= periods_num
            patient_values.T_QRS /= periods_num
            patient_values.A_T /= periods_num
            patient_values.T_T /= periods_num

            characteristic_values_list.append(patient_values)
            logging.info(characteristic_values_list[-1].to_string())

        return characteristic_values_list
