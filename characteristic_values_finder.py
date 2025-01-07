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
        self.nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)# budowa sieci neuronowej
        self.ifNormalize = _ifNormalize #czy_normalizacja
        self.plot_number = _plot_number #nr wykresu

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

        parts = [] # Tutaj trafią podzielone odcinki EKG pacjenta.
        labels = [] # Tutaj trafią etykiety (klasy) tych odcinków.

        # Podział sygnału na okresy
        peaks, _ = find_peaks(x_train, prominence=1) # peaks zawiera granicę okresów ZNALEZIENIE PEAKÓW
        #parametr prominence aby wykluczać szumy, zeby nie byly brane jako peaky
        periods = [x_train[peaks[i]:peaks[i+1]] for i in range(len(peaks)-1)] #DZIELENIE NA OKRESY
        # periods to lista list. Każdy jej element to jeden okres (a okres to lista danych
        # liczbowych).

        # WYKRES NR 2 - Wykryte peaki:
        if self.plot_number == 2:
            self.peak_plot(x_train, peaks)

        # Dalsze przetwarzanie robimy osobno dla każdego okresu:
        for period in periods:

            # Pomijamy dziwnie krótkie okresy (nie znaczą nic dla nas):
            period_len = len(period)
            if period_len < self.PART_LEN:
                continue

            # Ustalamy odgórnie, jaka część okresu EKG jest jakim rodzajem odcinka - uczenie
            # nadzorowane.
            qrs_part = period[0:int(0.05*period_len)]
            t_part = period[int(0.05*period_len):int(0.4*period_len)]
            p_part = period[int(0.67*period_len):int(0.85*period_len)]

            # Dodajemy do parts zinterpolowany(zeskalowany) odcinek danych (aby jego długość była zawsze taka
            # sama). Do labels dodajemy informację, że jest to odcinek QRS.
            parts.append(np.interp(range(self.PART_LEN), range(len(qrs_part)), qrs_part))
            labels.append(self.QRS)

            # Dla odcinka T analogicznie jak dla QRS:
            parts.append(np.interp(range(self.PART_LEN), range(len(t_part)), t_part))
            labels.append(self.T)

            # Dla odcinka P analogicznie jak dla QRS:
            parts.append(np.interp(range(self.PART_LEN), range(len(p_part)), p_part))
            labels.append(self.P)

        # Na koniec odcinki oraz etykiety podajemy sieci neuronowej do nauczenia się.
        return self.nn.fit(parts, labels)

    def predict(self, x_test: dict) -> list:
        """
        Metoda predict służy do przeprowadzenia ekstrakcji cech z pacjentów.

        Parametry:
        1. x_test - słownik pacjentów (klucz to id, wartość to przebieg EKG).

        Zwraca:
        1. characteristic_values_list - lista wyekstraktowanych cech ze wszytkich pacjentów.
        """
        # Lista, w której zamieścimy wyekstraktowane dane wszystkich pacjentów:
        characteristic_values_list = []

        # Dalsze działania przeprowadzamy osobno dla każdego pacjenta:
        for signal_id, voltage_signal in x_test.items():

            # Tworzymy obiekt przechowywujący wyekstraktowane cechy pacjenta i umieszczamy tam
            # informację, czy jest zdrowy, czy chory (etykietę):
            patient_values = CharacteristicValues(signal_id)
            patient_values.sick = not signal_id in HEALTH_IDS_LIST

            # Usuwamy składową stałą sygnału:
            mean_value = np.mean(voltage_signal)
            voltage_signal -= mean_value

            # Transformacja Fouriera:
            patient_values.F_max, patient_values.F_width = self.calculate_fft(voltage_signal, signal_id)

            # Dzielimy sygnał na okresy.
            peaks, _ = find_peaks(voltage_signal, prominence=1)
            periods = [voltage_signal[peaks[i]:peaks[i+1]] for i in range(len(peaks)-1)]
            periods_num = len(periods)
            # peaks to lista punktów, w których zaczyna się nowy okres.
            # periods to lista list, gdzie każdy element to okres,
            # periods_num to liczba okresów w sygnale.

            # Dalsze działania przeprowadzamy osobno dla każdego okresu:
            for period in periods:

                # Dodajemy zera do za krótkich okresów:
                period_len = len(period)
                if period_len < self.PART_LEN:
                    period = self.add_padding(period, i)

                # Normalizujemy wartości okresu, jeśli użytkownik wybrał takją opcję:
                if self.ifNormalize:

                    # Odejmujemy minimalną wartość tak aby minimum było w 0:
                    min_val = np.min(period)
                    period = np.subtract(period, min_val)

                    # Dzielimy przez maksymalną wartość, by maksimum było w 1:
                    max_val = np.max(period)
                    if max_val != 0:
                        period = np.divide(period, max_val)

                # Dzielimy okres na odcinki. Za krótki odcinek uzupełniamy zerami na końcu:
                parts = []
                for i in range(0, period_len, self.PART_LEN):

                    parts.append(self.add_padding(period, i))

                # Przeprowadzamy predykcję odcinków za pomocą nauczonej wcześniej sieci neuronowej.
                #TUTAJ SIEĆ NEURONOWA OKREŚLA KTÓRY ODCINEK JEST QRS, T I P
                # Używamy predict_proba aby dostać prawdopodobieństwo danej etykiety w danym
                # odcinku EKG. Następnie argmaxem wyłaniamy najbardziej prawdopodobny odcinek dla
                # każdej etykiety.
                res = self.nn.predict_proba(parts)
                predicted_parts = np.argmax(res, axis=0)

                # Szukamy amplitudy i czasu trwania odcinka QRS:
                qrs_part_begin = predicted_parts[0] * self.PART_LEN
                max_peak_height, max_peak_width = self.analyze_peaks(period, qrs_part_begin)

                patient_values.A_QRS += max_peak_height
                patient_values.T_QRS += max_peak_width * 2 # *2, bo od niego zaczyna i kończy się okres.

                # Szukamy amplitudy i czasu trwania odcinka T:
                t_part_begin = predicted_parts[1] * self.PART_LEN
                max_peak_height, max_peak_width = self.analyze_peaks(period, t_part_begin)
                
                patient_values.A_T += max_peak_height
                patient_values.T_T += max_peak_width

                # Szukamy amplitudy i czasu trwania odcinka P:
                p_part_begin = predicted_parts[2] * self.PART_LEN
                max_peak_height, max_peak_width = self.analyze_peaks(period, p_part_begin)

                patient_values.A_P += max_peak_height
                patient_values.T_P += max_peak_width

                # WYKRES NR 3 - Części okresu:
                if self.plot_number == 3:
                    self.peak_plot(period, [qrs_part_begin, t_part_begin, p_part_begin])

            # Po przerobieniu każdego okresu dzielimy sumy przez liczbę okresów aby dostać średnią:
            patient_values.A_P /= periods_num
            patient_values.T_P /= periods_num
            patient_values.A_QRS /= periods_num
            patient_values.T_QRS /= periods_num
            patient_values.A_T /= periods_num
            patient_values.T_T /= periods_num

            # Dodajemy uzupełnione dane wyekstraktowane z pacjenta do listy
            # characteristic_values_list:
            characteristic_values_list.append(patient_values)
            logging.info(characteristic_values_list[-1].to_string())

        # Zwracamy uzupełnioną listę z wyekstraktowanymi danymi wszystkich pacjentów:
        return characteristic_values_list
