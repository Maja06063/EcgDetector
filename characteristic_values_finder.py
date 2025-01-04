from common_finder import CommonFinder

from characteristic_values import CharacteristicValues

import logging

from health_table import HEALTH_IDS_LIST
from sklearn.neural_network import MLPClassifier
from scipy.signal import find_peaks, peak_widths

import numpy as np
import matplotlib.pyplot as plt

class CharacteristicValuesFinder(CommonFinder):

    def __init__(self):
        self.nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)

    def fit(self, x_train, y_train):

        parts = []
        labels = []

        # Podział sygnału na okresy
        peaks, _ = find_peaks(x_train, prominence=1)
        periods = [x_train[peaks[i]:peaks[i+1]] for i in range(len(peaks)-1)]

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

            # Wizualizacja sygnału i wykrytych peaków
            #plt.plot(period, label="Sygnał")
            #plt.plot(time[peaks], voltage_signal[peaks], "rx", label="Peaki")
            #plt.legend()
            #plt.xlabel("Czas")
            #plt.ylabel("Amplituda")
            #plt.title("Wykrywanie peaków w sygnale okresowym")
            #plt.show()

        self.nn.fit(parts, labels)

    def predict(self, x_test) -> list:

        characteristic_values_list = []

        for signal_id, voltage_signal in x_test.items():

            isHealth = signal_id in HEALTH_IDS_LIST
            #print(isHealth)

            mean_value = np.mean(voltage_signal)
            voltage_signal -= mean_value

            # Obliczanie FFT
            N = len(voltage_signal)
            fft_values = np.fft.fft(voltage_signal)
            fft_freqs = np.fft.fftfreq(N, d=1/self.SAMPLING_FREQ)

            # Wartości FFT dla pozytywnych częstotliwości
            positive_freqs = fft_freqs[:N // 2]
            positive_fft_values = np.abs(fft_values[:N // 2])

            # Wygładzenie
            window_size = 20
            positive_fft_values = np.convolve(positive_fft_values, np.ones(window_size)/window_size, mode='same')

            # Filtr górnoprzepustowy (wycięcie częstotliwości)
            lower_threshold_Hz = 0.5
            positive_fft_values[np.abs(positive_freqs) < lower_threshold_Hz] *= 0.1

            # Wycięcie 30 i 60 Hz (szumy zasilania)
            positive_fft_values[np.abs(positive_freqs - 60) < 0.1] *= 0.1

            # Normalizacja FFT
            max_fft = np.max(positive_fft_values)
            positive_fft_values = np.divide(positive_fft_values, max_fft)

            """
            # Rysowanie wykresu
            plt.figure(figsize=(12, 6))

            # Wykres funkcji napięcia od czasu
            plt.subplot(2, 1, 1)
            time = np.linspace(0, N / self.SAMPLING_FREQ, N)
            plt.plot(time, voltage_signal)
            plt.title(f"{signal_id}: Voltage Signal")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage")

            # Wykres FFT
            plt.subplot(2, 1, 2)
            plt.plot(positive_freqs, positive_fft_values)
            plt.title(f"{signal_id}: FFT")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            """

            # Znalezienie największego piku
            peak_index = np.argmax(positive_fft_values)  # Indeks największego piku
            peak_frequency = positive_freqs[peak_index]  # Częstotliwość największego piku

            # Szerokość pasma
            band_threshold = 0.1
            bandwidth_indices = np.where(positive_fft_values >= band_threshold)[0]
            bandwidth_start = positive_freqs[bandwidth_indices[0]]  # Początek pasma
            bandwidth_end = positive_freqs[bandwidth_indices[-1]]   # Koniec pasma
            bandwidth = bandwidth_end - bandwidth_start  # Szerokość pasma

            #print(peak_frequency)
            #print(bandwidth)

            patient_values = CharacteristicValues(signal_id)
            patient_values.sick = not isHealth
            patient_values.F_max = peak_frequency
            patient_values.F_width = bandwidth

            """
            # Wyświetlanie
            plt.tight_layout()
            plt.show()
            """

            time = np.linspace(0, N / self.SAMPLING_FREQ, N)

            parts = []
            labels = []
            np.set_printoptions(precision=3, suppress=True)

            # Podział sygnału na okresy
            peaks, _ = find_peaks(voltage_signal, prominence=1)
            periods = [voltage_signal[peaks[i]:peaks[i+1]] for i in range(len(peaks)-1)]
            periods_num = len(periods)

            for period in periods:

                period_len = len(period)
                if period_len < self.PART_LEN:
                    continue

                parts = []
                for i in range(0, period_len, self.PART_LEN):

                    parts.append((np.pad(
                        period[i:i+self.PART_LEN],
                        (0, self.PART_LEN - len(period[i:i+self.PART_LEN])),
                        'constant')
                    ))

                res = self.nn.predict_proba(parts)
                predicted_parts = np.argmax(res, axis=0)

                #print(res)
                #print(predicted_parts)

                # Znajdź amplitudę oraz czas trwania części:
                # Przykładowe dane pomiarowe

                # Znajdź peaki i ich właściwości
                qrs_part_begin = predicted_parts[0] * self.PART_LEN

                this_part = np.pad(
                    period[qrs_part_begin:qrs_part_begin + self.PART_LEN],
                    (0, self.PART_LEN - len(period[qrs_part_begin:qrs_part_begin + self.PART_LEN])),
                    'constant'
                )

                peaks, properties = find_peaks(this_part, height=-1)  # height=0 znajdzie wszystkie peaki
                heights = properties['peak_heights']  # Wysokości peaków

                #print(heights)
                # Znajdź indeks największego peak'u
                try:
                    max_peak_idx = np.argmax(heights)
                    max_peak = peaks[max_peak_idx]
                    max_peak_height = heights[max_peak_idx]

                    # Oblicz szerokość największego peak'u
                    widths, height, left_ips, right_ips = peak_widths(this_part, [max_peak], rel_height=0.7)
                    max_peak_width = widths[0]

                    # Wyniki
                    #print(f"Lokalizacja największego peak'u: {time[max_peak]}")
                    #print(f"Wysokość największego peak'u: {max_peak_height}")
                    #print(f"Szerokość największego peak'u (w jednostkach czasu): {max_peak_width}")
                    patient_values.A_QRS += max_peak_height
                    patient_values.T_QRS += max_peak_width * 2

                except(ValueError):
                    pass

                t_part_begin = predicted_parts[1] * self.PART_LEN

                this_part = np.pad(
                    period[t_part_begin:t_part_begin + self.PART_LEN],
                    (0, self.PART_LEN - len(period[t_part_begin:t_part_begin + self.PART_LEN])),
                    'constant'
                )

                peaks, properties = find_peaks(this_part, height=-1)  # height=0 znajdzie wszystkie peaki
                heights = properties['peak_heights']  # Wysokości peaków

                # Znajdź indeks największego peak'u
                try:
                    max_peak_idx = np.argmax(heights)
                    max_peak = peaks[max_peak_idx]
                    max_peak_height = heights[max_peak_idx]

                    # Oblicz szerokość największego peak'u
                    widths, height, left_ips, right_ips = peak_widths(this_part, [max_peak], rel_height=0.7)
                    max_peak_width = widths[0]

                    # Wyniki
                    #print(f"Lokalizacja największego peak'u: {time[max_peak]}")
                    #print(f"Wysokość największego peak'u: {max_peak_height}")
                    #print(f"Szerokość największego peak'u (w jednostkach czasu): {max_peak_width}")
                    patient_values.A_T += max_peak_height
                    patient_values.T_T += max_peak_width

                except(ValueError):
                    pass

                p_part_begin = predicted_parts[2] * self.PART_LEN

                this_part = np.pad(
                    period[p_part_begin:p_part_begin + self.PART_LEN],
                    (0, self.PART_LEN - len(period[p_part_begin:p_part_begin + self.PART_LEN])),
                    'constant'
                )

                peaks, properties = find_peaks(this_part, height=-1)  # height=0 znajdzie wszystkie peaki
                heights = properties['peak_heights']  # Wysokości peaków

                # Znajdź indeks największego peak'u
                try:
                    max_peak_idx = np.argmax(heights)
                    max_peak = peaks[max_peak_idx]
                    max_peak_height = heights[max_peak_idx]

                    # Oblicz szerokość największego peak'u
                    widths, height, left_ips, right_ips = peak_widths(this_part, [max_peak], rel_height=0.7)
                    max_peak_width = widths[0]

                    # Wyniki
                    #print(f"Lokalizacja największego peak'u: {time[max_peak]}")
                    #print(f"Wysokość największego peak'u: {max_peak_height}")
                    #print(f"Szerokość największego peak'u (w jednostkach czasu): {max_peak_width}")
                    patient_values.A_P += max_peak_height
                    patient_values.T_P += max_peak_width

                except(ValueError):
                    pass

                """
                # Wizualizacja
                plt.hlines(height, left_ips, right_ips, color="C1", label="Szerokość największego peak'u")
                plt.plot(self.get_time_list_for_value_list(this_part), this_part, label="Dane pomiarowe")
                plt.plot(this_part[max_peak], "x", label="Największy peak", color="red")
                plt.legend()
                plt.xlabel("Czas")
                plt.ylabel("Amplituda")
                plt.show()
                """
            patient_values.A_P /= periods_num
            patient_values.T_P /= periods_num
            patient_values.A_QRS /= periods_num
            patient_values.T_QRS /= periods_num
            patient_values.A_T /= periods_num
            patient_values.T_T /= periods_num

            characteristic_values_list.append(patient_values)
            logging.debug(characteristic_values_list[-1].to_string())

        return characteristic_values_list
