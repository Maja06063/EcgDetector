from scipy.signal import find_peaks, peak_widths

import matplotlib.pyplot as plt
import numpy as np

class CommonFinder:

    QRS = 1
    T = 2
    P = 3
    PART_LEN = 100
    SAMPLING_FREQ = 360

    def add_padding(self, signal, begin_idx) -> list:

        return np.pad(
            signal[begin_idx:begin_idx + self.PART_LEN],
            (0, self.PART_LEN - len(signal[begin_idx:begin_idx + self.PART_LEN])),
            'constant'
        )

    def calculate_fft(self, voltage_signal: list, signal_id: int) -> tuple:
        
        # Obliczanie FFT
        N = len(voltage_signal)
        fft_values = np.fft.fft(voltage_signal)
        fft_freqs = np.fft.fftfreq(N, d=1/self.SAMPLING_FREQ)

        # Wartości FFT dla pozytywnych częstotliwości
        positive_freqs = fft_freqs[:N // 2]
        positive_fft_values = np.abs(fft_values[:N // 2])

        # Wygładzenie wykresu:
        window_size = 20
        positive_fft_values = np.convolve(positive_fft_values, np.ones(window_size)/window_size, mode='same')

        # Filtr górnoprzepustowy (wycięcie częstotliwości do 0.5 Hz - nie interesują nas):
        lower_threshold_Hz = 0.5
        positive_fft_values[np.abs(positive_freqs) < lower_threshold_Hz] *= 0.1

        # Wycięcie 60 Hz (szumy zasilania)
        positive_fft_values[np.abs(positive_freqs - 60) < 0.1] *= 0.1

        # Normalizacja FFT
        max_fft = np.max(positive_fft_values)
        positive_fft_values = np.divide(positive_fft_values, max_fft)

        # WYKRES NR 1 - Sygnał i FFT:
        if self.plot_number == 1:
            self.time_fft_plot(voltage_signal, positive_freqs, positive_fft_values, signal_id)

        # Znalezienie największego piku
        peak_index = np.argmax(positive_fft_values)  # Indeks największego piku
        peak_frequency = positive_freqs[peak_index]  # Częstotliwość największego piku

        # Szerokość pasma
        band_threshold = 0.1
        bandwidth_indices = np.where(positive_fft_values >= band_threshold)[0]
        bandwidth_start = positive_freqs[bandwidth_indices[0]]  # Początek pasma
        bandwidth_end = positive_freqs[bandwidth_indices[-1]]   # Koniec pasma
        bandwidth = bandwidth_end - bandwidth_start  # Szerokość pasma

        return peak_frequency, bandwidth

    def analyze_peaks(self, period: list, part_begin_idx: int) -> tuple:

        this_part = self.add_padding(period, part_begin_idx)

        peaks, properties = find_peaks(this_part, height=-1)  # height=0 znajdzie wszystkie peaki
        heights = properties['peak_heights']  # Wysokości peaków

        #print(heights)
        # Znajdź indeks największego peak'u
        try:
            max_peak_idx = np.argmax(heights)
            max_peak = peaks[max_peak_idx]
            max_peak_height = heights[max_peak_idx]

            # Oblicz szerokość największego peak'u
            widths, _, _, _ = peak_widths(this_part, [max_peak], rel_height=0.7)
            max_peak_width = widths[0]

            return max_peak_height, max_peak_width

        except(ValueError):
            
            return 0, 0

    def peak_plot(self, signal, peaks):

        plt.vlines(peaks, -1, 1, colors="r", label="Peaki", alpha=0.5)
        plt.plot(signal, label="Sygnał")
        plt.legend()
        plt.xlabel("Próbki")
        plt.ylabel("Amplituda")
        plt.show()

    def time_fft_plot(self, signal, frequencies, fft_values, signal_id):

        plt.figure(figsize=(12, 6))
            
        plt.subplot(2, 1, 1)
        plt.title(f"Pacjent {signal_id}")
        plt.plot(self.get_time_list_for_value_list(signal), signal)
        plt.xlabel("Czas [s]")
        plt.ylabel("Napięcie [mV]")

        plt.subplot(2, 1, 2)
        plt.plot(frequencies, fft_values)
        plt.xlabel("Częstotliwość [Hz]")
        plt.ylabel("Amplituda")
        
        plt.show()

    def get_time_list_for_value_list(self, values_list: list) -> list:

        return list(np.arange(0, len(values_list)) / self.SAMPLING_FREQ)
