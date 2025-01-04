from experiment import Experiment

import glob
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    experiment = Experiment()

    records_files = glob.glob("./dataset/*.hea")
    experiment.read_dataset(records_files)

    """
    # Rysowanie wykresu
    plt.figure(figsize=(12, 6))
    #for i, channel_name in enumerate(channel_names[:2]):  # Rysujemy dwa pierwsze kanały
    plt.plot(time, signals[:, 0])

    # Dostosowanie wykresu
    plt.title("Sygnały z PhysioNet")
    plt.xlabel("Czas (s)")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Wyświetlenie wykresu
    plt.show()
    """

    experiment.run()
