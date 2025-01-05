from scipy.stats import ttest_ind

import numpy as np

def load_scores(file_name: str) -> np.array:
    """
    Funkcja load_scores służy do wczytania wartości z plików *.t_student.

    Parametry:
    1. file_name - nazwa pliku do wczytania.

    Zwraca:
    1. scores - wartości wyników foldów walidacji krzyżowej odczytane z plików.
    """
    try:
        with open(file_name, 'r') as f:
            scores = [float(line.strip()) for line in f]
        return np.array(scores)
    except FileNotFoundError:
        print(f"Plik {file_name} nie został znaleziony!")
        return None
    except ValueError:
        print(f"Plik {file_name} zawiera nieprawidłowe dane!")
        return None

if __name__ == "__main__":
    """
    Dodatkowy program do przeprowadzenia statystycznego testu T_studenta.
    """
    # Wczytujemmy dane z plików:
    files = ["normal.t_student", "image.t_student", "reference.t_student"]
    data = {}
    for file in files:
        scores = load_scores(file)
        if scores is not None:
            data[file] = scores

    # Sprawdzamy, czy wszystkie pliki zostały poprawnie wczytane:
    if len(data) != 3:
        raise IOError
    else:
        # Wypisujemy średnie i odchylenia standardowe:
        for name, scores in data.items():
            print(f"{name}:\tŚrednia={scores.mean():.2f},\tOdchylenie standardowe={scores.std():.2f}")

        # Przeprowadzamy test t-Studenta dla każdej pary:
        keys = list(data.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                alg1, alg2 = keys[i], keys[j]
                t_stat, p_val = ttest_ind(data[alg1], data[alg2])
                print(f"\nT-test między {alg1} a {alg2}: t-stat={t_stat:.2f}, p-value={p_val:.2f}")
                if p_val < 0.05:
                    print(f"\t-> Statystycznie istotna różnica między {alg1} a {alg2}")
                else:
                    print(f"\t-> Brak statystycznie istotnej różnicy między {alg1} a {alg2}")
