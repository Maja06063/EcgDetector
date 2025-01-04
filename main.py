from experiment import Experiment

import argparse
import logging

import glob
import numpy as np

if __name__ == "__main__":

    experiment = Experiment()

    # Parametry wywołania:
    parser = argparse.ArgumentParser(description="klasyfikacja EKG pacjentów z bazy MIT-BIH.")
    parser.add_argument('-i', '--image', action='store_true', help="użyj ekstrakcji cech z obrazu zamiast danych od czasu.")
    parser.add_argument('-n', '--normalize', action='store_true', help="normalizuj wartości na wykresach.")
    parser.add_argument('-k', '--knn', type=int, default=5, help="parametr k dla algorytmu k-NN.")
    parser.add_argument('-v', '--verbose', action='store_true', help="więcej komunikatów tekstowych.")
    parser.add_argument('-p', '--plot', type=int, default=0, help="wyświetl wykresy o danym nr.")
    args = parser.parse_args()

    records_files = glob.glob("./dataset/*.hea")

    experiment.FLAG_IMAGE = args.image
    experiment.FLAG_NORMALIZE = args.normalize
    experiment.FLAG_K = args.knn
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    experiment.read_dataset(records_files)
    experiment.run(args.plot)
