class CharacteristicValues:
    """
    Klasa CharacteristicValues agreguje dane wyektraktowane z przebiegu EKG jednego pacjenta.
    """
    def __init__(self, patient_id: int) -> None:
        """
        Konstruktor służący do inicjalizacji danych pacjenta o danym id.

        Parametry:
        1. patient_id - id pacjenta.

        Funkcja nie zwraca żadnych wartości.
        """
        self.id         = patient_id
        self.A_P        = 0.0
        self.A_QRS      = 0.0
        self.A_T        = 0.0
        self.T_P        = 0.0
        self.T_QRS      = 0.0
        self.T_T        = 0.0
        self.F_max      = 0.0
        self.F_width    = 0.0
        self.sick       = False

    def to_list(self) -> list:
        """
        Funkcja to_list służy do zwrócenia wszystkich istotnych danych podczas klasyfikacji w
        formie listy.

        Funkcja nie bierze żadnych parametrów.

        Zwraca:
        1. Lista zawierająca dane istotne do klasyfikacji (cechy).
        """
        return [
            self.A_P,
            self.A_QRS,
            self.A_T,
            self.T_P,
            self.T_QRS,
            self.T_T,
            self.F_max,
            self.F_width
        ]

    def is_sick(self) -> bool:
        """
        Metoda is_sick służy do weryfikacji, czy pacjent jest zdrowy czy chory.

        Funkcja nie bierze żadnych parametrów.

        Zwraca:
        - True - jeśli pacjent chory,
        - False - jeśli pacjent zdrowy.
        """
        return self.sick

    def to_string(self) -> str:
        """
        Metoda to_string służy do konwersji danych wyekstraktowanych z pacjenta na string w celu
        ich wyświetlenia.

        Funkcja nie bierze żadnych parametrów.

        Zwraca:
        1. String zawierający informacje wyektraktowane z pacjenta w formie czytelnej dla
        człowieka.
        """
        return f"""
        ******************************
        Pacjent nr {self.id}
        A_P: {self.A_P:.3f},\tA_QRS: {self.A_QRS:.3f},\tA_T: {self.A_T:.3f},
        T_P: {self.T_P:.3f},\tT_QRS: {self.T_QRS:.3f},\tT_T: {self.T_T:.3f},
        F: {self.F_max:.3f},\tF_width: {self.F_width:.3f},
        Pacjent w rzeczywistości chory? {self.sick}.
        """
