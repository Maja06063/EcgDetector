class CharacteristicValues:

    def __init__(self, patient_id: int) -> None:

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

        return self.sick
