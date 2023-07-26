class Base:
    file_path: str


# SPECIFIED RELATIVE TO ./torch/algorithms
class sg(Base):
    file_path = "cfpi.single_gaussian"


class mg(Base):
    file_path = "cfpi.mixture_gaussian"


class bc(Base):
    file_path = "bc"


class sarsa_iqn(Base):
    file_path = "sarsa_iqn"

