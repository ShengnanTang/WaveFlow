
from enum import Enum, auto

# 1. 在 Python 3.10 中手动定义 StrEnum 基类
class StrEnum(str, Enum):
    # 2. 覆盖默认的 auto() 行为
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        # 让 auto() 返回成员名的小写形式
        return name.lower()


class Setting(StrEnum):
    UNIVARIATE = auto()
    MULTIVARIATE = auto()


class Prior(StrEnum):
    SN = auto()  # 值为 'sn'
    ISO = auto() # 值为 'iso'
    OU = auto()  # 值为 'ou'
    SE = auto()
    PE = auto()


season_lengths = {
    "min":24,
    "H": 24,
    "D": 30,
    "1D": 30,
    "B": 30,
    "W": 36,
}

season_lengths_gluonts = {
    "S": 3600,  # 1 hour
    "T": 1440,  # 1 day
    "H": 24,  # 1 day
    "1D": 1,
    "D": 1,  # 1 day
    "W": 1,  # 1 week
    "M": 12,
    "B": 5,
    "Q": 4,
    "Y": 1,
}


def get_season_length(freq):
    return season_lengths[freq]


def get_lags_for_freq(freq_str: str):
    if freq_str == "H":
        lags_seq = [24 * i for i in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]]
    elif freq_str == "B":
        # TODO: Fix lags for B
        lags_seq = [30 * i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    elif freq_str == "1D":
        lags_seq = [30 * i for i in [1, 2, 3, 4, 5, 6, 7]]
    elif freq_str == "W":
        lags_seq = [1, 2, 3, 4, 13, 52]
    else:
        raise NotImplementedError(f"Lags for {freq_str} are not implemented yet.")
    return lags_seq
