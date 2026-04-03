from enum import Enum

class QuantityCalcType(Enum):
    NONE = 0
    REMAINING_INCLUDE_TERMINATE = 1
    REMAINING_EXCLUDE_TERMINATE = 2
    FULL = 3


class PayoffType(Enum):
    FIX = 1,
    FLOAT = 2,
    NONE = 3


class SettlementType(Enum):
    AT_OBSERVATION = 1
    AT_MATURITY = 2
    AT_EARLY_TERMINATION = 3


class BarrierType(Enum):
    UP_OUT = 1,
    DOWN_OUT = 2


class AccumulatorType(Enum):
    Accumulator = 1
    Deccumulator = 2
