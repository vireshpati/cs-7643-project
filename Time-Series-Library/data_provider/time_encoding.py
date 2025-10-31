from enum import Enum, unique


@unique
class TimeEncoding(Enum):
    """Modes for constructing temporal marker tensors."""

    NONE = 0
    TIME_FEATURES = 1  # GluonTS-style continuous features
    ABSOLUTE = 2  # absolute elapsed timestamps for irregular sampling


def normalize_time_encoding(value):
    """Convert raw values to TimeEncoding, accepting existing enums or ints."""
    if isinstance(value, TimeEncoding):
        return value
    return TimeEncoding(value)
