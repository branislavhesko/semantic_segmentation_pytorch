import enum


class State(enum.Enum):
    train = enum.auto()
    val = enum.auto()
    test = enum.auto()
