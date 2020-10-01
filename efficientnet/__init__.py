from enum import Enum
from .model import EfficientNet as EffNet

class Config(Enum):
    # Config valid config name
    B0 = 0
    B1 = 1
    B2 = 2
    B3 = 3
    B4 = 4
    B5 = 5
    B6 = 6
    B7 = 7

_CONFIG_TO_PHI = {
    Config.B0 : 0,
    Config.B1 : 1,
    Config.B2 : 2,
    Config.B3 : 3,
    Config.B4 : 4,
    Config.B5 : 5,
    Config.B6 : 6,
    Config.B7 : 7
}

def EfficientNet(cfg, with_relu=False):
    """Constructs an EfficientNet instance"""
    assert isinstance(cfg, Config)
    return EffNet(phi=_CONFIG_TO_PHI[cfg],
                  with_relu=with_relu)
