# Why Do We have BITs here?

# Bits is a clone of https://github.com/google/aqt

# AQT library by google, but I needed to be able to change a lot of parts of the code and I couldn't do this on Google
# Repo for sure so I just made a copy of the part of the library that I wanted (AQT is Apache 2 Licenced Project)
# Ill more focus on doing the job for the llm checkpoint in 8 bit so there's no need to project name be AQT
# Accurate QTraining

from .bits import (
    matmul_true_int8 as matmul_true_int8,
    matmul as matmul,
    q_matmul_int8 as q_matmul_int8
)
from .q_dot_general import (
    make_dot_general as make_dot_general,
    make_fake_quant as make_fake_quant,
    DotGeneralRes as DotGeneralRes
)
from .q_flax import (
    QuantMode as QuantMode,
    Freezer as Freezer,
    QDotGeneral as QDotGeneral,
    QEinsum as QEinsum,
    config_v4 as config_v4
)

from .config import fully_quantized

from .qk import (
    quantize_kv as quantize_kv,
    MAX_INT8 as MAX_INT8,
    unquantize_kv as unquantize_kv
)

__all__ = (
    "matmul_true_int8",
    "matmul",
    "q_matmul_int8",
    "make_dot_general",
    "make_fake_quant",
    "DotGeneralRes",
    "QuantMode",
    "Freezer",
    "QDotGeneral",
    "QEinsum",
    "config_v4",
    "fully_quantized",
    "quantize_kv",
    "unquantize_kv"
)
