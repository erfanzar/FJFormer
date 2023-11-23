# Why Do We have BITs here?

# Bits is a clone of https://github.com/google/aqt

# AQT library by google, but I needed to be able to change a lot of parts of the code and I couldn't do this on Google
# Repo for sure so I just made a copy of the part of the library that I wanted (AQT is Apache 2 Licenced Project)
# Ill more focus on doing the job for the llm load in 8 bit so there's no need to project name be AQT Accurate QTraining

from .bits import matmul_true_int8, matmul, q_matmul_int8
from .q_dot_general import make_dot_general, make_fake_quant, DotGeneralRes
from .q_flax import QuantMode, Freezer, QDotGeneral, QEinsum
