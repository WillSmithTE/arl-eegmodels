# good channels https://www.nature.com/articles/s41598-020-70569-y
# Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz

# good channels per YK
# Pz, P7, P8, Oz, O1, and O2
# 27, 29, 25, 31, 32, 30
import numpy as np

# namesToTake = np.array([1, 2, 5, 6, 7, 8, 9, 16, 17, 18, 26, 27, 28, 30, 32])
namesToTake = np.array([27, 29, 25, 31, 32, 30])

indexesToTake = namesToTake - 1

def takeOnlyCertainChannels(data):
    return data[indexesToTake]

#channels
# Channel name	take
# FP2'	1      y
# FP1'	2 y
# AF4'	3 n
# AF3'	4 n
# F8'	5 y
# F4'	6 y
# FZ'	7 y
# F3'	8 y
# F7'	9 y
# FT8'	10 n
# FC4'	11 n
# FCZ'	12 n
# FC3'	13 n
# FT7'	14 n
# T8'	15 n
# C4'	16 y
# CZ'	17 y
# C3'	18 y
# T7'	19 n
# TP8'	20 n
# CP4'	21 n
# CPZ'	22 n
# CP3'	23 n
# TP7'	24 n
# P8'	25 n
# P4'	26 y
# PZ'	27 y
# P3'	28 y
# P7'	29 n
# O2'	30 y
# OZ'	31 n
# O1'	32 y

