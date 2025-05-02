import numpy as np
from numba import uint64, njit


@njit
def pop_lsb(bb) -> tuple:
    if bb == np.uint64(0):
        return -1, np.uint64(0)

    # Safely compute lsb using uint64 mask
    lsb = np.bitwise_and(bb, np.uint64(-int(bb)))
    index = np.uint64(0)
    temp = lsb

    while temp > 1:
        temp = temp >> 1
        index += 1

    new_bb = np.bitwise_and(bb, bb - np.uint64(1))
    return int(index), new_bb

@njit
def rank_mask(rank):
    return uint64(0xFF) << uint64(rank * 8)

@njit
def file_mask(file_idx):
    return uint64(0x0101010101010101) << uint64(file_idx)

@njit
def square_mask(sq: int) -> np.uint64:
    return uint64(1) << uint64(sq)