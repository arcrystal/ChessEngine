
from numba import njit
import numpy as np
from src.bitboard_magic_tables import (
    BISHOP_MASKS, BISHOP_MAGICS, BISHOP_SHIFTS,
    ROOK_MASKS, ROOK_MAGICS, ROOK_SHIFTS,
    bishop_attack_table, rook_attack_table,
)

@njit
def bishop_attacks(square: int, occupancy: np.uint64) -> np.uint64:
    occ = occupancy & BISHOP_MASKS[square]
    index = (occ * BISHOP_MAGICS[square]) >> BISHOP_SHIFTS[square]
    return bishop_attack_table[square, index]

@njit
def rook_attacks(square: int, occupancy: np.uint64) -> np.uint64:
    occ = occupancy & ROOK_MASKS[square]
    index = (occ * ROOK_MAGICS[square]) >> ROOK_SHIFTS[square]
    return rook_attack_table[square, index]

@njit
def queen_attacks(square: int, occupancy: np.uint64) -> np.uint64:
    return bishop_attacks(square, occupancy) | rook_attacks(square, occupancy)
