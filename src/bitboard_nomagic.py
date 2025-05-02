import numpy as np
from numba import njit
from bitboard_utils import square_mask

KNIGHT_ATTACKS = np.zeros(64, dtype=np.uint64)
KING_ATTACKS = np.zeros(64, dtype=np.uint64)
WHITE_PAWN_ATTACKS = np.zeros(64, dtype=np.uint64)
BLACK_PAWN_ATTACKS = np.zeros(64, dtype=np.uint64)

@njit
def knight_attacks(square: int) -> np.uint64:
    return KNIGHT_ATTACKS[square]

@njit
def king_attacks(square: int) -> np.uint64:
    return KING_ATTACKS[square]

@njit
def pawn_attacks(square: int, is_white: bool) -> np.uint64:
    return WHITE_PAWN_ATTACKS[square] if is_white else BLACK_PAWN_ATTACKS[square]

def precompute_lookup_tables():
    knight_deltas = np.array([
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ])
    king_deltas = np.array([
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1)
    ])

    for sq in range(64):
        r, f = divmod(sq, 8)

        # Knight
        attacks = np.uint64(0)
        for dr, df in knight_deltas:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                attacks |= square_mask(nr * 8 + nf)
        KNIGHT_ATTACKS[sq] = attacks

        # King
        attacks = np.uint64(0)
        for dr, df in king_deltas:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                attacks |= square_mask(nr * 8 + nf)
        KING_ATTACKS[sq] = attacks

        # White pawn
        wp = np.uint64(0)
        if r < 7:
            if f > 0:
                wp |= square_mask((r + 1) * 8 + (f - 1))
            if f < 7:
                wp |= square_mask((r + 1) * 8 + (f + 1))
        WHITE_PAWN_ATTACKS[sq] = wp

        # Black pawn
        bp = np.uint64(0)
        if r > 0:
            if f > 0:
                bp |= square_mask((r - 1) * 8 + (f - 1))
            if f < 7:
                bp |= square_mask((r - 1) * 8 + (f + 1))
        BLACK_PAWN_ATTACKS[sq] = bp