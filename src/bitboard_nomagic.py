import numpy as np
from numba import njit, uint64
from src.bitboard_utils import square_mask

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
def pawn_attacks(square: int, white_to_move: bool) -> np.uint64:
    return WHITE_PAWN_ATTACKS[square] if white_to_move else BLACK_PAWN_ATTACKS[square]

@njit
def precompute_lookup_tables(knight_attacks, king_attacks, wp_attacks, bp_attacks):
    knight_deltas = np.array([
        [2, 1], [1, 2], [-1, 2], [-2, 1],
        [-2, -1], [-1, -2], [1, -2], [2, -1]
    ], dtype=np.int8)

    king_deltas = np.array([
        [1, 0], [1, 1], [0, 1], [-1, 1],
        [-1, 0], [-1, -1], [0, -1], [1, -1]
    ], dtype=np.int8)

    for sq in range(64):
        r = sq // 8
        f = sq % 8

        k_att = uint64(0)
        g_att = uint64(0)
        for i in range(knight_deltas.shape[0]):
            nr = r + knight_deltas[i, 0]
            nf = f + knight_deltas[i, 1]
            if 0 <= nr < 8 and 0 <= nf < 8:
                k_att |= square_mask(nr * 8 + nf)
        knight_attacks[sq] = k_att

        for i in range(king_deltas.shape[0]):
            nr = r + king_deltas[i, 0]
            nf = f + king_deltas[i, 1]
            if 0 <= nr < 8 and 0 <= nf < 8:
                g_att |= square_mask(nr * 8 + nf)
        king_attacks[sq] = g_att

        wp = uint64(0)
        bp = uint64(0)
        if r < 7:
            if f > 0: wp |= square_mask((r + 1) * 8 + (f - 1))
            if f < 7: wp |= square_mask((r + 1) * 8 + (f + 1))
        if r > 0:
            if f > 0: bp |= square_mask((r - 1) * 8 + (f - 1))
            if f < 7: bp |= square_mask((r - 1) * 8 + (f + 1))

        wp_attacks[sq] = wp
        bp_attacks[sq] = bp

precompute_lookup_tables(KNIGHT_ATTACKS, KING_ATTACKS, WHITE_PAWN_ATTACKS, BLACK_PAWN_ATTACKS)