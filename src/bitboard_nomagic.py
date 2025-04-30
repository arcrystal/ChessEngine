import numpy as np
from numba import njit

# === Lookup Tables ===
KNIGHT_ATTACKS = np.zeros(64, dtype=np.uint64)
KING_ATTACKS = np.zeros(64, dtype=np.uint64)
WHITE_PAWN_ATTACKS = np.zeros(64, dtype=np.uint64)
BLACK_PAWN_ATTACKS = np.zeros(64, dtype=np.uint64)

# === Basic Helpers ===
@njit
def square_mask(sq):
    return int(np.uint64(1) << np.uint64(sq))


# === Attack Access Functions ===
@njit
def knight_attacks(square):
    return int(KNIGHT_ATTACKS[square])

@njit
def king_attacks(square):
    return int(KING_ATTACKS[square])

@njit
def pawn_attacks(square, is_white):
    return int(WHITE_PAWN_ATTACKS[square] if is_white else BLACK_PAWN_ATTACKS[square])

# === Precompute Static Tables ===
def precompute_lookup_tables(n_attacks, k_attacks, white_p_attacks, black_p_attacks):
    KNIGHT_DELTAS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
    KING_DELTAS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    
    for sq in range(64):
        r = sq // 8
        f = sq % 8

        # Knight attacks
        knight_attacks = 0
        for dr, df in KNIGHT_DELTAS:
            nr, nc = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nc < 8:
                knight_attacks |= square_mask(nr * 8 + nc)
        n_attacks[sq] = knight_attacks

        # King attacks
        king_attacks = 0
        for dr, df in KING_DELTAS:
            nr, nc = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nc < 8:
                king_attacks |= square_mask(nr * 8 + nc)
        k_attacks[sq] = king_attacks

        # Pawn attacks
        # Pawn attacks
        wp_attack = 0
        bp_attack = 0

        if r < 7:
            if f > 0:
                wp_attack |= square_mask((r + 1) * 8 + (f - 1))
            if f < 7:
                wp_attack |= square_mask((r + 1) * 8 + (f + 1))
        if r > 0:
            if f > 0:
                bp_attack |= square_mask((r - 1) * 8 + (f - 1))
            if f < 7:
                bp_attack |= square_mask((r - 1) * 8 + (f + 1))

        WHITE_PAWN_ATTACKS[sq] = wp_attack
        BLACK_PAWN_ATTACKS[sq] = bp_attack
        

precompute_lookup_tables(KNIGHT_ATTACKS, KING_ATTACKS, WHITE_PAWN_ATTACKS, BLACK_PAWN_ATTACKS)
