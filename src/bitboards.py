import numpy as np
from numba import njit, uint64

# === CONSTANTS ===

# Pseudo-magic numbers (expand if needed)
ROOK_MAGICS = np.array([
    0x8A80104000800020, 0x140002000100040, 0x2801880A0017001, 0x100081001000420,
    0x200020010080420, 0x3001C0002010008, 0x8480008002000100, 0x2080088004402900,
], dtype=np.uint64)

BISHOP_MAGICS = np.array([
    0x40201008040200, 0x20080040020010, 0x8040201008040, 0x8040201008040,
], dtype=np.uint64)

# === Lookup Tables ===
KNIGHT_ATTACKS = np.zeros(64, dtype=np.uint64)
KING_ATTACKS = np.zeros(64, dtype=np.uint64)
WHITE_PAWN_ATTACKS = np.zeros(64, dtype=np.uint64)
BLACK_PAWN_ATTACKS = np.zeros(64, dtype=np.uint64)

ROOK_MASKS = np.zeros(64, dtype=np.uint64)
BISHOP_MASKS = np.zeros(64, dtype=np.uint64)

BISHOP_ATTACKS = np.zeros((64, 512), dtype=np.uint64)
ROOK_ATTACKS = np.zeros((64, 4096), dtype=np.uint64)

# === Basic Helpers ===
@njit
def square_mask(sq):
    return np.uint64(1) << np.uint64(sq)

@njit
def rank_mask(rank):
    return np.uint64(0xFF) << np.uint64(rank * 8)

@njit
def popcount(x):
    """Fast bit count."""
    count = 0
    while x:
        count += int(x & 1)
        x >>= 1
    return count

# === Precompute Static Tables ===
@njit
def precompute_lookup_tables(n_attacks, k_attacks, white_p_attacks, black_p_attacks):
    KNIGHT_DELTAS = [(2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2), (1,-2), (2,-1)]
    KING_DELTAS = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
    
    for sq in range(64):
        r = sq // 8
        c = sq % 8

        # Knight attacks
        knight_attacks = np.uint64(0)
        for dr, dc in KNIGHT_DELTAS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                knight_attacks |= square_mask(nr * 8 + nc)
        n_attacks[sq] = knight_attacks

        # King attacks
        king_attacks = np.uint64(0)
        for dr, dc in KING_DELTAS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                king_attacks |= square_mask(nr * 8 + nc)
        k_attacks[sq] = king_attacks

        # Pawn attacks
        white_pawn_attacks = np.uint64(0)
        black_pawn_attacks = np.uint64(0)

        if r < 7:
            if c > 0:
                white_pawn_attacks |= square_mask((r+1)*8 + (c-1))
            if c < 7:
                white_pawn_attacks |= square_mask((r+1)*8 + (c+1))
        if r > 0:
            if c > 0:
                black_pawn_attacks |= square_mask((r-1)*8 + (c-1))
            if c < 7:
                black_pawn_attacks |= square_mask((r-1)*8 + (c+1))

        white_p_attacks[sq] = white_pawn_attacks
        black_p_attacks[sq] = black_pawn_attacks

@njit
def generate_mask_rook(square):
    mask = np.uint64(0)
    r, f = divmod(square, 8)
    for i in range(r+1,7): mask |= square_mask(i*8+f)
    for i in range(r-1,0,-1): mask |= square_mask(i*8+f)
    for i in range(f+1,7): mask |= square_mask(r*8+i)
    for i in range(f-1,0,-1): mask |= square_mask(r*8+i)
    return mask

@njit
def generate_mask_bishop(square):
    mask = np.uint64(0)
    r, f = divmod(square, 8)
    for dr, df in [(1,1), (1,-1), (-1,1), (-1,-1)]:
        nr, nf = r + dr, f + df
        while 0 < nr < 7 and 0 < nf < 7:
            mask |= square_mask(nr*8+nf)
            nr += dr
            nf += df
    return mask

@njit
def occupancy_from_index(index, mask):
    occ = np.uint64(0)
    bit_idx = 0
    for square in range(64):
        if (mask >> square) & 1:
            if (index >> bit_idx) & 1:
                occ |= np.uint64(1) << np.uint64(square)
            bit_idx += 1
    return occ

@njit
def compute_rook_attacks(square, occ):
    attacks = np.uint64(0)
    r, f = divmod(square, 8)
    for dr, df in [(1,0), (-1,0), (0,1), (0,-1)]:
        nr, nf = r, f
        while True:
            nr += dr
            nf += df
            if not (0 <= nr < 8 and 0 <= nf < 8): break
            attacks |= square_mask(nr*8+nf)
            if occ & square_mask(nr*8+nf): break
    return attacks

@njit
def compute_bishop_attacks(square, occ):
    attacks = np.uint64(0)
    r, f = divmod(square, 8)
    for dr, df in [(1,1), (1,-1), (-1,1), (-1,-1)]:
        nr, nf = r, f
        while True:
            nr += dr
            nf += df
            if not (0 <= nr < 8 and 0 <= nf < 8):
                break
            to_sq = nr * 8 + nf
            attacks |= np.uint64(1) << np.uint64(to_sq)
            if (occ >> np.uint64(to_sq)) & np.uint64(1):
                break
    return attacks

@njit
def compute_magic_entries_for_square(sq, rook_magic, bishop_magic):
    rook_mask = generate_mask_rook(sq)
    bishop_mask = generate_mask_bishop(sq)

    rook_entries = []
    rook_bits = popcount(rook_mask)
    for idx in range(1 << rook_bits):
        occ = occupancy_from_index(idx, rook_mask)
        magic_idx = ((occ * rook_magic) & np.uint64(0xFFFFFFFFFFFFFFFF)) >> (64 - rook_bits)
        attack = compute_rook_attacks(sq, occ)
        rook_entries.append((magic_idx, attack))

    bishop_entries = []
    bishop_bits = popcount(bishop_mask)
    for idx in range(1 << bishop_bits):
        occ = occupancy_from_index(idx, bishop_mask)
        magic_idx = ((occ * bishop_magic) & np.uint64(0xFFFFFFFFFFFFFFFF)) >> (64 - bishop_bits)
        attack = compute_bishop_attacks(sq, occ)
        bishop_entries.append((magic_idx, attack))

    return rook_mask, bishop_mask, rook_entries, bishop_entries

# --- Fill real tables ---
def precompute_magic_tables():
    for sq in range(64):
        rook_mask, bishop_mask, rook_entries, bishop_entries = compute_magic_entries_for_square(
            sq,
            ROOK_MAGICS[sq % len(ROOK_MAGICS)],
            BISHOP_MAGICS[sq % len(BISHOP_MAGICS)],
        )

        ROOK_MASKS[sq] = rook_mask
        BISHOP_MASKS[sq] = bishop_mask

        for magic_idx, attack in rook_entries:
            ROOK_ATTACKS[sq][magic_idx] = attack

        for magic_idx, attack in bishop_entries:
            BISHOP_ATTACKS[sq][magic_idx] = attack

# === Attack Access Functions ===
@njit
def knight_attacks(square):
    return KNIGHT_ATTACKS[square]

@njit
def king_attacks(square):
    return KING_ATTACKS[square]

@njit
def pawn_attacks(square, is_white):
    return WHITE_PAWN_ATTACKS[square] if is_white else BLACK_PAWN_ATTACKS[square]

@njit
def rook_attacks(square, occupancy):
    mask = ROOK_MASKS[square]
    relevant = occupancy & mask
    magic = ROOK_MAGICS[square % len(ROOK_MAGICS)]
    index = ((relevant * magic) & np.uint64(0xFFFFFFFFFFFFFFFF)) >> (64 - popcount(mask))
    return ROOK_ATTACKS[square][index]

@njit
def bishop_attacks(square, occupancy):
    mask = BISHOP_MASKS[square]
    relevant = occupancy & mask
    magic = BISHOP_MAGICS[square % len(BISHOP_MAGICS)]
    index = ((relevant * magic) & np.uint64(0xFFFFFFFFFFFFFFFF)) >> (64 - popcount(mask))
    return BISHOP_ATTACKS[square][index]

@njit
def queen_attacks(square, occupancy):
    return rook_attacks(square, occupancy) | bishop_attacks(square, occupancy)

# === Initialize Tables ===
precompute_lookup_tables(KNIGHT_ATTACKS, KING_ATTACKS, WHITE_PAWN_ATTACKS, BLACK_PAWN_ATTACKS)
precompute_magic_tables()