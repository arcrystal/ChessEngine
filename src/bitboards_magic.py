
import numpy as np
from numba import njit
from constants import BISHOP_MASKS, BISHOP_MAGICS, BISHOP_SHIFTS
from constants import ROOK_MASKS, ROOK_MAGICS, ROOK_SHIFTS


bishop_attack_table_np = np.zeros((64, 512), dtype=np.uint64)
rook_attack_table_np = np.zeros((64, 4096), dtype=np.uint64)

@njit
def popcount(x):
    """Fast bit count."""
    count = 0
    while x:
        count += int(x & 1)
        x >>= 1
    return count

@njit
def square_mask(sq):
    return np.uint64(1) << np.uint64(sq)

def divmod64(square: int):
    return divmod(square, 8)

def generate_mask_bishop(square: int) -> int:
    mask = 0
    rank, file = divmod(square, 8)
    for dr, df in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        r, f = rank + dr, file + df
        while 0 < r < 7 and 0 < f < 7:  # exclude board edges
            mask |= 1 << (r * 8 + f)
            r += dr
            f += df
    return mask

def generate_mask_rook(square: int) -> int:
    mask = 0
    r, f = divmod64(square)
    for dr in [-1, 1]:
        nr = r + dr
        while 0 < nr < 7:
            mask |= square_mask(nr * 8 + f)
            nr += dr
    for df in [-1, 1]:
        nf = f + df
        while 0 < nf < 7:
            mask |= square_mask(r * 8 + nf)
            nf += df
    return mask

def bishop_attacks_on_the_fly(square: int, blockers: int) -> int:
    attacks = 0
    rank, file = divmod(square, 8)
    for dr, df in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        r, f = rank + dr, file + df
        while 0 <= r <= 7 and 0 <= f <= 7:
            sq = r * 8 + f
            attacks |= 1 << sq
            if blockers & (1 << sq):  # stop at first blocker
                break
            r += dr
            f += df
    return attacks

def rook_attacks_on_the_fly(square: int, occupancy: int) -> int:
    attacks = 0
    r, f = divmod64(square)
    for dr, df in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nf = r + dr, f + df
        while 0 <= nr <= 7 and 0 <= nf <= 7:
            sq = nr * 8 + nf
            attacks |= square_mask(sq)
            if occupancy & square_mask(sq):
                break
            nr += dr
            nf += df
    return attacks

def set_occupancy(index: int, bits_in_mask: int, mask: int) -> int:
    occ = 0
    bit_index = 0
    for i in range(64):
        if int(mask) & (1 << i):
            if index & (1 << bit_index):
                occ |= (1 << i)
            bit_index += 1
            if bit_index >= bits_in_mask:
                break
    return occ

def init_bishop_table_numpy():
    for square in range(64):
        mask = BISHOP_MASKS[square]
        magic = int(BISHOP_MAGICS[square])
        shift = int(BISHOP_SHIFTS[square])
        relevant_bits = popcount(mask)
        for index in range(1 << relevant_bits):
            occ = set_occupancy(index, relevant_bits, mask)
            magic_index = ((occ * magic) & 0xFFFFFFFFFFFFFFFF) >> shift
            bishop_attack_table_np[square, magic_index] = bishop_attacks_on_the_fly(square, occ)

def init_rook_table_numpy():
    for square in range(64):
        mask = ROOK_MASKS[square]
        magic = int(ROOK_MAGICS[square])
        shift = int(ROOK_SHIFTS[square])
        relevant_bits = popcount(mask)
        for index in range(1 << relevant_bits):
            occ = set_occupancy(index, relevant_bits, mask)
            magic_index = ((occ * magic) & 0xFFFFFFFFFFFFFFFF) >> shift
            rook_attack_table_np[square, magic_index] = rook_attacks_on_the_fly(square, occ)

@njit
def bishop_attacks(square: int, occupancy: int) -> np.uint64:
    mask = BISHOP_MASKS[square]
    magic = BISHOP_MAGICS[square]
    shift = BISHOP_SHIFTS[square]
    relevant_occ = occupancy & mask
    index = ((relevant_occ * magic) & 0xFFFFFFFFFFFFFFFF) >> shift
    return bishop_attack_table_np[square, index]

@njit
def rook_attacks(square: int, occupancy: int) -> np.uint64:
    mask = ROOK_MASKS[square]
    magic = ROOK_MAGICS[square]
    shift = ROOK_SHIFTS[square]
    relevant_occ = occupancy & mask
    index = ((relevant_occ * magic) & 0xFFFFFFFFFFFFFFFF) >> shift
    return rook_attack_table_np[square, index]

@njit
def queen_attacks(square: int, occupancy: int) -> np.uint64:
    return bishop_attacks(square, occupancy) | rook_attacks(square, occupancy)

def print_bitboard(bitboard: int, label: str = ""):
    if label:
        print(f"{label}")
    print("  +-----------------+")
    for rank in range(7, -1, -1):
        row = f"{rank + 1} |"
        for file in range(8):
            square = rank * 8 + file
            row += " X" if (bitboard >> square) & 1 else " ."
        row += " |"
        print(row)
    print("  +-----------------+")
    print("    a b c d e f g h\n")

# Initialize tables at module load
init_bishop_table_numpy()
init_rook_table_numpy()

