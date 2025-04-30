import random
# import pickle
# from pathlib import Path

# bishop_attack_table = [[0 for _ in range(512)] for _ in range(64)]
# rook_attack_table = [[0 for _ in range(4096)] for _ in range(64)]

# BISHOP_MASKS = [0] * 64
# BISHOP_MAGICS = [0] * 64
# BISHOP_SHIFTS = [0] * 64

# ROOK_MASKS = [0] * 64
# ROOK_MAGICS = [0] * 64
# ROOK_SHIFTS = [0] * 64

from magic_tables import (
    BISHOP_MASKS, 
    BISHOP_MAGICS, 
    BISHOP_SHIFTS,
    ROOK_MASKS, 
    ROOK_MAGICS, 
    ROOK_SHIFTS, 
    rook_attack_table, 
    bishop_attack_table
)

def popcount(x):
    return bin(x).count("1")

def generate_mask_bishop(square):
    mask = 0
    r, f = divmod(square, 8)
    for dr, df in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nf = r + dr, f + df
        while 0 < nr < 7 and 0 < nf < 7:
            mask |= 1 << (nr * 8 + nf)
            nr += dr
            nf += df
    return mask

def generate_mask_rook(square):
    mask = 0
    r, f = divmod(square, 8)
    for i in range(r + 1, 7): mask |= 1 << (i * 8 + f)
    for i in range(r - 1, 0, -1): mask |= 1 << (i * 8 + f)
    for i in range(f + 1, 7): mask |= 1 << (r * 8 + i)
    for i in range(f - 1, 0, -1): mask |= 1 << (r * 8 + i)
    return mask

def compute_bishop_attacks(square, occupancy):
    attacks = 0
    r, f = divmod(square, 8)
    for dr, df in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nf = r + dr, f + df
        while 0 <= nr < 8 and 0 <= nf < 8:
            sq = nr * 8 + nf
            attacks |= 1 << sq
            if occupancy & (1 << sq): break
            nr += dr
            nf += df
    return attacks

def compute_rook_attacks(square, occupancy):
    attacks = 0
    r, f = divmod(square, 8)
    for dr, df in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nf = r + dr, f + df
        while 0 <= nr < 8 and 0 <= nf < 8:
            sq = nr * 8 + nf
            attacks |= 1 << sq
            if occupancy & (1 << sq): break
            nr += dr
            nf += df
    return attacks

def set_occupancy(index, bits, mask):
    occ = 0
    bit = 0
    for i in range(64):
        if (mask >> i) & 1:
            if (index >> bit) & 1:
                occ |= 1 << i
            bit += 1
            if bit >= bits:
                break
    return occ

def find_magic(square, mask_fn, attack_fn, entries, max_attempts=1000000):
    mask = mask_fn(square)
    bits = popcount(mask)
    occupancy_count = 1 << bits

    occupancies = [set_occupancy(i, bits, mask) for i in range(occupancy_count)]
    attacks = [attack_fn(square, occ) for occ in occupancies]

    for _ in range(max_attempts):
        magic = random.getrandbits(64) & random.getrandbits(64) & random.getrandbits(64)
        used = {}
        success = True
        for i in range(occupancy_count):
            index = ((occupancies[i] * magic) & 0xFFFFFFFFFFFFFFFF) >> (64 - bits)
            if index in used:
                if used[index] != attacks[i]:
                    success = False
                    break
            else:
                used[index] = attacks[i]
        if success:
            for i in range(occupancy_count):
                index = ((occupancies[i] * magic) & 0xFFFFFFFFFFFFFFFF) >> (64 - bits)
                entries[square][index] = attacks[i]
            return mask, magic, 64 - bits
    raise RuntimeError(f"Failed to find magic for square {square}")

def generate_all_magics():
    for square in range(64):
        mask, magic, shift = find_magic(square, generate_mask_bishop, compute_bishop_attacks, bishop_attack_table)
        BISHOP_MASKS[square] = mask
        BISHOP_MAGICS[square] = magic
        BISHOP_SHIFTS[square] = shift

        mask, magic, shift = find_magic(square, generate_mask_rook, compute_rook_attacks, rook_attack_table)
        ROOK_MASKS[square] = mask
        ROOK_MAGICS[square] = magic
        ROOK_SHIFTS[square] = shift

def bishop_attacks(square, occupancy):
    mask = BISHOP_MASKS[square]
    magic = BISHOP_MAGICS[square]
    shift = BISHOP_SHIFTS[square]
    index = ((int(occupancy) & mask) * magic & 0xFFFFFFFFFFFFFFFF) >> shift
    return bishop_attack_table[square][index]

def rook_attacks(square, occupancy):
    mask = ROOK_MASKS[square]
    magic = ROOK_MAGICS[square]
    shift = ROOK_SHIFTS[square]
    index = ((int(occupancy) & mask) * magic & 0xFFFFFFFFFFFFFFFF) >> shift
    return rook_attack_table[square][index]

def queen_attacks(square, occupancy):
    return bishop_attacks(square, occupancy) | rook_attacks(square, occupancy)

def write_constants_to_file():
    with open("src/magic_tables.py", "w") as f:
        f.write("BISHOP_MASKS = [\n")
        for val in BISHOP_MASKS:
            f.write(f"    0x{val:016x},\n")
        f.write("]\n\n")

        f.write("BISHOP_MAGICS = [\n")
        for val in BISHOP_MAGICS:
            f.write(f"    0x{val:016x},\n")
        f.write("]\n\n")

        f.write("BISHOP_SHIFTS = [\n")
        for val in BISHOP_SHIFTS:
            f.write(f"    {val},\n")
        f.write("]\n\n")

        f.write("ROOK_MASKS = [\n")
        for val in ROOK_MASKS:
            f.write(f"    0x{val:016x},\n")
        f.write("]\n\n")

        f.write("ROOK_MAGICS = [\n")
        for val in ROOK_MAGICS:
            f.write(f"    0x{val:016x},\n")
        f.write("]\n\n")

        f.write("ROOK_SHIFTS = [\n")
        for val in ROOK_SHIFTS:
            f.write(f"    {val},\n")
        f.write("]\n\n")

        f.write("bishop_attack_table = [\n")
        for row in bishop_attack_table:
            f.write("    [")
            f.write(", ".join(f"0x{x:016x}" for x in row))
            f.write("],\n")
        f.write("]\n\n")

        f.write("rook_attack_table = [\n")
        for row in rook_attack_table:
            f.write("    [")
            f.write(", ".join(f"0x{x:016x}" for x in row))
            f.write("],\n")
        f.write("]\n")

# generate_all_magics()
# write_constants_to_file()
