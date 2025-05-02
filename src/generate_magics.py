import numpy as np


bishop_attack_table = np.zeros((64, 512), dtype=np.uint64)
rook_attack_table = np.zeros((64, 4096), dtype=np.uint64)

BISHOP_MASKS = np.zeros(64, dtype=np.uint64)
BISHOP_MAGICS = np.zeros(64, dtype=np.uint64)
BISHOP_SHIFTS = np.zeros(64, dtype=np.uint8)

ROOK_MASKS = np.zeros(64, dtype=np.uint64)
ROOK_MAGICS = np.zeros(64, dtype=np.uint64)
ROOK_SHIFTS = np.zeros(64, dtype=np.uint8)

def popcount(x):
    return bin(x).count("1")

def generate_mask_bishop(square):
    mask = np.uint64(0)
    r, f = divmod(square, 8)
    for dr, df in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        nr, nf = r + dr, f + df
        while 0 < nr < 7 and 0 < nf < 7:
            mask |= np.uint64(1) << np.uint64(nr * 8 + nf)
            nr += dr
            nf += df
    return mask

def generate_mask_rook(square):
    mask = np.uint64(0)
    r, f = divmod(square, 8)
    for i in range(r + 1, 7): mask |= np.uint64(1) << np.uint64(i * 8 + f)
    for i in range(r - 1, 0, -1): mask |= np.uint64(1) << np.uint64(i * 8 + f)
    for i in range(f + 1, 7): mask |= np.uint64(1) << np.uint64(r * 8 + i)
    for i in range(f - 1, 0, -1): mask |= np.uint64(1) << np.uint64(r * 8 + i)
    return mask

def compute_slider_attacks(square, occupancy, deltas):
    attacks = np.uint64(0)
    r, f = divmod(square, 8)
    for dr, df in deltas:
        nr, nf = r + dr, f + df
        while 0 <= nr < 8 and 0 <= nf < 8:
            sq = nr * 8 + nf
            attacks |= np.uint64(1) << np.uint64(sq)
            if occupancy & (np.uint64(1) << np.uint64(sq)):
                break
            nr += dr
            nf += df
    return attacks

def set_occupancy(index, bits, mask):
    occ = np.uint64(0)
    bit = 0
    for i in range(64):
        if (mask >> np.uint64(i)) & np.uint64(1):
            if (index >> bit) & 1:
                occ |= np.uint64(1) << np.uint64(i)
            bit += 1
            if bit >= bits:
                break
    return occ

def find_magic(square, mask_fn, attack_fn, entries, deltas):
    mask = np.uint64(mask_fn(square))
    bits = popcount(mask)
    occupancy_count = 1 << bits

    occupancies = [set_occupancy(i, bits, mask) for i in range(occupancy_count)]
    attacks = [attack_fn(square, occ, deltas) for occ in occupancies]

    for _ in range(1000000):
        rng = np.random.default_rng()
        found = {}
        magic = (
            rng.integers(0, 2**64, dtype='uint64') &
            rng.integers(0, 2**64, dtype='uint64') &
            rng.integers(0, 2**64, dtype='uint64')
        )
        used = {}
        success = True
        for i in range(occupancy_count):
            index = int(((occupancies[i] * magic) & np.uint64(0xFFFFFFFFFFFFFFFF)) >> np.uint64(64 - bits))
            if index in used:
                if used[index] != attacks[i]:
                    success = False
                    break
            else:
                used[index] = attacks[i]
        if success:
            for i in range(occupancy_count):
                index = int(((occupancies[i] * magic) & np.uint64(0xFFFFFFFFFFFFFFFF)) >> np.uint64(64 - bits))
                entries[square][index] = attacks[i]
                if square not in found:
                    found.add(square)
                    print(f"Magic found for square {square}")
            return mask, magic, 64 - bits
    raise RuntimeError(f"Failed to find magic for square {square}")

def generate_all_magics():
    for square in range(64):
        mask, magic, shift = find_magic(square, generate_mask_bishop, compute_slider_attacks, bishop_attack_table, [(-1,-1), (-1,1), (1,-1), (1,1)])
        BISHOP_MASKS[square] = mask
        BISHOP_MAGICS[square] = magic
        BISHOP_SHIFTS[square] = shift

        mask, magic, shift = find_magic(square, generate_mask_rook, compute_slider_attacks, rook_attack_table, [(-1,0), (1,0), (0,-1), (0,1)])
        ROOK_MASKS[square] = mask
        ROOK_MAGICS[square] = magic
        ROOK_SHIFTS[square] = shift

generate_all_magics()

def write_constants_to_file():
    with open("src/bitboard_magics.py", "w") as f:
        f.write("import numpy as np\n\n")

        f.write("BISHOP_MASKS = np.array([\n")
        for val in BISHOP_MASKS:
            f.write(f"    np.uint64(0x{int(val):016x}),\n")
        f.write("], dtype=np.uint64)\n\n")

        f.write("BISHOP_MAGICS = np.array([\n")
        for val in BISHOP_MAGICS:
            f.write(f"    np.uint64(0x{int(val):016x}),\n")
        f.write("], dtype=np.uint64)\n\n")

        f.write("BISHOP_SHIFTS = np.array([\n")
        for val in BISHOP_SHIFTS:
            f.write(f"    {int(val)},\n")
        f.write("], dtype=np.uint8)\n\n")

        f.write("ROOK_MASKS = np.array([\n")
        for val in ROOK_MASKS:
            f.write(f"    np.uint64(0x{int(val):016x}),\n")
        f.write("], dtype=np.uint64)\n\n")

        f.write("ROOK_MAGICS = np.array([\n")
        for val in ROOK_MAGICS:
            f.write(f"    np.uint64(0x{int(val):016x}),\n")
        f.write("], dtype=np.uint64)\n\n")

        f.write("ROOK_SHIFTS = np.array([\n")
        for val in ROOK_SHIFTS:
            f.write(f"    {int(val)},\n")
        f.write("], dtype=np.uint8)\n\n")

        f.write("bishop_attack_table = np.array([\n")
        for row in bishop_attack_table:
            f.write("    [")
            f.write(", ".join(f"np.uint64(0x{int(x):016x})" for x in row))
            f.write("],\n")
        f.write("], dtype=np.uint64)\n\n")

        f.write("rook_attack_table = np.array([\n")
        for row in rook_attack_table:
            f.write("    [")
            f.write(", ".join(f"np.uint64(0x{int(x):016x})" for x in row))
            f.write("],\n")
        f.write("], dtype=np.uint64)\n")

generate_all_magics()
write_constants_to_file()
