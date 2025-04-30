import random

# Constants
NUM_SQUARES = 64

# Directions for bishop movement
DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

def square_to_coords(square):
    return divmod(square, 8)

def coords_to_square(rank, file):
    return rank * 8 + file

def is_on_board(rank, file):
    return 0 <= rank < 8 and 0 <= file < 8

def generate_bishop_mask(square):
    mask = 0
    rank, file = square_to_coords(square)
    for dr, df in DIRECTIONS:
        r, f = rank + dr, file + df
        while is_on_board(r, f):
            if r == 0 or r == 7 or f == 0 or f == 7:
                break  # Exclude edge squares
            mask |= 1 << coords_to_square(r, f)
            r += dr
            f += df
    return mask

def generate_blocker_boards(mask):
    bits = [i for i in range(64) if (mask >> i) & 1]
    blocker_boards = []
    for i in range(1 << len(bits)):
        blocker = 0
        for j in range(len(bits)):
            if (i >> j) & 1:
                blocker |= 1 << bits[j]
        blocker_boards.append(blocker)
    return blocker_boards

def bishop_attacks_on_the_fly(square, blockers):
    attacks = 0
    rank, file = square_to_coords(square)
    for dr, df in DIRECTIONS:
        r, f = rank + dr, file + df
        while is_on_board(r, f):
            sq = coords_to_square(r, f)
            attacks |= 1 << sq
            if blockers & (1 << sq):
                break
            r += dr
            f += df
    return attacks

def find_magic_number(square, mask, relevant_bits):
    blocker_boards = generate_blocker_boards(mask)
    attack_sets = [bishop_attacks_on_the_fly(square, b) for b in blocker_boards]
    for _ in range(1000000):
        magic = random.getrandbits(64) & random.getrandbits(64) & random.getrandbits(64)
        if bin((mask * magic) & 0xFF00000000000000).count('1') < 6:
            used = {}
            success = True
            for b, a in zip(blocker_boards, attack_sets):
                index = ((b * magic) & 0xFFFFFFFFFFFFFFFF) >> (64 - relevant_bits)
                if index in used:
                    if used[index] != a:
                        success = False
                        break
                else:
                    used[index] = a
            if success:
                return magic
    raise Exception(f"No magic number found for square {square}")

def generate_bishop_magics():
    masks = []
    magics = []
    shifts = []
    for square in range(NUM_SQUARES):
        mask = generate_bishop_mask(square)
        relevant_bits = bin(mask).count('1')
        magic = find_magic_number(square, mask, relevant_bits)
        masks.append(f"0x{mask:016x}")
        magics.append(f"0x{magic:016x}")
        shifts.append(64 - relevant_bits)
    return masks, magics, shifts

# Generate the bishop magic table
mask, magic, shift = generate_bishop_magics()

print(mask)
print(magic)
print(shift)