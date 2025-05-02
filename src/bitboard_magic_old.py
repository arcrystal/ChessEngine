
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
