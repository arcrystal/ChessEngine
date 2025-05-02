import numpy as np
from numba import njit, uint64
from bitboard_nomagic import pawn_attacks, king_attacks, knight_attacks
from bitboard_magic import bishop_attacks, rook_attacks, queen_attacks
from numba import int8, int32, int64


@njit
def pop_lsb(bb: uint64) -> tuple:
    if bb == np.uint64(0):
        return -1, np.uint64(0)

    # Safely compute lsb using uint64 mask
    lsb = np.bitwise_and(bb, np.uint64(-int(bb)))
    index = np.uint64(0)
    temp = lsb

    while temp > 1:
        temp = temp >> 1
        index += 1

    new_bb = np.bitwise_and(bb, bb - np.uint64(1))
    return int(index), new_bb
        
@njit
def update_occupancies_numba(gs):
    gs.white_occupancy = (
        gs.white_pawns | gs.white_knights | gs.white_bishops |
        gs.white_rooks | gs.white_queens | gs.white_king
    )
    gs.black_occupancy = (
        gs.black_pawns | gs.black_knights | gs.black_bishops |
        gs.black_rooks | gs.black_queens | gs.black_king
    )
    gs.occupied = gs.white_occupancy | gs.black_occupancy

@njit
def apply_move_numba(gs, move):
    from_sq, to_sq, promo = move
    mover_bb = uint64(1) << uint64(from_sq)
    to_bb = uint64(1) << uint64(to_sq)

    wp = np.array([
        gs.white_pawns, gs.white_knights, gs.white_bishops,
        gs.white_rooks, gs.white_queens, gs.white_king
    ], dtype=uint64)

    bp = np.array([
        gs.black_pawns, gs.black_knights, gs.black_bishops,
        gs.black_rooks, gs.black_queens, gs.black_king
    ], dtype=uint64)

    is_white = gs.white_to_move
    ep_target = gs.en_passant_target

    side = wp if is_white else bp
    opp = bp if is_white else wp

    for i in range(6):
        if side[i] & mover_bb:
            side[i] ^= mover_bb
            side[i] |= to_bb
            # 0 == PAWN
            if i == 0:
                if to_sq == ep_target:
                    ep_sq = to_sq + (-8 if is_white else 8)
                    opp[0] &= ~(uint64(1) << uint64(ep_sq))
                if promo != 0:
                    side[0] &= ~to_bb
                    side[promo] |= to_bb
                gs.en_passant_target = (from_sq + to_sq) // 2 if abs(from_sq - to_sq) == 16 else -1
            else:
                gs.en_passant_target = -1
            break

    for i in range(6):
        if opp[i] & to_bb:
            opp[i] &= ~to_bb
            break

    if is_white:
        gs.white_pawns, gs.white_knights, gs.white_bishops, \
        gs.white_rooks, gs.white_queens, gs.white_king = side
        gs.black_pawns, gs.black_knights, gs.black_bishops, \
        gs.black_rooks, gs.black_queens, gs.black_king = opp
    else:
        gs.black_pawns, gs.black_knights, gs.black_bishops, \
        gs.black_rooks, gs.black_queens, gs.black_king = side
        gs.white_pawns, gs.white_knights, gs.white_bishops, \
        gs.white_rooks, gs.white_queens, gs.white_king = opp

    update_occupancies_numba(gs)
    gs.white_to_move = not gs.white_to_move
    # === Append current state ===
    return (
        uint64(gs.white_pawns), uint64(gs.white_knights), uint64(gs.white_bishops),
        uint64(gs.white_rooks), uint64(gs.white_queens), uint64(gs.white_king),
        uint64(gs.black_pawns), uint64(gs.black_knights), uint64(gs.black_bishops),
        uint64(gs.black_rooks), uint64(gs.black_queens), uint64(gs.black_king),
        uint64(gs.white_occupancy), uint64(gs.black_occupancy), uint64(gs.occupied),
        gs.white_to_move,
        (int8(gs.castling_rights[0]), int8(gs.castling_rights[1]),
        int8(gs.castling_rights[2]), int8(gs.castling_rights[3])),
        int64(gs.en_passant_target),
        int32(gs.halfmove_clock),
        int32(gs.fullmove_number)
    )

@njit
def undo_move_numba(gs, move_info):
    (
        gs.white_pawns, gs.white_knights, gs.white_bishops, gs.white_rooks, gs.white_queens, gs.white_king,
        gs.black_pawns, gs.black_knights, gs.black_bishops, gs.black_rooks, gs.black_queens, gs.black_king,
        gs.white_occupancy, gs.black_occupancy, gs.occupied,
        gs.white_to_move, prev_castling_rights, gs.en_passant_target, gs.halfmove_clock, gs.fullmove_number
    ) = move_info.pop()
    for i in range(4):
        gs.castling_rights[i] = prev_castling_rights[i]

@njit
def is_check_numba(gs, is_white: bool) -> bool:
    king_bb = gs.white_king if is_white else gs.black_king
    king_sq, _ = pop_lsb(king_bb)
    occ = gs.white_occupancy | gs.black_occupancy

    ep = not is_white
    if pawn_attacks(king_sq, ep) & (gs.black_pawns if is_white else gs.white_pawns):
        return True
    if knight_attacks(king_sq) & (gs.black_knights if is_white else gs.white_knights):
        return True
    if bishop_attacks(king_sq, occ) & ((gs.black_bishops | gs.black_queens) if is_white else (gs.white_bishops | gs.white_queens)):
        return True
    if rook_attacks(king_sq, occ) & ((gs.black_rooks | gs.black_queens) if is_white else (gs.white_rooks | gs.white_queens)):
        return True
    if king_attacks(king_sq) & (gs.black_king if is_white else gs.white_king):
        return True

    return False

@njit
def attack_map_numba(gs, is_white: bool) -> uint64:
    occ = gs.occupied
    attacks = uint64(0)

    pawns   = gs.white_pawns if is_white else gs.black_pawns
    knights = gs.white_knights if is_white else gs.black_knights
    bishops = gs.white_bishops if is_white else gs.black_bishops
    rooks   = gs.white_rooks if is_white else gs.black_rooks
    queens  = gs.white_queens if is_white else gs.black_queens
    king    = gs.white_king if is_white else gs.black_king

    while pawns:
        sq, pawns = pop_lsb(pawns)
        attacks |= pawn_attacks(sq, is_white)

    while knights:
        sq, knights = pop_lsb(knights)
        attacks |= knight_attacks(sq)

    while bishops:
        sq, bishops = pop_lsb(bishops)
        attacks |= bishop_attacks(sq, occ)

    while rooks:
        sq, rooks = pop_lsb(rooks)
        attacks |= rook_attacks(sq, occ)

    while queens:
        sq, queens = pop_lsb(queens)
        attacks |= queen_attacks(sq, occ)

    if king:
        sq, _ = pop_lsb(king)
        attacks |= king_attacks(sq)

    return attacks