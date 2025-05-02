import numpy as np
from numba import njit, uint64

from bitboard_nomagic import (
    square_mask, knight_attacks, king_attacks, pawn_attacks
)
from bitboard_magic import bishop_attacks, rook_attacks, queen_attacks
from constants import KNIGHT, BISHOP, ROOK, QUEEN
from bitboard_utils import attack_map_numba, pop_lsb

@njit
def rank_mask(rank):
    return uint64(0xFF) << uint64(rank * 8)

@njit
def file_mask(file_idx):
    return uint64(0x0101010101010101) << uint64(file_idx)

@njit
def generate_pawn_moves(gs, pawns, is_white):
    moves = []
    empty = ~gs.occupied
    enemy = gs.black_occupancy if is_white else gs.white_occupancy

    if is_white:
        single_push = (pawns << 8) & empty
        double_push = ((single_push & rank_mask(2)) << 8) & empty
        left_attacks = (pawns << 7) & enemy & ~file_mask(7)
        right_attacks = (pawns << 9) & enemy & ~file_mask(0)
        promo_rank = 6
        ep_rank = 4
        direction = 8
    else:
        single_push = (pawns >> 8) & empty
        double_push = ((single_push & rank_mask(5)) >> 8) & empty
        left_attacks = (pawns >> 9) & enemy & ~file_mask(7)
        right_attacks = (pawns >> 7) & enemy & ~file_mask(0)
        promo_rank = 1
        ep_rank = 3
        direction = -8

    for push in [single_push, double_push, left_attacks, right_attacks]:
        temp = push
        while temp:
            sq, temp = pop_lsb(temp)
            from_sq = sq - direction if push is not double_push else sq - 2 * direction
            if from_sq // 8 == promo_rank:
                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves.append((from_sq, sq, promo))
            else:
                moves.append((from_sq, sq, 0))

    # En passant
    if gs.en_passant_target != -1:
        ep_sq = gs.en_passant_target
        ep_bb = square_mask(ep_sq)
        adj = ((pawns << 1) | (pawns >> 1)) & rank_mask(ep_rank)
        if adj & ep_bb:
            from_sq = ep_sq - direction + (1 if (pawns >> (ep_sq - 1)) & 1 else -1)
            moves.append((from_sq, ep_sq, 0))

    return moves

@njit
def generate_slider_moves(gs, pieces, is_white, attack_fn):
    moves = []
    own_pieces = gs.white_occupancy if is_white else gs.black_occupancy
    temp = pieces
    while temp:
        from_sq, temp = pop_lsb(temp)
        attacks = attack_fn(from_sq, gs.occupied) & ~own_pieces
        while attacks:
            to_sq, attacks = pop_lsb(attacks)
            moves.append((from_sq, to_sq, 0))
    return moves

@njit
def generate_knight_moves(gs, knights, is_white):
    moves = []
    own_pieces = gs.white_occupancy if is_white else gs.black_occupancy
    while knights:
        from_sq, knights = pop_lsb(knights)
        attacks = knight_attacks(from_sq) & ~own_pieces
        while attacks:
            to_sq, attacks = pop_lsb(attacks)
            moves.append((from_sq, to_sq, 0))
    return moves

@njit
def generate_king_moves(gs, king_bb, is_white):
    moves = []
    from_sq = pop_lsb(king_bb)[0]
    own_pieces = gs.white_occupancy if is_white else gs.black_occupancy
    enemy_attacks = attack_map_numba(gs, not is_white)
    legal = king_attacks(from_sq) & ~own_pieces & ~enemy_attacks
    while legal:
        to_sq, legal = pop_lsb(legal)
        moves.append((from_sq, to_sq, 0))
    return moves

@njit
def generate_all_moves(gs):
    if gs.white_to_move:
        return (
            generate_pawn_moves(gs, gs.white_pawns, True)
            + generate_knight_moves(gs, gs.white_knights, True)
            + generate_slider_moves(gs, gs.white_bishops, True, bishop_attacks)
            + generate_slider_moves(gs, gs.white_rooks, True, rook_attacks)
            + generate_slider_moves(gs, gs.white_queens, True, queen_attacks)
            + generate_king_moves(gs, gs.white_king, True)
        )
    else:
        return (
            generate_pawn_moves(gs, gs.black_pawns, False)
            + generate_knight_moves(gs, gs.black_knights, False)
            + generate_slider_moves(gs, gs.black_bishops, False, bishop_attacks)
            + generate_slider_moves(gs, gs.black_rooks, False, rook_attacks)
            + generate_slider_moves(gs, gs.black_queens, False, queen_attacks)
            + generate_king_moves(gs, gs.black_king, False)
        )
        
# def generate_all_moves(gs, verbose=False):
#     if gs.white_to_move:
#         return generate_pawn_moves(gs, int(gs.white_pawns), True, verbose=True) \
#         + generate_knight_moves(gs, int(gs.white_knights), True, verbose=verbose) \
#         + generate_bishop_moves(gs, int(gs.white_bishops), True, verbose=verbose) \
#         + generate_rook_moves(gs, int(gs.white_rooks), True, verbose=verbose) \
#         + generate_queen_moves(gs, int(gs.white_queens), True, verbose=verbose) \
#         + generate_king_moves(gs, int(gs.white_king), True, verbose=verbose) \
#         + generate_castling_moves(gs, True, verbose=verbose)

#     return generate_pawn_moves(gs, int(gs.black_pawns), False, verbose=verbose) \
#     + generate_knight_moves(gs, int(gs.black_knights), False, verbose=verbose) \
#     + generate_bishop_moves(gs, int(gs.black_bishops), False, verbose=verbose) \
#     + generate_rook_moves(gs, int(gs.black_rooks), False, verbose=verbose) \
#     + generate_queen_moves(gs, int(gs.black_queens), False, verbose=verbose) \
#     + generate_king_moves(gs, int(gs.black_king), False, verbose=verbose) \
#     + generate_castling_moves(gs, False, verbose=verbose)
