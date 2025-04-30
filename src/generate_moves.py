import numpy as np
from numba import njit

from bitboard_nomagic import (
    WHITE_PAWN_ATTACKS, BLACK_PAWN_ATTACKS, square_mask,
    knight_attacks, king_attacks,
)
from bitboard_magic import bishop_attacks, rook_attacks, queen_attacks
from constants import KNIGHT, BISHOP, ROOK, QUEEN


@njit
def rank_mask(rank):
    return np.uint64(0xFF) << np.uint64(rank * 8)

def pop_lsb(bb):
    """Pop least significant bit and return (index, new_bb)."""
    bb = int(bb)
    lsb = bb & -bb
    index = (lsb).bit_length() - 1
    bb &= bb - 1
    return index, bb

def bitwise_not_64(x):
    return np.uint64(~x)

def file_mask(file_idx):
    return 0x0101010101010101 << file_idx

# pawns = int(pawns)
# empty = int(~gs.occupied & 0xFFFFFFFFFFFFFFFF)
# enemy = int(gs.black_occupancy if is_white else gs.white_occupancy)

# =========== Move Generators ==============
def generate_pawn_moves(gs, pawns, is_white, verbose=False):
    moves = []
    pawns = int(pawns)
    empty = int(~gs.occupied & 0xFFFFFFFFFFFFFFFF)
    enemy = int(gs.black_occupancy if is_white else gs.white_occupancy)

    if is_white:
        single_push = (pawns << 8) & empty
        double_push = ((single_push & rank_mask(2)) << 8) & empty
        left_attacks = (pawns << 7) & enemy & ~file_mask(7)  # can't wrap from h-file
        right_attacks = (pawns << 9) & enemy & ~file_mask(0)  # can't wrap from a-file
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

    # Single push
    temp = single_push
    while temp:
        to_sq = (temp & -temp).bit_length() - 1
        from_sq = to_sq - direction
        if from_sq // 8 == promo_rank:
            for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(promo)))
        else:
            moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))
        temp &= temp - 1

    # Double push
    temp = double_push
    while temp:
        to_sq = (temp & -temp).bit_length() - 1
        from_sq = to_sq - 2 * direction
        moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))
        temp &= temp - 1

    # Captures (left)
    temp = left_attacks
    while temp:
        to_sq = (temp & -temp).bit_length() - 1
        from_sq = to_sq - (7 if is_white else -9)
        if from_sq // 8 == promo_rank:
            for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(promo)))
        else:
            moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))
        temp &= temp - 1

    # Captures (right)
    temp = right_attacks
    while temp:
        to_sq = (temp & -temp).bit_length() - 1
        from_sq = to_sq - (9 if is_white else -7)
        if from_sq // 8 == promo_rank:
            for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(promo)))
        else:
            moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))
        temp &= temp - 1

    # En passant
    if gs.en_passant_target != -1:
        ep_sq = gs.en_passant_target
        ep_file = ep_sq % 8
        ep_rank_bb = rank_mask(ep_rank)
        pawns_on_ep_rank = pawns & ep_rank_bb

        # Check if any pawn is in position to capture en passant
        for shift, file_diff in ((7, 1), (9, -1)) if is_white else ((-9, 1), (-7, -1)):
            from_sq = ep_sq - shift
            if 0 <= from_sq < 64 and abs((from_sq % 8) - ep_file) == 1:
                if (pawns_on_ep_rank >> from_sq) & 1:
                    moves.append((np.int8(from_sq), np.int8(ep_sq), np.int8(0)))

    return moves

def generate_knight_moves(gs, knights, is_white, verbose=False):
    moves = []
    own_pieces = int(gs.white_occupancy if is_white else gs.black_occupancy)
    while knights:
        from_sq, knights = pop_lsb(knights)
        attacks = knight_attacks(from_sq) & ~own_pieces
        if verbose:
            print(f"Knight moves from square {gs.get_standard_algebraic(from_sq)}")
            gs.print_board()
            gs.print_bitboard(attacks)
            print("--------\n")
        while attacks:
            to_sq, attacks = pop_lsb(attacks)
            moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))

    return moves

def generate_bishop_moves(gs, bishops, is_white, verbose=False):
    moves = []
    own_pieces = int(gs.white_occupancy if is_white else gs.black_occupancy)
    while bishops:
        from_sq, bishops = pop_lsb(bishops)
        attacks = bishop_attacks(from_sq, gs.occupied) & ~own_pieces
        if verbose:
            print(f"Bishop moves from square {gs.get_standard_algebraic(from_sq)}")
            gs.print_board()
            gs.print_bitboard(attacks)
            print("--------\n")

        while attacks:
            to_sq, attacks = pop_lsb(attacks)
            moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))

    return moves

def generate_rook_moves(gs, rooks, is_white, verbose=False):
    moves = []
    own_pieces = np.uint64(gs.white_occupancy if is_white else gs.black_occupancy)
    while rooks:
        from_sq, rooks = pop_lsb(rooks)
        attacks = np.uint64(rook_attacks(from_sq, gs.occupied)) & ~own_pieces
        if verbose:
            print(f"Rook moves from square {gs.get_standard_algebraic(from_sq)}")
            gs.print_board()
            gs.print_bitboard(attacks)
            print("--------\n")
        while attacks:
            to_sq, attacks = pop_lsb(attacks)
            moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))

    return moves

def generate_queen_moves(gs, queens, is_white, verbose=False):
    moves = []
    own_pieces = np.uint64(gs.white_occupancy if is_white else gs.black_occupancy)
    while queens:
        from_sq, queens = pop_lsb(queens)
        attacks = np.uint64(queen_attacks(from_sq, gs.occupied)) & ~own_pieces
        if verbose:
            print(f"Queen moves from square {gs.get_standard_algebraic(from_sq)}")
            gs.print_board()
            gs.print_bitboard(attacks)
            print("--------\n")

        while attacks:
            to_sq, attacks = pop_lsb(attacks)
            moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))

    return moves

def generate_king_moves(gs, king, is_white, verbose=False):
    moves = []
    own_pieces = np.uint64(gs.white_occupancy if is_white else gs.black_occupancy)

    if king == np.uint64(0):
        return moves  # No king found (should not happen, but safe)

    from_sq = int(np.log2(int(king)))  # Find king square quickly

    attacks = np.uint64(king_attacks(from_sq)) & ~own_pieces
    if verbose:
        print(f"King moves from square {gs.get_standard_algebraic(from_sq)}")
        gs.print_board()
        gs.print_bitboard(attacks)
        print("--------\n")

    while attacks:
        to_sq, attacks = pop_lsb(attacks)
        moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))

    return moves

def generate_castling_moves(gs, is_white, verbose=False):
    moves = []
    occupancy = int(gs.occupied)
    if is_white:
        king_sq = 4
        # White kingside castling (e1 to g1)
        if gs.castling_rights[0]:
            if not (occupancy  & (square_mask(5) | square_mask(6))):
                if not (gs.is_square_attacked(4, False) or gs.is_square_attacked(5, False) or gs.is_square_attacked(6, False)):
                    moves.append((np.int8(king_sq), np.int8(6), np.int8(0)))
        # White queenside castling (e1 to c1)
        if gs.castling_rights[1]:
            if not (occupancy  & (square_mask(1) | square_mask(2) | square_mask(3))):
                if not (gs.is_square_attacked(4, False) or gs.is_square_attacked(3, False) or gs.is_square_attacked(2, False)):
                    moves.append((np.int8(king_sq), np.int8(2), np.int8(0)))
    else:
        king_sq = 60
        # Black kingside castling (e8 to g8)
        if gs.castling_rights[2]:
            if not (occupancy  & (square_mask(61) | square_mask(62))):
                if not (gs.is_square_attacked(60, True) or gs.is_square_attacked(61, True) or gs.is_square_attacked(62, True)):
                    moves.append((np.int8(king_sq), np.int8(62), np.int8(0)))
        # Black queenside castling (e8 to c8)
        if gs.castling_rights[3]:
            if not (occupancy  & (square_mask(57) | square_mask(58) | square_mask(59))):
                if not (gs.is_square_attacked(60, True) or gs.is_square_attacked(59, True) or gs.is_square_attacked(58, True)):
                    moves.append((np.int8(king_sq), np.int8(58), np.int8(0)))
    
    if verbose:
        print(f"Castling moves from square {gs.get_standard_algebraic(king_sq)}:")
        gs.print_board()
        if moves:
            for move in moves:
                print(move)
        else:
            print("None")
        
        print("--------\n")  
            
    return moves

def generate_all_moves(gs, verbose=False):
    if gs.white_to_move:
        return generate_pawn_moves(gs, int(gs.white_pawns), True, verbose=verbose) \
        + generate_knight_moves(gs, int(gs.white_knights), True, verbose=verbose) \
        + generate_bishop_moves(gs, int(gs.white_bishops), True, verbose=verbose) \
        + generate_rook_moves(gs, int(gs.white_rooks), True, verbose=verbose) \
        + generate_queen_moves(gs, int(gs.white_queens), True, verbose=verbose) \
        + generate_king_moves(gs, int(gs.white_king), True, verbose=verbose) \
        + generate_castling_moves(gs, True, verbose=verbose)

    return generate_pawn_moves(gs, int(gs.black_pawns), False, verbose=verbose) \
    + generate_knight_moves(gs, int(gs.black_knights), False, verbose=verbose) \
    + generate_bishop_moves(gs, int(gs.black_bishops), False, verbose=verbose) \
    + generate_rook_moves(gs, int(gs.black_rooks), False, verbose=verbose) \
    + generate_queen_moves(gs, int(gs.black_queens), False, verbose=verbose) \
    + generate_king_moves(gs, int(gs.black_king), False, verbose=verbose) \
    + generate_castling_moves(gs, False, verbose=verbose)
        
# def generate_all_legal_moves(gs, verbose=False):
#     """Generate only *legal* moves (no king left in check)."""
#     legal_moves = []
#     side_to_move = gs.white_to_move

#     for move in generate_all_moves(gs, verbose):
#         gs.make_move(move)
#         if not gs.is_check(side_to_move):
#             legal_moves.append(move)
        
#         gs.undo_move()
        
#     return legal_moves