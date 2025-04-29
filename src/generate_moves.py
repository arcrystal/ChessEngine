import numpy as np
from bitboards import (
    WHITE_PAWN_ATTACKS, BLACK_PAWN_ATTACKS, 
    rank_mask, square_mask,
    knight_attacks, king_attacks,
    bishop_attacks, rook_attacks, queen_attacks
)
from constants import KNIGHT, BISHOP, ROOK, QUEEN

def pop_lsb(bb):
    """Pop least significant bit and return (index, new_bb)."""
    bb = int(bb)
    lsb = bb & -bb
    index = (lsb).bit_length() - 1
    bb &= bb - 1
    return index, bb

def bitwise_not_64(x):
    return np.uint64(~x)

# =========== Move Generators ==============
def generate_pawn_moves(gs, pawns, is_white):
    moves = []
    empty = int(~gs.occupied)
    enemy = int(gs.black_occupancy if is_white else gs.white_occupancy)

    if is_white:
        push_one = (pawns << 8) & empty
        push_two = ((pawns & rank_mask(1)) << 16) & empty
        attack_table = WHITE_PAWN_ATTACKS
        promo_rank = 6
        ep_rank = 4
    else:
        push_one = (pawns >> 8) & empty
        push_two = ((pawns & rank_mask(6)) >> 16) & empty
        attack_table = BLACK_PAWN_ATTACKS
        promo_rank = 1
        ep_rank = 3

    # --- Single Pushes ---
    mask = push_one
    while mask:
        to_sq = (mask & -mask).bit_length() - 1
        from_sq = to_sq - (8 if is_white else -8)
        if from_sq // 8 == promo_rank:
            for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(promo)))
        else:
            moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))
        mask &= mask - 1

    # --- Double Pushes ---
    mask = push_two
    while mask:
        to_sq = (mask & -mask).bit_length() - 1
        from_sq = to_sq - (16 if is_white else -16)
        moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))
        mask &= mask - 1

    # --- Captures ---
    mask = pawns
    while mask:
        from_sq = (mask & -mask).bit_length() - 1
        attacks = int(attack_table[from_sq]) & enemy
        while attacks:
            to_sq = (attacks & -attacks).bit_length() - 1
            if from_sq // 8 == promo_rank:
                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(promo)))
            else:
                moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))
            attacks &= attacks - 1
        mask &= mask - 1

    # --- En Passant ---
    if gs.en_passant_target != -1:
        ep_sq = gs.en_passant_target
        ep_rank_file = divmod(ep_sq, 8)
        mask = pawns
        while mask:
            from_sq = (mask & -mask).bit_length() - 1
            from_rank, from_file = divmod(from_sq, 8)
            if abs(from_file - ep_rank_file[1]) == 1 and from_rank == ep_rank:
                moves.append((np.int8(from_sq), np.int8(ep_sq), np.int8(0)))
            mask &= mask - 1

    return moves

def generate_knight_moves(gs, knights, is_white):
    moves = []
    own_pieces = int(gs.white_occupancy if is_white else gs.black_occupancy)
    while knights:
        from_sq, knights = pop_lsb(knights)
        attacks = knight_attacks(from_sq) & ~own_pieces
        while attacks:
            to_sq, attacks = pop_lsb(attacks)
            moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))

    return moves

def generate_bishop_moves(gs, bishops, is_white):
    moves = []
    own_pieces = int(gs.white_occupancy if is_white else gs.black_occupancy)
    while bishops:
        from_sq, bishops = pop_lsb(bishops)
        attacks = bishop_attacks(from_sq, gs.occupied) & ~own_pieces

        while attacks:
            to_sq, attacks = pop_lsb(attacks)
            moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))

    return moves

def generate_rook_moves(gs, rooks, is_white):
    moves = []
    own_pieces = np.uint64(gs.white_occupancy if is_white else gs.black_occupancy)
    while rooks:
        from_sq, rooks = pop_lsb(rooks)
        attacks = np.uint64(rook_attacks(from_sq, gs.occupied)) & ~own_pieces
        attack_mask = attacks

        while attack_mask:
            to_sq, attack_mask = pop_lsb(attack_mask)
            moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))

    return moves

def generate_queen_moves(gs, queens, is_white):
    moves = []
    own_pieces = np.uint64(gs.white_occupancy if is_white else gs.black_occupancy)
    while queens:
        from_sq, queens = pop_lsb(queens)
        attacks = np.uint64(queen_attacks(from_sq, gs.occupied)) & ~own_pieces
        attack_mask = attacks

        while attack_mask:
            to_sq, attack_mask = pop_lsb(attack_mask)
            moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))

    return moves

def generate_king_moves(gs, king, is_white):
    moves = []
    own_pieces = np.uint64(gs.white_occupancy if is_white else gs.black_occupancy)

    if king == np.uint64(0):
        return moves  # No king found (should not happen, but safe)

    from_sq = int(np.log2(int(king)))  # Find king square quickly

    attacks = np.uint64(king_attacks(from_sq)) & ~own_pieces
    attack_mask = attacks

    while attack_mask:
        to_sq, attack_mask = pop_lsb(attack_mask)
        moves.append((np.int8(from_sq), np.int8(to_sq), np.int8(0)))

    return moves

def generate_castling_moves(gs, is_white):
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
                    
    return moves

def generate_all_moves(gs):
    if gs.white_to_move:
        return generate_pawn_moves(gs, int(gs.white_pawns), True) \
        + generate_knight_moves(gs, int(gs.white_knights), True) \
        + generate_bishop_moves(gs, int(gs.white_bishops), True) \
        + generate_rook_moves(gs, int(gs.white_rooks), True) \
        + generate_queen_moves(gs, int(gs.white_queens), True) \
        + generate_king_moves(gs, int(gs.white_king), True) \
        + generate_castling_moves(gs, True)

    return generate_pawn_moves(gs, int(gs.black_pawns), False) \
    + generate_knight_moves(gs, int(gs.black_knights), False) \
    + generate_bishop_moves(gs, int(gs.black_bishops), False) \
    + generate_rook_moves(gs, int(gs.black_rooks), False) \
    + generate_queen_moves(gs, int(gs.black_queens), False) \
    + generate_king_moves(gs, int(gs.black_king), False) \
    + generate_castling_moves(gs, False)
        
def generate_all_legal_moves(gs):
    """Generate only *legal* moves (no king left in check)."""
    legal_moves = []
    side_to_move = gs.white_to_move

    for move in generate_all_moves(gs):
        print(move)
        gs.make_move(move)
        if not gs.is_in_check(side_to_move):
            legal_moves.append(move)
        
        gs.undo_move()


    return legal_moves

def index_to_square(index):
    """Convert a 0-63 index to a chessboard square in algebraic notation."""
    rank = (index // 8) + 1
    file = chr(index % 8 + ord('a'))
    return f"{file}{rank}"

def get_standard_algebraic(move):
    """Convert a list of moves from index notation to algebraic notation."""
    from_sq, to_sq, promo = move
    from_square_algebraic = index_to_square(from_sq)
    to_square_algebraic = index_to_square(to_sq)
    # Adding promotion notation if needed
    if promo != 0:
        promo_piece = {KNIGHT: 'N', BISHOP: 'B', ROOK: 'R', QUEEN: 'Q'}
        move_notation = f"{from_square_algebraic} {to_square_algebraic} {promo_piece}"
    else:
        move_notation = f"{from_square_algebraic} {to_square_algebraic}"
        
    return move_notation