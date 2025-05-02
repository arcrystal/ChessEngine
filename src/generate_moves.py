from numba import njit, uint64

from bitboard_nomagic import knight_attacks, king_attacks
from bitboard_magic import bishop_attacks, rook_attacks, queen_attacks
from constants import KNIGHT, BISHOP, ROOK, QUEEN
from bitboard_utils import pop_lsb, rank_mask, file_mask
from bitboard_gamestate_utils import attack_map_numba


@njit
def generate_pawn_moves(gs, pawns, white_to_move):
    moves = []
    empty = ~gs.occupied
    enemy = gs.black_occupancy if white_to_move else gs.white_occupancy

    if white_to_move:
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

    # --- Single pushes ---
    temp = uint64(single_push)
    while temp:
        to_sq, temp = pop_lsb(temp)
        from_sq = to_sq - direction
        if from_sq // 8 == promo_rank:
            for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                moves.append((from_sq, to_sq, promo))
        else:
            moves.append((from_sq, to_sq, 0))

    # --- Double pushes ---
    temp = uint64(double_push)
    while temp:
        to_sq, temp = pop_lsb(temp)
        from_sq = to_sq - 2 * direction
        moves.append((from_sq, to_sq, 0))

    # --- Left captures ---
    temp = uint64(left_attacks)
    while temp:
        to_sq, temp = pop_lsb(temp)
        from_sq = to_sq - (7 if white_to_move else -9)
        if from_sq // 8 == promo_rank:
            for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                moves.append((from_sq, to_sq, promo))
        else:
            moves.append((from_sq, to_sq, 0))

    # --- Right captures ---
    temp = uint64(right_attacks)
    while temp:
        to_sq, temp = pop_lsb(temp)
        from_sq = to_sq - (9 if white_to_move else -7)
        if from_sq // 8 == promo_rank:
            for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                moves.append((from_sq, to_sq, promo))
        else:
            moves.append((from_sq, to_sq, 0))

    # --- En Passant ---
    if gs.en_passant_target != -1:
        ep_sq = gs.en_passant_target
        ep_file = ep_sq % 8
        ep_rank_bb = rank_mask(ep_rank)
        pawns_on_rank = pawns & ep_rank_bb

        temp = uint64(pawns_on_rank)
        while temp:
            from_sq, temp = pop_lsb(temp)
            from_file = from_sq % 8
            if abs(from_file - ep_file) == 1:
                if white_to_move and (from_sq + 7 == ep_sq or from_sq + 9 == ep_sq):
                    moves.append((from_sq, (ep_sq), 0))
                elif not white_to_move and (from_sq - 9 == ep_sq or from_sq - 7 == ep_sq):
                    moves.append((from_sq, (ep_sq), 0))

    return moves

@njit
def generate_knight_moves(gs, knights, white_to_move):
    moves = []
    own_pieces = gs.white_occupancy if white_to_move else gs.black_occupancy
    temp = uint64(knights)
    while temp:
        from_sq, temp = pop_lsb(temp)
        attacks = knight_attacks(from_sq) & ~own_pieces
        while attacks:
            to_sq, attacks = pop_lsb(attacks)
            moves.append((from_sq, to_sq, 0))
    return moves

@njit
def generate_king_moves(gs, king_bb, white_to_move):
    moves = []
    from_sq = pop_lsb(uint64(king_bb))[0]
    own_pieces = gs.white_occupancy if white_to_move else gs.black_occupancy
    enemy_attacks = attack_map_numba(gs, not white_to_move)
    legal = king_attacks(from_sq) & ~own_pieces & ~enemy_attacks
    temp = uint64(legal)
    while temp:
        to_sq, temp = pop_lsb(temp)
        moves.append((from_sq, to_sq, 0))
    return moves

@njit
def generate_bishop_moves(gs, bishops, white_to_move):
    moves = []
    own_pieces = gs.white_occupancy if white_to_move else gs.black_occupancy
    temp = uint64(bishops)
    while temp:
        from_sq, temp = pop_lsb(temp)
        attacks = bishop_attacks(from_sq, gs.occupied) & ~own_pieces
        while attacks:
            to_sq, attacks = pop_lsb(attacks)
            moves.append((from_sq, to_sq, 0))
    return moves

@njit
def generate_rook_moves(gs, rooks, white_to_move):
    moves = []
    own_pieces = gs.white_occupancy if white_to_move else gs.black_occupancy
    temp = uint64(rooks)
    while temp:
        from_sq, temp = pop_lsb(temp)
        attacks = rook_attacks(from_sq, gs.occupied) & ~own_pieces
        while attacks:
            to_sq, attacks = pop_lsb(attacks)
            moves.append((from_sq, to_sq, 0))
    return moves

@njit
def generate_queen_moves(gs, queens, white_to_move):
    moves = []
    own_pieces = gs.white_occupancy if white_to_move else gs.black_occupancy
    temp = uint64(queens)
    while temp:
        from_sq, temp = pop_lsb(temp)
        attacks = queen_attacks(from_sq, gs.occupied) & ~own_pieces
        while attacks:
            to_sq, attacks = pop_lsb(attacks)
            moves.append((from_sq, to_sq, 0))
    return moves

@njit
def generate_all_moves(gs):
    if gs.white_to_move:
        return (
            generate_pawn_moves(gs, gs.white_pawns, True)
            + generate_knight_moves(gs, gs.white_knights, True)
            + generate_bishop_moves(gs, gs.white_bishops, True)
            + generate_rook_moves(gs, gs.white_rooks, True)
            + generate_queen_moves(gs, gs.white_queens, True)
            + generate_king_moves(gs, gs.white_king, True)
        )
    else:
        return (
            generate_pawn_moves(gs, gs.black_pawns, False)
            + generate_knight_moves(gs, gs.black_knights, False)
            + generate_bishop_moves(gs, gs.black_bishops, False)
            + generate_rook_moves(gs, gs.black_rooks, False)
            + generate_queen_moves(gs, gs.black_queens, False)
            + generate_king_moves(gs, gs.black_king, False)
        )
