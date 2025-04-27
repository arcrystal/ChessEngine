from numba import njit, types, int8
from numba.typed import List
import numpy as np

# numba tuple type: (from_row, from_col, to_row, to_col, promotion_piece)
MoveType = types.UniTuple(int8, 5)

# --- Constants ---
EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5, 6
MATERIAL_VALUES = np.array([0, 100, 320, 330, 500, 900, 20000])

# Movement tables
BISHOP_MOVES = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)], dtype=np.int8)
ROOK_MOVES = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int8)
QUEEN_MOVES = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int8)
KNIGHT_MOVES = np.array([(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)], dtype=np.int8)

@njit
def generate_pseudo_legal_moves(gs):
    moves = List.empty_list(MoveType)
    for from_row in range(8):
        for from_col in range(8):
            piece = gs.board[from_row, from_col]
            if piece == EMPTY or (piece > 0) != gs.white_to_move:
                continue
            abs_piece = abs(piece)

            if abs_piece == PAWN:
                add_pawn_moves(gs, from_row, from_col, moves)
            elif abs_piece == KNIGHT:
                add_knight_moves(gs, from_row, from_col, moves)
            elif abs_piece in (BISHOP, ROOK, QUEEN):
                directions = BISHOP_MOVES if abs_piece == BISHOP else ROOK_MOVES if abs_piece == ROOK else QUEEN_MOVES
                add_slider_moves(gs, from_row, from_col, moves, directions)
            elif abs_piece == KING:
                add_king_moves(gs, from_row, from_col, moves)
    return moves

@njit
def generate_legal_moves(gs):
    pseudo_moves = generate_pseudo_legal_moves(gs)
    legal_moves = List.empty_list(MoveType)

    for move in pseudo_moves:
        from_row, from_col, to_row, to_col, promotion_piece = move
        moving_piece = gs.board[from_row, from_col]
        captured_piece = gs.board[to_row, to_col]

        gs.board[to_row, to_col] = moving_piece
        gs.board[from_row, from_col] = EMPTY

        if not is_king_in_check(gs, gs.white_to_move):
            legal_moves.append(move)

        gs.board[from_row, from_col] = moving_piece
        gs.board[to_row, to_col] = captured_piece

    return legal_moves

@njit
def add_pawn_moves(gs, from_row, from_col, moves):
    piece = gs.board[from_row, from_col]
    direction = -1 if piece > 0 else 1
    start_row = 6 if piece > 0 else 1
    promotion_row = 0 if piece > 0 else 7
    to_row = from_row + direction

    if 0 <= to_row < 8:
        if gs.board[to_row, from_col] == EMPTY:
            if to_row == promotion_row:
                for promo in [QUEEN, ROOK, BISHOP, KNIGHT]:
                    moves.append((np.int8(from_row), np.int8(from_col), np.int8(to_row), np.int8(from_col), np.int8(promo)))
            else:
                moves.append((np.int8(from_row), np.int8(from_col), np.int8(to_row), np.int8(from_col), np.int8(0)))

            if from_row == start_row:
                to_row2 = from_row + 2 * direction
                if gs.board[to_row2, from_col] == EMPTY:
                    moves.append((np.int8(from_row), np.int8(from_col), np.int8(to_row2), np.int8(from_col), np.int8(0)))

    for dc in [-1, 1]:
        to_col = from_col + dc
        if 0 <= to_col < 8:
            target = gs.board[to_row, to_col]
            if target != EMPTY and (target > 0) != gs.white_to_move:
                if to_row == promotion_row:
                    for promo in [QUEEN, ROOK, BISHOP, KNIGHT]:
                        moves.append((np.int8(from_row), np.int8(from_col), np.int8(to_row), np.int8(to_col), np.int8(promo)))
                else:
                    moves.append((np.int8(from_row), np.int8(from_col), np.int8(to_row), np.int8(to_col), np.int8(0)))
            if (to_row, to_col) == (gs.en_passant_target[0], gs.en_passant_target[1]):
                moves.append((np.int8(from_row), np.int8(from_col), np.int8(to_row), np.int8(to_col), np.int8(0)))

@njit
def add_knight_moves(gs, from_row, from_col, moves):
    for i in range(KNIGHT_MOVES.shape[0]):
        dr, dc = KNIGHT_MOVES[i]
        to_row = from_row + dr
        to_col = from_col + dc
        if 0 <= to_row < 8 and 0 <= to_col < 8:
            target = gs.board[to_row, to_col]
            if target == EMPTY or (target > 0) != gs.white_to_move:
                moves.append((np.int8(from_row), np.int8(from_col), np.int8(to_row), np.int8(to_col), np.int8(0)))

@njit
def add_slider_moves(gs, from_row, from_col, moves, directions):
    for i in range(directions.shape[0]):
        dr, dc = directions[i]
        for step in range(1, 8):
            to_row = from_row + dr * step
            to_col = from_col + dc * step
            if not (0 <= to_row < 8 and 0 <= to_col < 8):
                break
            target = gs.board[to_row, to_col]
            if target == EMPTY:
                moves.append((np.int8(from_row), np.int8(from_col), np.int8(to_row), np.int8(to_col), np.int8(0)))
            elif (target > 0) != gs.white_to_move:
                moves.append((np.int8(from_row), np.int8(from_col), np.int8(to_row), np.int8(to_col), np.int8(0)))
                break
            else:
                break

@njit
def add_king_moves(gs, from_row, from_col, moves):
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            to_row = from_row + dr
            to_col = from_col + dc
            if 0 <= to_row < 8 and 0 <= to_col < 8:
                target = gs.board[to_row, to_col]
                if target == EMPTY or (target > 0) != gs.white_to_move:
                    moves.append((np.int8(from_row), np.int8(from_col), np.int8(to_row), np.int8(to_col), np.int8(0)))

    if can_castle(gs, True):
        moves.append((np.int8(from_row), np.int8(from_col), np.int8(from_row), np.int8(from_col + 2), np.int8(0)))
    if can_castle(gs, False):
        moves.append((np.int8(from_row), np.int8(from_col), np.int8(from_row), np.int8(from_col - 2), np.int8(0)))
        
@njit
def apply_move(gs, move):
    from_row, from_col, to_row, to_col, promotion_piece = move
    moving_piece = gs.board[from_row, from_col]
    captured_piece = gs.board[to_row, to_col]

    # Save pre-move state
    prev_en_passant = (gs.en_passant_target[0], gs.en_passant_target[1])
    prev_castling_rights = (gs.castling_rights[0], gs.castling_rights[1], gs.castling_rights[2], gs.castling_rights[3])
    prev_halfmove_clock = gs.halfmove_clock

    # Handle en passant capture
    if (to_row, to_col) == (gs.en_passant_target[0], gs.en_passant_target[1]) and abs(moving_piece) == PAWN:
        captured_row = from_row
        gs.board[captured_row, to_col] = EMPTY

    # Move the piece
    gs.board[to_row, to_col] = moving_piece
    gs.board[from_row, from_col] = EMPTY

    # Handle promotion
    if promotion_piece != 0:
        gs.board[to_row, to_col] = promotion_piece if gs.white_to_move else -promotion_piece

    # Castling move
    if abs(moving_piece) == KING and abs(from_col - to_col) == 2:
        if to_col == 6:  # Kingside
            rook_from_col = 7
            rook_to_col = 5
        else:  # Queenside
            rook_from_col = 0
            rook_to_col = 3
        rook_from_row = from_row
        gs.board[rook_from_row, rook_to_col] = gs.board[rook_from_row, rook_from_col]
        gs.board[rook_from_row, rook_from_col] = EMPTY

    # Update en passant target
    gs.en_passant_target[0] = -1
    gs.en_passant_target[1] = -1
    if abs(moving_piece) == PAWN and abs(to_row - from_row) == 2:
        gs.en_passant_target[0] = (to_row + from_row) // 2
        gs.en_passant_target[1] = from_col

    # Update castling rights
    if abs(moving_piece) == KING:
        if gs.white_to_move:
            gs.castling_rights[0] = 0
            gs.castling_rights[1] = 0
        else:
            gs.castling_rights[2] = 0
            gs.castling_rights[3] = 0
    elif abs(moving_piece) == ROOK:
        if gs.white_to_move:
            if from_col == 0: gs.castling_rights[1] = 0
            if from_col == 7: gs.castling_rights[0] = 0
        else:
            if from_col == 0: gs.castling_rights[3] = 0
            if from_col == 7: gs.castling_rights[2] = 0

    # Halfmove clock
    if abs(moving_piece) == PAWN or captured_piece != EMPTY:
        gs.halfmove_clock = 0
    else:
        gs.halfmove_clock += 1

    # Switch turn
    gs.switch_turn()

    # --- RETURN move information ---
    return moving_piece, captured_piece, prev_en_passant, prev_castling_rights, prev_halfmove_clock

@njit
def undo_move(gs, move, moving_piece, captured_piece, prev_en_passant, prev_castling_rights, prev_halfmove_clock):
    from_row, from_col, to_row, to_col, promotion_piece = move

    # Undo turn switch first
    gs.white_to_move = not gs.white_to_move
    if gs.white_to_move:
        gs.fullmove_number -= 1

    # Undo board
    if promotion_piece != 0:
        # If it was a promotion, revert promoted piece back to pawn
        gs.board[from_row, from_col] = PAWN if gs.white_to_move else -PAWN
    else:
        gs.board[from_row, from_col] = moving_piece

    # Restore captured piece
    gs.board[to_row, to_col] = captured_piece

    # Undo en passant capture (special case)
    if abs(moving_piece) == PAWN and (to_row, to_col) == (prev_en_passant[0], prev_en_passant[1]):
        captured_row = from_row
        gs.board[captured_row, to_col] = -PAWN if gs.white_to_move else PAWN
        gs.board[to_row, to_col] = EMPTY

    # Restore old en passant target
    gs.en_passant_target[0] = prev_en_passant[0]
    gs.en_passant_target[1] = prev_en_passant[1]

    # Restore castling rights
    gs.castling_rights[:] = prev_castling_rights[:]

    # Restore halfmove clock
    gs.halfmove_clock = prev_halfmove_clock
    
@njit
def is_square_attacked(gs, target_row, target_col, attacker_is_white):
    for i in range(KNIGHT_MOVES.shape[0]):
        dr, dc = KNIGHT_MOVES[i]
        r, c = target_row + dr, target_col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            piece = gs.board[r, c]
            if (piece > 0) == attacker_is_white and abs(piece) == KNIGHT:
                return True

    direction = -1 if attacker_is_white else 1
    for dc in [-1, 1]:
        r, c = target_row + direction, target_col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            piece = gs.board[r, c]
            if (piece > 0) == attacker_is_white and abs(piece) == PAWN:
                return True

    for i in range(QUEEN_MOVES.shape[0]):
        dr, dc = QUEEN_MOVES[i]
        for dist in range(1, 8):
            r, c = target_row + dr * dist, target_col + dc * dist
            if not (0 <= r < 8 and 0 <= c < 8):
                break
            piece = gs.board[r, c]
            if piece == EMPTY:
                continue
            if (piece > 0) == attacker_is_white:
                abs_piece = abs(piece)
                if (dr == 0 or dc == 0) and abs_piece in (ROOK, QUEEN):
                    return True
                if (dr != 0 and dc != 0) and abs_piece in (BISHOP, QUEEN):
                    return True
            break

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = target_row + dr, target_col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                piece = gs.board[r, c]
                if (piece > 0) == attacker_is_white and abs(piece) == KING:
                    return True

    return False
    
@njit
def is_king_in_check(gs, white_to_move):
    """Detect if side's king is in check after move."""
    king_row, king_col = -1, -1
    for r in range(8):
        for c in range(8):
            piece = gs.board[r, c]
            if (white_to_move and piece == 6) or (not white_to_move and piece == -6):
                king_row, king_col = r, c
                break

    if king_row == -1:
        return True  # king captured (invalid position)

    # Now check if opponent can attack king
    return is_square_attacked(gs, king_row, king_col, not white_to_move)

@njit
def can_castle(gs, kingside):
    rights = gs.castling_rights
    row = 7 if gs.white_to_move else 0
    king_side = rights[0] if gs.white_to_move else rights[2]
    queen_side = rights[1] if gs.white_to_move else rights[3]

    if kingside:
        if not king_side:
            return False
        if gs.board[row, 5] != EMPTY or gs.board[row, 6] != EMPTY:
            return False
        if (is_square_attacked(gs, row, 4, not gs.white_to_move) or
            is_square_attacked(gs, row, 5, not gs.white_to_move) or
            is_square_attacked(gs, row, 6, not gs.white_to_move)):
            return False
        return True
    else:
        if not queen_side:
            return False
        if gs.board[row, 1] != EMPTY or gs.board[row, 2] != EMPTY or gs.board[row, 3] != EMPTY:
            return False
        if (is_square_attacked(gs, row, 4, not gs.white_to_move) or
            is_square_attacked(gs, row, 3, not gs.white_to_move) or
            is_square_attacked(gs, row, 2, not gs.white_to_move)):
            return False
        return True