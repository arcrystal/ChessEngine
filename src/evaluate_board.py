import numpy as np
from numba import njit
from engine_utils import KNIGHT_MOVES, QUEEN_MOVES, ROOK_MOVES, BISHOP_MOVES

# Pawn, Bishop, Knight, Rook, Queen, King PST (opening and endgame)
PAWN_PST_OPENING = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.int16)

PAWN_PST_ENDGAME = np.array([
    [0, 5, 5, 0, 0, 5, 5, 0],
    [10, 10, 10, 10, 10, 10, 10, 10],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [20, 20, 30, 40, 40, 30, 20, 20],
    [30, 30, 40, 50, 50, 40, 30, 30],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.int16)

KNIGHT_PST_OPENING = np.array([
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
], dtype=np.int16)

KNIGHT_PST_ENDGAME = KNIGHT_PST_OPENING

BISHOP_PST_OPENING = np.array([
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
], dtype=np.int16)

BISHOP_PST_ENDGAME = BISHOP_PST_OPENING

ROOK_PST_OPENING = np.array([
    [0, 0, 5, 10, 10, 5, 0, 0],
    [0, 0, 5, 10, 10, 5, 0, 0],
    [0, 0, 5, 10, 10, 5, 0, 0],
    [0, 0, 5, 10, 10, 5, 0, 0],
    [0, 0, 5, 10, 10, 5, 0, 0],
    [0, 0, 5, 10, 10, 5, 0, 0],
    [25, 25, 25, 25, 25, 25, 25, 25],
    [0, 0, 5, 10, 10, 5, 0, 0]
], dtype=np.int16)

ROOK_PST_ENDGAME = np.array([
    [0, 0, 5, 10, 10, 5, 0, 0],
    [5, 10, 10, 20, 20, 10, 10, 5],
    [5, 10, 10, 20, 20, 10, 10, 5],
    [5, 10, 10, 20, 20, 10, 10, 5],
    [5, 10, 10, 20, 20, 10, 10, 5],
    [5, 10, 10, 20, 20, 10, 10, 5],
    [5, 10, 10, 20, 20, 10, 10, 5],
    [5, 10, 10, 20, 20, 10, 10, 5]
], dtype=np.int16)

QUEEN_PST_OPENING = np.array([
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20]
], dtype=np.int16)

QUEEN_PST_ENDGAME = QUEEN_PST_OPENING

KING_PST_OPENING = np.array([
    [20, 30, 10, 0, 0, 10, 30, 20],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30]
], dtype=np.int16)

KING_PST_ENDGAME = np.array([
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10, 0, 0, -10, -20, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -30, 0, 0, 0, 0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50]
], dtype=np.int16)

# --- Basic Definitions ---
EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5, 6

# --- Piece Values ---
MATERIAL_VALUES = np.array([0, 100, 320, 330, 500, 900, 20000], dtype=np.int16)

# --- Phase Values for Scaling ---
PHASE_VALUES = np.array([0, 1, 1, 2, 4, 0, 0], dtype=np.int16)
PHASE_MATERIAL_THRESHOLD = 20000
MIDGAME_WEIGHT = 1.0
ENDGAME_WEIGHT = 1.2

# --- Strategic Bonuses and Penalties ---
MOBILITY_BONUS = 5
KING_SAFETY_PENALTY = 10
PASSED_PAWN_BONUS = 30
CONNECTED_PASSED_PAWN_BONUS = 50
ISOLATED_PAWN_PENALTY = -20
DOUBLED_PAWN_PENALTY = -15
ROOK_OPEN_FILE_BONUS = 20
ROOK_SEMI_OPEN_FILE_BONUS = 10
BISHOP_PAIR_BONUS = 30
BAD_BISHOP_PENALTY = 15
COORDINATION_BONUS = 10
CENTER_CONTROL_BONUS = 3
KNIGHT_OUTPOST_BONUS = 30
SUPERFLUOUS_KNIGHTS_PENALTY = 15
TEMPO_BONUS = 10

@njit
def evaluate_board(gs):
    board = gs.board
    score = 0
    white_material, black_material = 0, 0
    phase = 0
    white_pawn_structure = np.zeros(8, dtype=np.int16)
    black_pawn_structure = np.zeros(8, dtype=np.int16)

    for r in range(8):
        for c in range(8):
            piece = board[r, c]
            if piece == EMPTY:
                continue
            side = 1 if piece > 0 else -1
            abs_piece = abs(piece)

            value = MATERIAL_VALUES[abs_piece]
            if side == 1:
                white_material += value
            else:
                black_material += value

            # PSTs
            if abs_piece == PAWN:
                if side == 1:
                    if phase < PHASE_MATERIAL_THRESHOLD:
                        score += PAWN_PST_OPENING[7 - r, c]
                    else:
                        score += PAWN_PST_ENDGAME[7 - r, c]
                    white_pawn_structure[c] += 1
                else:
                    if phase < PHASE_MATERIAL_THRESHOLD:
                        score -= PAWN_PST_OPENING[r, c]
                    else:
                        score -= PAWN_PST_ENDGAME[r, c]
                    black_pawn_structure[c] += 1
            elif abs_piece == KNIGHT:
                score += side * KNIGHT_PST_OPENING[7 - r, c]
            elif abs_piece == BISHOP:
                score += side * BISHOP_PST_OPENING[7 - r, c]
            elif abs_piece == ROOK:
                if phase < PHASE_MATERIAL_THRESHOLD:
                    score += side * ROOK_PST_ENDGAME[7 - r, c]
                else:
                    score += side * ROOK_PST_OPENING[7 - r, c]
            elif abs_piece == QUEEN:
                score += side * QUEEN_PST_OPENING[7 - r, c]
            elif abs_piece == KING:
                if phase < PHASE_MATERIAL_THRESHOLD:
                    score += side * KING_PST_ENDGAME[7 - r, c]
                else:
                    score += side * KING_PST_OPENING[7 - r, c]

            # Phase accumulation
            phase += PHASE_VALUES[abs_piece]

    # --- Pawn structure ---
    score += evaluate_pawn_structure(white_pawn_structure) - evaluate_pawn_structure(black_pawn_structure)

    # --- Bishop Pair ---
    score += bishop_pair_bonus(gs, True) - bishop_pair_bonus(gs, False)

    # --- Rooks Open/Semi-Open Files ---
    score += rook_open_files(gs, True) - rook_open_files(gs, False)

    # --- Mobility ---
    score += mobility_score(gs, True) - mobility_score(gs, False)

    # --- Passed and Connected Pawns ---
    score += passed_pawns_bonus(gs, True) - passed_pawns_bonus(gs, False)
    
    score += connected_passed_pawns_bonus(gs, True) - connected_passed_pawns_bonus(gs, False)

    # --- Attack King Zones ---
    score += attack_king_zone(gs, True) - attack_king_zone(gs, False)

    # --- Coordination ---
    score += coordination_bonus(gs, True) - coordination_bonus(gs, False)

    # --- Center Control ---
    score += center_control_bonus(gs, True) - center_control_bonus(gs, False)

    # --- Knight Outposts ---
    score += knight_outpost_bonus(gs, True) - knight_outpost_bonus(gs, False)

    # --- Bad Bishops ---
    score += bad_bishop_penalty(gs, False) - bad_bishop_penalty(gs, True)

    # --- Superfluous Knights ---
    score += superfluous_knights_penalty(gs, False) - superfluous_knights_penalty(gs, True)

    # --- King Safety ---
    score += king_safety_penalty(gs, False) - king_safety_penalty(gs, True)

    # --- Tempo ---
    score += TEMPO_BONUS if gs.white_to_move else -TEMPO_BONUS

    # --- Material balance ---
    score += white_material - black_material

    # --- Final Scaling ---
    score *= ENDGAME_WEIGHT if phase < PHASE_MATERIAL_THRESHOLD else MIDGAME_WEIGHT

    return np.int16(score)


@njit
def evaluate_pawn_structure(pawn_structure):
    score = 0
    for c in range(8):
        if pawn_structure[c] > 1:
            score += DOUBLED_PAWN_PENALTY
        if pawn_structure[c] == 1:
            if (c == 0 or pawn_structure[c-1] == 0) and (c == 7 or pawn_structure[c+1] == 0):
                score += ISOLATED_PAWN_PENALTY
    return np.int16(score)

@njit
def rook_open_files(gs, white_to_move):
    board = gs.board
    bonus = 0
    for c in range(8):
        file_empty = True
        for r in range(8):
            if abs(board[r, c]) == PAWN:
                file_empty = False
                break
        if file_empty:
            for r in range(8):
                if board[r, c] == (ROOK if white_to_move else -ROOK):
                    bonus += ROOK_OPEN_FILE_BONUS
    return np.int16(bonus)

@njit
def passed_pawn_bonus(gs, white_to_move):
    board = gs.board
    bonus = 0
    direction = -1 if white_to_move else 1
    for r in range(8):
        for c in range(8):
            piece = board[r, c]
            if piece == (PAWN if white_to_move else -PAWN):
                passed = True
                for dr in range(r + direction, 8 if white_to_move else -1, direction):
                    for dc in (c-1, c, c+1):
                        if 0 <= dc < 8:
                            if board[dr, dc] == (-PAWN if white_to_move else PAWN):
                                passed = False
                                break
                    if not passed:
                        break
                if passed:
                    advance = (7 - r) if white_to_move else r
                    bonus += PASSED_PAWN_BONUS + (advance * 5)
    return np.int16(bonus)

@njit
def attack_king_zone(gs, white_to_move):
    board = gs.board
    king_piece = KING if white_to_move else -KING
    bonus = 0

    # Locate king
    king_row, king_col = -1, -1
    for r in range(8):
        for c in range(8):
            if board[r, c] == king_piece:
                king_row, king_col = r, c
                break
        if king_row != -1:
            break

    if king_row == -1:
        return 0

    # Check surrounding squares
    danger_zone = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = king_row + dr, king_col + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                danger_zone.append((nr, nc))

    # Count how many attackers
    for r, c in danger_zone:
        piece = board[r, c]
        if (piece > 0) != white_to_move and piece != EMPTY:
            piece_type = abs(piece)
            if piece_type == QUEEN:
                bonus += 50
            elif piece_type == ROOK:
                bonus += 30
            elif piece_type == BISHOP or piece_type == KNIGHT:
                bonus += 20
            elif piece_type == PAWN:
                bonus += 10

    return np.int16(bonus)

@njit
def pseudo_pawn_moves(gs, from_row, from_col, white_to_move):
    board = gs.board
    count = 0
    direction = -1 if white_to_move else 1
    start_row = 6 if white_to_move else 1
    promotion_row = 0 if white_to_move else 7
    to_row = from_row + direction

    if 0 <= to_row < 8:
        if board[to_row, from_col] == EMPTY:
            count += 1
            if from_row == start_row:
                to_row2 = from_row + 2 * direction
                if board[to_row2, from_col] == EMPTY:
                    count += 1

        for dc in (-1, 1):
            to_col = from_col + dc
            if 0 <= to_col < 8:
                target = board[to_row, to_col]
                if target != EMPTY and (target > 0) != white_to_move:
                    count += 1
                if (to_row, to_col) == (gs.en_passant_target[0], gs.en_passant_target[1]):
                    count += 1
    return count

@njit
def pseudo_knight_moves(gs, from_row, from_col, white_to_move):
    board = gs.board
    count = 0
    for i in range(KNIGHT_MOVES.shape[0]):
        dr, dc = KNIGHT_MOVES[i]
        to_row = from_row + dr
        to_col = from_col + dc
        if 0 <= to_row < 8 and 0 <= to_col < 8:
            target = board[to_row, to_col]
            if target == EMPTY or (target > 0) != white_to_move:
                count += 1
    return count

@njit
def pseudo_slider_moves(gs, from_row, from_col, white_to_move, directions):
    board = gs.board
    count = 0
    for i in range(directions.shape[0]):
        dr, dc = directions[i]
        for step in range(1, 8):
            to_row = from_row + dr * step
            to_col = from_col + dc * step
            if not (0 <= to_row < 8 and 0 <= to_col < 8):
                break
            target = board[to_row, to_col]
            if target == EMPTY:
                count += 1
            elif (target > 0) != white_to_move:
                count += 1
                break
            else:
                break
    return count

@njit
def pseudo_king_moves(gs, from_row, from_col, white_to_move):
    board = gs.board
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            to_row = from_row + dr
            to_col = from_col + dc
            if 0 <= to_row < 8 and 0 <= to_col < 8:
                target = board[to_row, to_col]
                if target == EMPTY or (target > 0) != white_to_move:
                    count += 1
    return count


@njit
def mobility_score(gs, white_to_move):
    board = gs.board
    score = 0
    for from_row in range(8):
        for from_col in range(8):
            piece = board[from_row, from_col]
            if piece == EMPTY or (piece > 0) != white_to_move:
                continue
            abs_piece = abs(piece)

            if abs_piece == PAWN:
                score += pseudo_pawn_moves(gs, from_row, from_col, white_to_move)
            elif abs_piece == KNIGHT:
                score += pseudo_knight_moves(gs, from_row, from_col, white_to_move)
            elif abs_piece == BISHOP:
                score += pseudo_slider_moves(gs, from_row, from_col, white_to_move, BISHOP_MOVES)
            elif abs_piece == ROOK:
                score += pseudo_slider_moves(gs, from_row, from_col, white_to_move, ROOK_MOVES)
            elif abs_piece == QUEEN:
                score += pseudo_slider_moves(gs, from_row, from_col, white_to_move, QUEEN_MOVES)
            elif abs_piece == KING:
                score += pseudo_king_moves(gs, from_row, from_col, white_to_move)

    return np.int16(score * MOBILITY_BONUS)

@njit
def king_safety_penalty(gs, white_to_move):
    board = gs.board
    king_piece = KING if white_to_move else -KING
    row = 7 if white_to_move else 0
    king_row, king_col = -1, -1
    for r in range(8):
        for c in range(8):
            if board[r, c] == king_piece:
                king_row, king_col = r, c
                break
        if king_row != -1:
            break

    if king_row == -1:
        return 0

    penalty = 0
    forward = -1 if white_to_move else 1
    shield_rows = [king_row + forward * i for i in range(1, 3)]

    for sr in shield_rows:
        if 0 <= sr < 8:
            for dc in [-1, 0, 1]:
                sc = king_col + dc
                if 0 <= sc < 8:
                    if board[sr, sc] != (PAWN if white_to_move else -PAWN):
                        penalty += KING_SAFETY_PENALTY
    return np.int16(penalty)

@njit
def passed_pawns_bonus(gs, white_to_move):
    board = gs.board
    count = 0
    for col in range(8):
        for row in (range(6, -1, -1) if white_to_move else range(1, 8)):
            piece = board[row, col]
            if (piece == PAWN if white_to_move else piece == -PAWN):
                blocked = False
                for dr in range(1, 8):
                    r = row - dr if white_to_move else row + dr
                    if 0 <= r < 8:
                        for dc in [-1, 0, 1]:
                            c = col + dc
                            if 0 <= c < 8:
                                enemy = board[r, c]
                                if (enemy > 0) != white_to_move and abs(enemy) == PAWN:
                                    blocked = True
                if not blocked:
                    count += 1
    return np.int16(count * PASSED_PAWN_BONUS)

@njit
def connected_passed_pawns_bonus(gs, white_to_move):
    board = gs.board
    count = 0
    direction = -1 if white_to_move else 1
    passed = np.zeros(8, dtype=np.bool_)

    for r in range(8):
        for c in range(8):
            piece = board[r, c]
            if piece == (PAWN if white_to_move else -PAWN):
                is_passed = True
                for dr in range(r + direction, 8 if white_to_move else -1, direction):
                    for dc in (c-1, c, c+1):
                        if 0 <= dc < 8:
                            if board[dr, dc] == (-PAWN if white_to_move else PAWN):
                                is_passed = False
                                break
                    if not is_passed:
                        break
                if is_passed:
                    passed[c] = True

    for c in range(7):
        if passed[c] and passed[c+1]:
            count += 1

    return np.int16(count * CONNECTED_PASSED_PAWN_BONUS)

@njit
def coordination_bonus(gs, white_to_move):
    # Very simple: count double attacks
    board = gs.board
    attack_map = np.zeros((8,8), dtype=np.int16)
    score = 0

    for from_row in range(8):
        for from_col in range(8):
            piece = board[from_row, from_col]
            if piece == EMPTY or (piece > 0) != white_to_move:
                continue
            abs_piece = abs(piece)

            # Mark attacked squares
            if abs_piece == KNIGHT:
                for i in range(KNIGHT_MOVES.shape[0]):
                    dr, dc = KNIGHT_MOVES[i]
                    r, c = from_row + dr, from_col + dc
                    if 0 <= r < 8 and 0 <= c < 8:
                        attack_map[r, c] += 1
            elif abs_piece in (BISHOP, ROOK, QUEEN):
                directions = BISHOP_MOVES if abs_piece == BISHOP else ROOK_MOVES if abs_piece == ROOK else QUEEN_MOVES
                for i in range(directions.shape[0]):
                    dr, dc = directions[i]
                    for step in range(1, 8):
                        r, c = from_row + dr*step, from_col + dc*step
                        if not (0 <= r < 8 and 0 <= c < 8):
                            break
                        attack_map[r, c] += 1
                        if board[r, c] != EMPTY:
                            break

    # Bonus for squares attacked by multiple pieces
    for r in range(8):
        for c in range(8):
            if attack_map[r, c] > 1:
                score += 5 * (attack_map[r, c] - 1)

    return np.int16(score * COORDINATION_BONUS)

@njit
def center_control_bonus(gs, white_to_move):
    board = gs.board
    bonus = 0
    centers = [(3, 3), (3, 4), (4, 3), (4, 4)]
    for r, c in centers:
        piece = board[r, c]
        if piece != EMPTY and (piece > 0) == white_to_move:
            bonus += CENTER_CONTROL_BONUS
            
    return np.int16(bonus)

@njit
def rooks_open_files(gs, white_to_move):
    board = gs.board
    bonus = 0
    for col in range(8):
        has_friendly_pawn = False
        has_enemy_pawn = False
        for row in range(8):
            piece = board[row, col]
            if piece == (PAWN if white_to_move else -PAWN):
                has_friendly_pawn = True
            elif piece == (-PAWN if white_to_move else PAWN):
                has_enemy_pawn = True

        for row in range(8):
            piece = board[row, col]
            if piece == (ROOK if white_to_move else -ROOK):
                if not has_friendly_pawn and not has_enemy_pawn:
                    bonus += ROOK_OPEN_FILE_BONUS
                elif not has_friendly_pawn:
                    bonus += ROOK_SEMI_OPEN_FILE_BONUS
    return np.int16(bonus)

@njit
def knight_outpost_bonus(gs, white_to_move):
    board = gs.board
    knight_outposts = 0
    for r in range(8):
        for c in range(8):
            piece = board[r, c]
            if piece == (KNIGHT if white_to_move else -KNIGHT):
                if ((white_to_move and r <= 4) or (not white_to_move and r >= 3)):  # advanced
                    # Check supporting pawn
                    if white_to_move and r+1 < 8:
                        if (c-1 >= 0 and board[r+1, c-1] == PAWN) or (c+1 < 8 and board[r+1, c+1] == PAWN):
                            knight_outposts += 1
                    if not white_to_move and r-1 >= 0:
                        if (c-1 >= 0 and board[r-1, c-1] == -PAWN) or (c+1 < 8 and board[r-1, c+1] == -PAWN):
                            knight_outposts += 1
    return np.int16(knight_outposts * KNIGHT_OUTPOST_BONUS)

@njit
def superfluous_knights_penalty(gs, white_to_move):
    board = gs.board
    penalty = 0
    for r in range(8):
        for c in range(8):
            if board[r, c] == (KNIGHT if white_to_move else -KNIGHT):
                knight_neighbors = 0
                for dr, dc in KNIGHT_MOVES:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        if board[rr, cc] == (KNIGHT if white_to_move else -KNIGHT):
                            knight_neighbors += 1
                if knight_neighbors > 0:
                    penalty += SUPERFLUOUS_KNIGHTS_PENALTY

    return np.int16(penalty)

@njit
def bad_bishop_penalty(gs, white_to_move):
    board = gs.board
    num_bad = 0
    for r in range(8):
        for c in range(8):
            piece = board[r, c]
            if piece == (BISHOP if white_to_move else -BISHOP):
                color = (r + c) % 2
                for dr in [-1, 1]:
                    for dc in [-1, 1]:
                        for i in range(1, 8):
                            nr = r + dr*i
                            nc = c + dc*i
                            if 0 <= nr < 8 and 0 <= nc < 8:
                                blocker = board[nr, nc]
                                if (blocker == PAWN if white_to_move else blocker == -PAWN):
                                    num_bad += 1
                                    break
                            else:
                                break
    return np.int16(num_bad * BAD_BISHOP_PENALTY)

@njit
def bishop_pair_bonus(gs, white_to_move):
    board = gs.board
    num_bishops = 0
    for r in range(8):
        for c in range(8):
            piece = board[r, c]
            if piece == (BISHOP if white_to_move else -BISHOP):
                num_bishops += 1
                
    return np.int16(BISHOP_PAIR_BONUS if num_bishops==2 else 0)