import pytest
import numpy as np
from game import GameState, EMPTY, ROOK, KING
from engine_utils import generate_legal_moves, apply_move

@pytest.fixture
def new_game():
    return GameState()

def move(from_row, from_col, to_row, to_col, promotion=0):
    return (np.int8(from_row), np.int8(from_col), np.int8(to_row), np.int8(to_col), np.int8(promotion))

def simulate_moves(game, moves):
    for move in moves:
        apply_move(game, (move[0], move[1], move[2], move[3], 0))

def legal_moves(game):
    return generate_legal_moves(game)

# --- Success Castling Tests ---

def test_kingside_castle_success(new_game):
    # e2 e4, e7 e5
    # f1 c4, f8 c5
    # g1 f3, g8 g6
    simulate_moves(new_game, [
        (6,4,4,4), (1,4,3,4),
        (7,5,4,2), (0,5,3,2),
        (7,6,5,5), (0,6,2,6),
    ])

    moves = legal_moves(new_game)
    # White: e1 g1
    assert move(7,4,7,6,0) in moves, "White kingside castling move missing!"
    simulate_moves(new_game, [(7,4,7,6)])

    # Check White rook position after castling
    assert new_game.board[7,5] == ROOK, "White rook did not move to f1 after castling!"
    assert new_game.board[7,7] == EMPTY, "White rook did not leave original square!"

    moves = legal_moves(new_game)
    # Black: e8 g8
    assert move(0,4,0,6,0) in moves, "Black kingside castling move missing!"
    simulate_moves(new_game, [(0,4,0,6)])

    # Check Black rook position after castling
    assert new_game.board[0,5] == -ROOK, "Black rook did not move to f8 after castling!"
    assert new_game.board[0,7] == EMPTY, "Black rook did not leave original square!"

def test_queenside_castle_success(new_game):
    # d2 d4, d7 d5
    # c1 c4, c8 c5
    # b1 c3, b8 b6
    # d1 d2, d8 d7
    simulate_moves(new_game, [
        (6,3,4,3), (1,3,3,3),
        (7,2,4,2), (0,2,3,2),
        (7,1,5,2), (0,1,2,1),
        (7,3,6,3), (0,3,1,3),
    ])

    moves = legal_moves(new_game)

    # White: e1 c1
    assert move(7,4,7,2,0) in moves, "White queenside castling move missing!"
    simulate_moves(new_game, [(7,4,7,2)])

    # Check White rook position after castling
    assert new_game.board[7,3] == ROOK, "White rook did not move to d1 after queenside castling!"
    assert new_game.board[7,0] == 0, "White rook did not leave a1 after queenside castling!"

    moves = legal_moves(new_game)

    # Black: e8 c8
    assert move(0,4,0,2,0) in moves, "Black queenside castling move missing!"
    simulate_moves(new_game, [(0,4,0,2)])

    # Check Black rook position after castling
    assert new_game.board[0,3] == -ROOK, "Black rook did not move to d8 after queenside castling!"
    assert new_game.board[0,0] == 0, "Black rook did not leave a8 after queenside castling!"

# # --- Failure Castling Tests ---

def test_kingside_castle_after_king_move(new_game):
    # e2 e4, e7 e5
    # f1 c4, f8 c5
    # g1 f3, g8 g6
    # e1 f1, e8 f8
    # f1 e1, f8 e8
    simulate_moves(new_game, [
        (6, 4, 4, 4), (1, 4, 3, 4),
        (7, 5, 4, 2), (0, 5, 3, 2),
        (7, 6, 5, 5), (0, 6, 2, 6),
        (7, 4, 7, 5), (0, 4, 0, 5),
        (7, 5, 7, 4), (0, 5, 0, 4),
    ])
    moves = legal_moves(new_game)
    # White: e1 g1
    assert move(7,4,7,6,0) not in moves, "White kingside castling should not be allowed!"
    # Black: e8 g8
    assert move(0,4,0,6,0) not in moves, "Black kingside castling should not be allowed!"

def test_queenside_castle_after_king_move(new_game):
    # d2 d4, d7 d5
    # c1 c4, c8 c5
    # b1 c3, b8 b6
    # d1 d2, d8 d7
    # e1 d1, e8 d8
    # d1 e1, d8 e8
    simulate_moves(new_game, [
        (6, 3, 4, 3), (1, 3, 3, 3),
        (7, 2, 4, 2), (0, 2, 3, 2),
        (7, 1, 5, 2), (0, 1, 2, 1),
        (7, 3, 6, 3), (0, 3, 1, 3),
        (7, 4, 7, 3), (0, 4, 0, 3),
        (7, 3, 7, 4), (0, 3, 0, 4),
    ])
    moves = legal_moves(new_game)
    # White: e1 c1
    assert move(7,4,7,2,0) not in moves, "White queenside castling should not be allowed!"
    # Black: e8 c8
    assert move(0,4,0,2,0) not in moves, "Black queenside castling should not be allowed!"
    
def test_kingside_castle_after_rook_move(new_game):
    # e2 e4, e7 e5
    # f1 c4, f8 c5
    # g1 f3, g8 g6
    # h1 g1, h8 g8
    # g1 h1, g8 h8
    simulate_moves(new_game, [
        (6, 4, 4, 4), (1, 4, 3, 4),
        (7, 5, 4, 2), (0, 5, 3, 2),
        (7, 6, 5, 5), (0, 6, 2, 6),
        (7, 7, 7, 6), (0, 7, 0, 6),
        (7, 6, 7, 7), (0, 6, 0, 7),
    ])
    moves = legal_moves(new_game)
    # White: e1 g1
    assert move(7,4,7,6,0) not in moves, "White kingside castling should not be allowed!"
    # Black: e8 g8
    assert move(0,4,0,6,0) not in moves, "Black kingside castling should not be allowed!"

def test_queenside_castle_after_rook_move(new_game):
    # d2 d4, d7 d5
    # c1 c4, c8 c5
    # b1 c3, b8 b6
    # d1 d2, d8 d7
    # a1 b1, a8 b8
    # b1 a1, b8 a8
    simulate_moves(new_game, [
        (6, 3, 4, 3), (1, 3, 3, 3),
        (7, 2, 4, 2), (0, 2, 3, 2),
        (7, 1, 5, 2), (0, 1, 2, 1),
        (7, 3, 6, 3), (0, 3, 1, 3),
        (7, 0, 7, 1), (0, 0, 0, 1),
        (7, 1, 7, 0), (0, 1, 0, 0),
    ])
    moves = legal_moves(new_game)
    # White: e1 c1
    assert move(7,4,7,2,0) not in moves, "White queenside castling should not be allowed!"
    # Black: e8 c8
    assert move(0,4,0,2,0) not in moves, "Black queenside castling should not be allowed!"

def test_cannot_castle_through_check(new_game):
    # e2 e4, e7 e5
    # f1 c4, f8 c5
    # g1 h3, g8 h6
    # f2 f3, f7 f6
    simulate_moves(new_game, [
        (6, 4, 4, 4), (1, 4, 3, 4),
        (7, 5, 4, 2), (0, 5, 3, 2),
        (7, 6, 5, 7), (0, 6, 2, 7),
        (6, 5, 5, 5), (1, 5, 2, 5),
    ])
    moves = legal_moves(new_game)
    # White: e1 g1
    assert move(7,4,7,6,0) not in moves, "White kingside castling through check not allowed!"
    # Black: e8 g8
    assert move(0,4,0,6,0) not in moves, "Black kingside castling through check not allowed!"

def test_cannot_castle_into_check(new_game):
    # e2 e4, e7 e5
    # f1 c4, f8 c5
    # g1 h3, g8 h6
    # d1 h5, d8 h4
    # g2 g3, g7 g6
    # h5 g6, h4 g3
    simulate_moves(new_game, [
        (6, 4, 4, 4), (1, 4, 3, 4),
        (7, 5, 4, 2), (0, 5, 3, 2),
        (7, 6, 5, 7), (0, 6, 2, 7),
        (7, 3, 3, 7), (0, 3, 4, 7),
        (6, 6, 5, 6), (1, 6, 2, 6),
        (3, 7, 2, 6), (4, 7, 5, 6),
    ])
    moves = legal_moves(new_game)
    # White: e1 g1
    assert move(7,4,7,6,0) not in moves, "White kingside castling into check not allowed!"
    # Black: e8 g8
    assert move(0,4,0,6,0) not in moves, "Black kingside castling into check not allowed!"

def test_cannot_castle_through_piece(new_game):
    # e2 e3, e7 e6
    # b2 b3, b7 b6
    # d1 f3, d8 f6
    # c1 b2, c8 b7
    simulate_moves(new_game, [
        (6, 4, 5, 4), (1, 4, 2, 4),
        (6, 1, 5, 1), (1, 1, 2, 1),
        (7, 3, 5, 5), (0, 3, 2, 5),
        (7, 2, 6, 1), (0, 2, 1, 1),
    ])
    moves = legal_moves(new_game)
    # White: e1 c1
    assert move(7,4,7,2,0) not in moves, "White illegally allowed to castle when blocked by piece!"
    # Black: e8 c8
    assert move(0,4,0,2,0) not in moves, "Black illegally allowed to castle when blocked by piece!"