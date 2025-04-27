import pytest
import numpy as np
from game import GameState
from engine_utils import generate_legal_moves, apply_move

@pytest.fixture
def new_game():
    return GameState()

def move(from_row, from_col, to_row, to_col, promotion=0):
    return (np.int8(from_row), np.int8(from_col), np.int8(to_row), np.int8(to_col), np.int8(promotion))

def simulate_moves(game, moves):
    for move in moves:
        apply_move(game, (move[0], move[1], move[2], move[3], 0))


# --- Success En Passant Tests ---
def test_white_en_passant_left(new_game):
    # e2 e4
    # d7 d5
    # e4 d5
    # c7 c5
    simulate_moves(new_game, [
        (6, 4, 4, 4),
        (1, 3, 3, 3),
        (4, 4, 3, 3),
        (1, 2, 3, 2),
    ])
    moves = generate_legal_moves(new_game)
    # d5 c6
    assert move(3, 3, 2, 2, 0) in moves, "White en passant left missing!"

def test_white_en_passant_right(new_game):
    # e2 e4
    # f7 f5
    # e4 f5
    # g7 g5
    simulate_moves(new_game, [
        (6, 4, 4, 4),
        (1, 5, 3, 5),
        (4, 4, 3, 5),
        (1, 6, 3, 6),
    ])
    moves = generate_legal_moves(new_game)
    # f5 g6
    assert move(3, 5, 2, 6, 0) in moves, "White en passant right missing!"

def test_black_en_passant_left(new_game):
    # h2 h3
    # e7 e5
    # h3 h4
    # e5 e4
    # d2 d4
    simulate_moves(new_game, [
        (6, 7, 5, 7),
        (1, 4, 3, 4),
        (5, 7, 4, 7),
        (3, 4, 4, 4),
        (6, 3, 4, 3),
    ])
    moves = generate_legal_moves(new_game)
    # e4 d3
    assert move(4, 4, 5, 3, 0) in moves, "Black en passant left missing!"

def test_black_en_passant_right(new_game):
    # h2 h3
    # e7 e5
    # h3 h4
    # e5 e4
    # f2 f4
    simulate_moves(new_game, [
        (6, 7, 5, 7),
        (1, 4, 3, 4),
        (5, 7, 4, 7),
        (3, 4, 4, 4),
        (6, 5, 4, 5),
    ])
    moves = generate_legal_moves(new_game)
    # e4 f3
    assert move(4, 4, 5, 5, 0) in moves, "Black en passant right missing!"


# --- Failure En Passant Tests ---

def test_en_passant_waited_extra_turn(new_game):
    # e2 e4
    # d7 d5
    # e4 d5
    # c7 c5
    # a2 a3
    # a7 a6
    simulate_moves(new_game, [
        (6, 4, 4, 4),
        (1, 3, 3, 3),
        (4, 4, 3, 3),
        (1, 2, 3, 2),
        (6, 0, 5, 0),
        (1, 0, 2, 0),
    ])
    moves = generate_legal_moves(new_game)
    # Should NOT allow d5 c6 en passant anymore
    assert move(3, 3, 2, 2, 0) not in moves, "Illegal en passant allowed after waiting a turn!"


def test_en_passant_wrong_rank(new_game):
    # e2 e3
    # d7 d5
    # e3 e4
    # d5 d4
    simulate_moves(new_game, [
        (6, 4, 5, 4),
        (1, 3, 3, 3),
        (5, 4, 4, 4),
        (3, 3, 4, 3),
    ])
    moves = generate_legal_moves(new_game)
    # Should NOT allow e4 d5 en passant because pawn didn't double move from rank 2
    assert move(4, 4, 3, 3, 0) not in moves, "Illegal en passant from wrong rank!"

def test_en_passant_no_pawn_next(new_game):
    # e2 e4
    # a7 a6
    # e4 e5
    # a6 a5
    simulate_moves(new_game, [
        (6, 4, 4, 4),
        (1, 0, 2, 0),
        (4, 4, 3, 4),
        (2, 0, 3, 0),
    ])
    moves = generate_legal_moves(new_game)
    # Should NOT allow e5 d6 or e5 c6 en passant, no pawn adjacent
    assert move(3, 4, 2, 5, 0) not in moves, "Illegal en passant when no adjacent pawn (right)!"
    assert move(3, 4, 2, 3, 0) not in moves, "Illegal en passant when no adjacent pawn (left)!"

def test_en_passant_pawn_moved_one_square(new_game): 
    # e2 e4
    # d7 d6
    # e4 e5
    # d6 d5
    simulate_moves(new_game, [
        (6, 4, 4, 4),
        (1, 3, 2, 3),
        (4, 4, 3, 4),
        (2, 3, 3, 3),
    ])
    moves = generate_legal_moves(new_game)
    # Should NOT allow e5 d6 en passant
    assert move(3, 4, 2, 3, 0) not in moves, "Illegal en passant after pawn only moved one square!"
    
def test_en_passant_to_capture_bishop(new_game):
    # e2 e4
    # d7 d6
    # e4 e5
    # c8 f5
    simulate_moves(new_game, [
        (6, 4, 4, 4),
        (1, 3, 2, 3),
        (4, 4, 3, 4),
        (0, 2, 3, 5)
    ])
    moves = generate_legal_moves(new_game)
    # Should NOT allow e5 f6
    assert move(3, 4, 2, 5, 0) not in moves, "Illegal en passant to capture Bishop"