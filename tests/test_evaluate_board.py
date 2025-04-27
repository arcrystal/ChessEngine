import numpy as np
import pytest
from evaluate_board import evaluate_board
from game import GameState

@pytest.fixture
def fresh_game():
    return GameState()

def test_starting_position_eval(fresh_game):
    """At start position, eval should be ~0 (balanced)."""
    eval_score = evaluate_board(fresh_game)
    assert -50 <= eval_score <= 50, f"Unexpected evaluation at start: {eval_score}"

def test_material_advantage_white(fresh_game):
    """White up a rook should evaluate strongly positive."""
    fresh_game.board[1, 0] = 0  # Remove black pawn
    fresh_game.board[0, 0] = 0  # Remove black rook
    eval_score = evaluate_board(fresh_game)
    assert eval_score > 400, f"White material advantage not detected: {eval_score}"

def test_material_advantage_black(fresh_game):
    """Black up a queen should evaluate strongly negative."""
    fresh_game.board[6, 3] = 0  # Remove white pawn
    fresh_game.board[7, 3] = 0  # Remove white queen
    eval_score = evaluate_board(fresh_game)
    assert eval_score < -800, f"Black material advantage not detected: {eval_score}"

def test_king_safety_penalty(fresh_game):
    """King shelter penalty: if pawns missing, eval should worsen."""
    fresh_game.board[6, 4] = 0  # Remove pawn in front of white king
    eval_score_without_shelter = evaluate_board(fresh_game)
    eval_score_normal = evaluate_board(GameState())  # Fresh game

    assert eval_score_without_shelter < eval_score_normal, (
        "King shelter penalty not working: missing pawn should worsen evaluation"
    )

def test_passed_pawn_bonus(fresh_game):
    """Passed pawn bonus should apply if pawn has no opposing pawns."""
    fresh_game.board[1, 4] = 0  # Remove black pawn in e-file
    fresh_game.board[6, 4] = 1  # White pawn at e2
    fresh_game.board[5, 4] = 0  # No obstruction
    eval_score = evaluate_board(fresh_game)
    assert eval_score > 50, "Passed pawn bonus not applied!"