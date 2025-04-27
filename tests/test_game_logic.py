import pytest
import numpy as np
from engine_utils import *
from game import GameState

def new_game():
    return GameState()

def simulate_move(gs, move):
    apply_move(gs, move)

def find_move(moves, from_sq, to_sq, promotion=0):
    for m in moves:
        if (m[0], m[1], m[2], m[3], m[4]) == (from_sq[0], from_sq[1], to_sq[0], to_sq[1], promotion):
            return True
    return False

def test_pawn_single_and_double_move():
    gs = new_game()
    moves = generate_legal_moves(gs)
    assert find_move(moves, (6, 0), (5, 0))
    assert find_move(moves, (6, 0), (4, 0))

def test_pawn_capture():
    gs = new_game()
    simulate_move(gs, (6,4,4,4,0))  # e2 e4
    simulate_move(gs, (1,3,3,3,0))  # d7 d5
    moves = generate_legal_moves(gs)
    assert find_move(moves, (4,4), (3,3))  # e4 d5 capture

def test_knight_moves():
    gs = new_game()
    moves = generate_legal_moves(gs)
    assert find_move(moves, (7,1), (5,2))
    assert find_move(moves, (7,1), (5,0))

def test_bishop_blocked():
    gs = new_game()
    moves = generate_legal_moves(gs)
    assert not any(abs(gs.board[m[2],m[3]]) == BISHOP for m in moves)

def test_king_moves_one_square():
    gs = new_game()
    gs.board[7,4] = KING
    gs.board[7,5] = EMPTY
    gs.board[6,4] = EMPTY
    moves = generate_legal_moves(gs)
    assert find_move(moves, (7,4), (6,4))
    assert find_move(moves, (7,4), (7,5))

def test_en_passant():
    gs = new_game()
    simulate_move(gs, (6,4,4,4,0)) # e2 e4
    simulate_move(gs, (1,0,3,0,0)) # a7 a5
    simulate_move(gs, (4,4,3,4,0)) # e4 e5
    simulate_move(gs, (1,5,3,5,0)) # f7 f5
    moves = generate_legal_moves(gs)
    assert find_move(moves, (3,4), (2,5))  # e5 f6 capture en passant

def test_castling_rights_start():
    gs = new_game()
    moves = generate_legal_moves(gs)
    # Should not allow castling immediately at start (blocked)
    for m in moves:
        assert not (abs(m[1] - m[3]) == 2 and abs(gs.board[m[0],m[1]]) == KING)

def test_castling_after_clear():
    gs = new_game()
    gs.board[7,5] = EMPTY
    gs.board[7,6] = EMPTY
    moves = generate_legal_moves(gs)
    can_castle_kingside = any((m[0], m[1], m[2], m[3]) == (7,4,7,6) for m in moves)
    assert can_castle_kingside

def test_illegal_king_move_into_check():
    gs = new_game()
    gs.board[0,4] = EMPTY
    gs.board[1,4] = EMPTY
    gs.board[6,5] = EMPTY
    gs.board[7,5] = ROOK
    moves = generate_legal_moves(gs)
    # King should not be allowed to move into e8 if rook attacks
    assert not any((m[2], m[3]) == (0,4) for m in moves)

def test_promotion_moves():
    gs = new_game()
    gs.board[1,0] = PAWN
    gs.board[6,0] = EMPTY
    moves = generate_legal_moves(gs)
    promotions = [m for m in moves if m[4] != 0]
    assert promotions  # There should be promotion moves