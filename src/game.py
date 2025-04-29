import numpy as np
from numba import int8, boolean, int16
from numba.experimental import jitclass

# --- Constants ---
EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5, 6
WHITE, BLACK = 1, -1

INITIAL_BOARD = np.array([
    [-ROOK, -KNIGHT, -BISHOP, -QUEEN, -KING, -BISHOP, -KNIGHT, -ROOK],
    [-PAWN, -PAWN,   -PAWN,   -PAWN,  -PAWN,  -PAWN,   -PAWN,   -PAWN],
    [ EMPTY, EMPTY,  EMPTY,   EMPTY,  EMPTY,  EMPTY,   EMPTY,   EMPTY],
    [ EMPTY, EMPTY,  EMPTY,   EMPTY,  EMPTY,  EMPTY,   EMPTY,   EMPTY],
    [ EMPTY, EMPTY,  EMPTY,   EMPTY,  EMPTY,  EMPTY,   EMPTY,   EMPTY],
    [ EMPTY, EMPTY,  EMPTY,   EMPTY,  EMPTY,  EMPTY,   EMPTY,   EMPTY],
    [ PAWN,  PAWN,   PAWN,    PAWN,   PAWN,   PAWN,    PAWN,    PAWN],
    [ ROOK,  KNIGHT, BISHOP,  QUEEN,  KING,   BISHOP,  KNIGHT,  ROOK]
], dtype=np.int8)

# --- GameState schema for JIT ---
spec = [
    ('board', int8[:, :]),
    ('white_to_move', boolean),
    ('en_passant_target', int8[:]),
    ('castling_rights', int8[:]),  # 4 flags: w_kingside, w_queenside, b_kingside, b_queenside
    ('halfmove_clock', int16),
    ('fullmove_number', int16)
]


@jitclass(spec)
class GameState:
    def __init__(self):
        self.board = np.zeros((8,8), dtype=np.int8)
        self.white_to_move = True
        self.en_passant_target = np.array([-1, -1], dtype=np.int8)
        self.castling_rights = np.ones(4, dtype=np.int8)
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.reset()

    def reset(self):
        for r in range(8):
            for c in range(8):
                self.board[r, c] = INITIAL_BOARD[r, c]
        self.white_to_move = True
        self.en_passant_target[0] = -1
        self.en_passant_target[1] = -1
        self.castling_rights[:] = 1
        self.halfmove_clock = 0
        self.fullmove_number = 1

    def switch_turn(self):
        self.white_to_move = not self.white_to_move
        if self.white_to_move:
            self.fullmove_number += 1