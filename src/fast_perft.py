from numba import njit
from numba.typed import List

from engine_utils import generate_legal_moves, apply_move
from evaluate_board import evaluate_board
from game import GameState

import time
import numpy as np

@njit
def fast_perft(gs, depth):
    if depth == 0:
        return 1

    nodes = 0
    moves = generate_legal_moves(gs)

    original_board = gs.board.copy()
    original_en_passant = np.copy(gs.en_passant_target)
    original_castling_rights = np.copy(gs.castling_rights)
    original_halfmove_clock = gs.halfmove_clock
    original_fullmove_number = gs.fullmove_number
    original_white_to_move = gs.white_to_move

    for move in moves:
        moving_piece = gs.board[move[0], move[1]]
        captured_piece = gs.board[move[2], move[3]]

        apply_move(gs, move)
        evaluate_board(gs)
        
        nodes += fast_perft(gs, depth - 1)

        # Undo move when finished traversing 1-move subtree
        gs.board[:, :] = original_board
        gs.en_passant_target[:] = original_en_passant
        gs.castling_rights[:] = original_castling_rights
        gs.halfmove_clock = original_halfmove_clock
        gs.fullmove_number = original_fullmove_number
        gs.white_to_move = original_white_to_move

    return nodes

if __name__=="__main__":
    gs = GameState()
    start = time.time()
    nodes = fast_perft(gs, 3)
    runtime = time.time() - start
    print("Nodes:", nodes)
    print(f"Time: {round(runtime)}s")
    print("Nodes/second:", round(nodes / runtime))