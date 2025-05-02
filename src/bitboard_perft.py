from bitboard_game import BitboardGameState
from bitboard_gamestate_utils import is_check_numba, apply_move_numba, undo_move_numba, update_occupancies_numba
from generate_moves import generate_all_moves

from numba import uint64, boolean, types, njit
from numba.typed import List

import time
import sys
import functools
import ast
import chess

import warnings
warnings.simplefilter('ignore', category=Warning, lineno=0, append=False)

move_state_type = types.Tuple((
    uint64, uint64, uint64, uint64, uint64, uint64,   # white pieces
    uint64, uint64, uint64, uint64, uint64, uint64,   # black pieces
    types.UniTuple(types.int8, 4),                    # castling_rights
    types.int64,                                      # en_passant_target
    types.int32,                                      # halfmove_clock
    types.int32                                       # fullmove_number
))

@njit
def _bitboard_perft(gs, depth, move_info):
    if depth == 0:
        return 1
    
    nodes = 0
    moves = generate_all_moves(gs)
    for move in moves:
        prev_state = apply_move_numba(gs, move)
        move_info.append(prev_state)
        if not is_check_numba(gs, not gs.white_to_move):
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move
            nodes += _bitboard_perft(gs, depth - 1, move_info)
            undo_move_numba(gs, move_info)
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move

    return nodes

def _bitboard_perft_sequences(gs, depth, sequence, outfile, move_info):
    if depth == 0:
        outfile.write(str(sequence)+"\n")
        return

    for move in generate_all_moves(gs):
        prev_state = apply_move_numba(gs, move)
        if not is_check_numba(gs, not gs.white_to_move):
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move
            move_info.append(prev_state)
            sequence.append(move)
            _bitboard_perft_sequences(gs, depth-1, sequence, outfile, move_info)
            sequence.pop()
            undo_move_numba(gs, move_info)
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move

    return

@njit
def bitboard_perft(gs, depth):
    move_info = List.empty_list(move_state_type)
    return _bitboard_perft(gs, depth, move_info)

def bitboard_perft_sequences(gs, depth, outfile):
    move_info = List.empty_list(move_state_type)
    sequence = []
    _bitboard_perft_sequences(gs, depth, sequence, outfile, move_info)

def parse_tuple_list(s: str):
    return ast.literal_eval(s)

def validate(depth: int):
    with open(f"logs/moves_depth{depth}.txt", "r") as f:
        sequences = [parse_tuple_list(line) for line in f.readlines()]
        
    n = 0
    for seq in sequences:
        board = chess.Board()
        moves = [chess.Move(move[0], move[1]) for move in seq]
        for move in moves:
            if move not in board.legal_moves:
                msg = str(move) + " invalid from "
                for m in moves:
                    msg += str(m) + ", "
                print(msg[:-2])
                n += 1
                break
            
            board.push(move)
            
    print(f"{n} invalid moves")

# ========= ^ FOR DEBUGGING =========
# ===================================

    
if __name__ == "__main__":
    correct_nodes = [1, 20, 400, 8902, 197281, 4865609]
    gs = BitboardGameState()
    if len(sys.argv) > 1:
        print("Logging moves...")
        depth = int(sys.argv[1])
        with open(f"logs/moves_depth{depth}.txt", "w") as f:
            print()
            start = time.perf_counter()
            res = bitboard_perft_sequences(gs, depth, f)
            end = time.perf_counter()
            print(f"Depth {depth}\nExecuted in {(end-start):.6f}\nNodes count: {res}\nAnticipated: {correct_nodes[depth]}")
            validate(depth)
    else:
        for depth in range(5):
            print()
            start = time.perf_counter()
            res = bitboard_perft(gs, depth)
            end = time.perf_counter()
            print(f"Depth {depth}\nExecuted in {(end-start):.6f}\nNodes count: {res}\nAnticipated: {correct_nodes[depth]}")

    print()
            