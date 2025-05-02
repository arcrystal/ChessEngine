from bitboard_game import BitboardGameState
from bitboard_gamestate_utils import is_check_numba, apply_move_numba, undo_move_numba, update_occupancies_numba
from generate_moves import generate_all_moves
from debugging import get_standard_algebraic
from numba import uint64, types, njit
from numba.typed import List

import time
import sys
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
    # For move in moves at depth d:
    for move in generate_all_moves(gs):
        prev_state = apply_move_numba(gs, move)
        # If move is valid: recurse
        if not is_check_numba(gs, not gs.white_to_move):
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move
            move_info.append(prev_state)
            nodes += _bitboard_perft(gs, depth-1, move_info)
            undo_move_numba(gs, move_info)
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move

    return nodes

def _bitboard_perft_sequences(gs, depth, sequence, outfile, move_info):
    # If reached base case, log the sequence of moves to reach this leaf
    if depth == 0:
        outfile.write(str(sequence)+"\n")
        return 1

    nodes = 0
    # For move in moves at depth d:
    for move in generate_all_moves(gs):
        prev_state = apply_move_numba(gs, move)
        # If move is valid: recurse
        if not is_check_numba(gs, not gs.white_to_move):
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move
            move_info.append(prev_state)
            sequence.append(move)
            nodes += _bitboard_perft_sequences(gs, depth-1, sequence, outfile, move_info)
            sequence.pop()
            undo_move_numba(gs, move_info)
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move

    return nodes

@njit
def bitboard_perft(gs, depth):
    move_info = List.empty_list(move_state_type)
    return _bitboard_perft(gs, depth, move_info)

def bitboard_perft_sequences(gs, depth, outfile):
    move_info = List.empty_list(move_state_type)
    sequence = []
    return _bitboard_perft_sequences(gs, depth, sequence, outfile, move_info)

def parse_tuple_list(s: str):
    return ast.literal_eval(s)

import chess

def validate(depth: int):
    with open(f"logs/moves_depth{depth}.txt", "r") as f:
        sequences = [parse_tuple_list(line) for line in f.readlines()]
        
    invalid_moves = 0
    missing_moves = 0

    # Organize sequences by their prefix of length `d`
    from collections import defaultdict
    prefix_map = defaultdict(list)
    for seq in sequences:
        for d in range(len(seq)):
            prefix = tuple(seq[:d])
            move = seq[d]
            prefix_map[prefix].append(move)

    for prefix, logged_moves in prefix_map.items():
        board = chess.Board()
        for move_tuple in prefix:
            board.push(chess.Move(move_tuple[0], move_tuple[1]))

        legal = set(board.legal_moves)
        logged = set(chess.Move(m[0], m[1]) for m in logged_moves)

        # Check for invalid moves (logged but illegal)
        for move in logged:
            if move not in legal:
                print(f"{invalid_moves+1}. {prefix} --> {move}")
                invalid_moves += 1

        # Check for missing legal moves (legal but not logged)
        for move in legal:
            if move not in logged:
                prefix_str = "("
                for p in prefix:
                    prefix_str += get_standard_algebraic(p) + ", "
                print(f"{missing_moves+1}. {prefix_str[:-2]}) --> ({move})")
                missing_moves += 1

    print(f"{invalid_moves} invalid moves")
    print(f"{missing_moves} missing moves")

# ========= ^ FOR DEBUGGING =========
# ===================================

    
if __name__ == "__main__":
    correct_nodes = [1, 20, 400, 8902, 197281, 4865609]
    gs = BitboardGameState()
    if len(sys.argv) > 1:
        print("Logging and validating moves...")
        depth = int(sys.argv[1])
        with open(f"logs/moves_depth{depth}.txt", "w") as f:
            print()
            start = time.perf_counter()
            res = bitboard_perft_sequences(gs, depth, f)
            end = time.perf_counter()
            print(f"Depth {depth}\nExecuted in {(end-start):.6f}\nNodes count: {res}\nAnticipated: {correct_nodes[depth]}")
            validate(depth)
    else:
        for depth in range(6):
            print()
            start = time.perf_counter()
            res = bitboard_perft(gs, depth)
            end = time.perf_counter()
            print(f"Depth {depth}\nExecuted in {(end-start):.6f}\nNodes count: {res}\nAnticipated: {correct_nodes[depth]}")

    print()
            