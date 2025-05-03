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
from collections import defaultdict


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
        # If move does not leave moving player in check: recurse
        if not is_check_numba(gs, gs.white_to_move):
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move
            move_info.append(prev_state)
            nodes += _bitboard_perft(gs, depth-1, move_info)
            undo_move_numba(gs, move_info)
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move
        else:
            undo_move_numba(gs, prev_state)

    return nodes

@njit
def _bitboard_perft_sequences(gs, depth, sequence, move_info, result_sequences):
    if depth == 0:
        result_sequences.append(sequence[:])
        return 1

    nodes = 0
    for move in generate_all_moves(gs):
        prev_state = apply_move_numba(gs, move)
        if not is_check_numba(gs, gs.white_to_move):
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move
            move_info.append(prev_state)
            sequence.append(move)
            nodes += _bitboard_perft_sequences(gs, depth-1, sequence, move_info, result_sequences)
            sequence.pop()
            undo_move_numba(gs, move_info)
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move
        else:
            undo_move_numba(gs, prev_state)
    return nodes

def bitboard_perft(gs, depth):
    move_info = List.empty_list(move_state_type)
    return _bitboard_perft(gs, depth, move_info)

def bitboard_perft_sequences(gs, depth):
    move_info = List.empty_list(move_state_type)
    sequence = List.empty_list(types.UniTuple(types.int64, 3))  # assuming move = (from_sq, to_sq)
    result_sequences = List.empty_list(types.ListType(types.UniTuple(types.int64, 3)))
    _bitboard_perft_sequences(gs, depth, sequence, move_info, result_sequences)
    validate(result_sequences)
    return len(result_sequences)

def parse_tuple_list(s: str):
    return ast.literal_eval(s)

def validate(sequences):
    invalid_moves = 0
    missing_moves = 0

    # Organize sequences by their prefix of length `d`
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
                prefix_str = "("
                for p in prefix:
                    prefix_str += get_standard_algebraic(p) + ", "
                print(f"{invalid_moves+1}. {prefix_str[:-2]}) --> {move} - invalid")
                invalid_moves += 1

        # Check for missing legal moves (legal but not logged)
        for move in legal:
            if move not in logged:
                prefix_str = "("
                for p in prefix:
                    prefix_str += get_standard_algebraic(p) + ", "
                print(f"{missing_moves+1}. {prefix_str[:-2]}) --> ({move}) - missing")
                missing_moves += 1

    print(f"{invalid_moves} invalid moves")
    print(f"{missing_moves} missing moves")

# ========= ^ FOR DEBUGGING =========
# ===================================

    
if __name__ == "__main__":
    number_of_positions  = [1, 20, 400, 8902, 197281, 4865609, 119060324]
    number_of_checkmates = [0, 0, 0, 8, 347, 10828]
    gs = BitboardGameState()
    if len(sys.argv) > 1:
        print("\nJitting methods...")
        bitboard_perft(gs, 1)
        print("\nLogging and validating moves...")
        depth = int(sys.argv[1])
        start = time.perf_counter()
        positions_reached = bitboard_perft_sequences(gs, depth)
        end = time.perf_counter()
        runtime = end-start
        print(f"\n--- Depth {depth} ---")
        print(f"Runtime     : {runtime:.6f}s")
        print(f"Nodes count : {positions_reached}")
        print(f"Anticipated : {number_of_positions[depth]}")
        print(f"Nodes/second: {positions_reached/runtime:.2f}")
    else:
        for depth in range(7):
            print()
            start = time.perf_counter()
            positions_reached = bitboard_perft(gs, depth)
            end = time.perf_counter()
            runtime = end-start
            print(f"--- Depth {depth} ---")
            print(f"Runtime     : {runtime:.6f}s")
            print(f"Nodes count : {positions_reached}")
            print(f"Anticipated : {number_of_positions[depth]}")
            print(f"Nodes/second: {positions_reached/runtime:.2f}")

    print()
            