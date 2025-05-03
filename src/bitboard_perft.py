from bitboard_game import BitboardGameState
from bitboard_gamestate_utils import is_check_numba, apply_move_numba, undo_move_numba, update_occupancies_numba
from generate_moves import generate_all_moves
from bitboard_debugging import get_standard_algebraic
from constants import move4_mates
from numba import uint64, types, njit
from numba.typed import List
import chess

import time
import sys
import ast
import chess
from collections import defaultdict
import tqdm


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
    legal_move_found = False
    moves = generate_all_moves(gs)
    for move in moves:
        prev_state = apply_move_numba(gs, move)
        update_occupancies_numba(gs)
        gs.white_to_move = not gs.white_to_move

        if not is_check_numba(gs, not gs.white_to_move):  # check moving player's king
            legal_move_found = True
            move_info.append(prev_state)
            nodes += _bitboard_perft(gs, depth - 1, move_info)
            undo_move_numba(gs, move_info)
        else:
            undo_move_numba(gs, prev_state)

        update_occupancies_numba(gs)
        gs.white_to_move = not gs.white_to_move

    if not legal_move_found and is_check_numba(gs, gs.white_to_move):
        print("Checkmate!")

    return nodes

@njit
def _bitboard_perft_sequences(gs, depth, sequence, move_info, result_sequences, all_checkmates):
    if depth == 0:
        result_sequences.append(sequence[:])
        return 1
    
    nodes = 0
    for move in generate_all_moves(gs):
        prev_state = apply_move_numba(gs, move)
        update_occupancies_numba(gs)
        gs.white_to_move = not gs.white_to_move

        if not is_check_numba(gs, not gs.white_to_move):  # check moving player's king
            move_info.append(prev_state)
            sequence.append(move)
            # for mate_sequence in all_checkmates:
            #     checkmate = True
            #     for move in mate_sequence:
            #         if move not in sequence:
            #             checkmate = False
            #             break
            #     if checkmate:
            #         print("Checkmate!")

            nodes += _bitboard_perft_sequences(gs, depth - 1, sequence, move_info, result_sequences, all_checkmates)
            sequence.pop()
            undo_move_numba(gs, move_info)
        else:
            undo_move_numba(gs, prev_state)

        update_occupancies_numba(gs)
        gs.white_to_move = not gs.white_to_move

    return nodes

def bitboard_perft(gs, depth):
    move_info = List.empty_list(move_state_type)
    return _bitboard_perft(gs, depth, move_info)

def bitboard_perft_sequences(gs, depth):
    move_info = List.empty_list(move_state_type)
    sequence = List.empty_list(types.UniTuple(types.int64, 3))  # assuming move = (from_sq, to_sq)
    result_sequences = List.empty_list(types.ListType(types.UniTuple(types.int64, 3)))
    _bitboard_perft_sequences(gs, depth, sequence, move_info, result_sequences, move4_mates)
    #validate(result_sequences)
    diff_vs_python_chess(result_sequences)
    return len(result_sequences)

def parse_tuple_list(s: str):
    return ast.literal_eval(s)

def validate(sequences):
    invalid_moves = 0
    missing_moves = 0

    # Map each prefix → list of attempted next moves
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

        # Check for invalid moves
        for move in logged:
            if move not in legal:
                prefix_str = "("
                for p in prefix:
                    prefix_str += get_standard_algebraic(p) + ", "
                print(f"{invalid_moves+1}. {prefix_str[:-2]}) --> {move.uci()} - invalid")
                invalid_moves += 1

        # Check for missing legal moves
        for move in legal:
            if move not in logged:
                prefix_str = "("
                for p in prefix:
                    prefix_str += get_standard_algebraic(p) + ", "
                print(f"{missing_moves+1}. {prefix_str[:-2]}) --> ({move.uci()}) - missing")
                missing_moves += 1

    print(f"{invalid_moves} invalid moves")
    print(f"{missing_moves} missing moves")


def diff_vs_python_chess(sequences):
    prefix_map = defaultdict(list)
    for seq in tqdm.tqdm(sequences, "Building found sequences map"):
        for d in range(len(seq)):
            prefix = tuple(seq[:d])
            move = seq[d]
            prefix_map[prefix].append(move)

    invalid = 0
    missing = 0

    for prefix, logged_moves in tqdm.tqdm(prefix_map.items(), "Verifying and finding missed sequences"):
        board = chess.Board()
        for move in prefix:
            board.push(chess.Move(move[0], move[1]))

        legal_moves = set(board.legal_moves)
        logged_set = set(chess.Move(m[0], m[1]) for m in logged_moves)

        illegal_logged = logged_set - legal_moves
        missed_legal = legal_moves - logged_set

        if illegal_logged or missed_legal:
            prefix_str = ", ".join(get_standard_algebraic(m) for m in prefix)
            print(f"\nPrefix: ({prefix_str})")

        for move in illegal_logged:
            print(f"  ❌ Invalid: {move.uci()}")
            invalid += 1

        for move in missed_legal:
            print(f"  ❗ Missing: {move.uci()}")
            missing += 1

    print(f"\nTotal invalid: {invalid}")
    print(f"Total missing: {missing}")

if __name__ == "__main__":
    number_of_positions  = [1, 20, 400, 8902, 197281, 4865609, 119060324]
    number_of_checkmates = [0, 0, 0, 8, 347, 10828]
    gs = BitboardGameState()
    if len(sys.argv) > 1:
        print("\nJitting methods...")
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
            print(f"Perft Runtime : {runtime:.6f}s")
            print(f"Nodes counted : {positions_reached}")
            print(f"Actual number : {number_of_positions[depth]}")
            print(f"Nodes/second  : {positions_reached/runtime:.2f}")

    print()
            