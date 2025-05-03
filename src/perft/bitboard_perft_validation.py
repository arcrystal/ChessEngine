
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.bitboard_game import BitboardGameState
from src.bitboard_gamestate_utils import is_check_numba, apply_move_numba, undo_move_numba, update_occupancies_numba
from src.generate_moves import generate_all_moves
from src.bitboard_debugging import get_standard_algebraic
from src.constants import move4_mates, move_state_type, number_of_checkmates, number_of_positions
from numba import types, njit
from numba.typed import List
import chess

import time
import sys
import ast
import chess
from collections import defaultdict
import tqdm
from datetime import datetime, timedelta
import logging

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

def bitboard_perft_sequences(gs, depth):
    move_info = List.empty_list(move_state_type)
    sequence = List.empty_list(types.UniTuple(types.int64, 3))  # assuming move = (from_sq, to_sq)
    result_sequences = List.empty_list(types.ListType(types.UniTuple(types.int64, 3)))
    _bitboard_perft_sequences(gs, depth, sequence, move_info, result_sequences, move4_mates)
    validate(result_sequences)
    return len(result_sequences)

def parse_tuple_list(s: str):
    return ast.literal_eval(s)


def validate(sequences):
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
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = f"logs/perft_{timestamp}.log"

    logging.basicConfig(
        filename=logfile,
        filemode='w',
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    gs = BitboardGameState()
    if len(sys.argv) > 1:
        logging.info("\nJitting methods...")
        logging.info("\nLogging and validating moves...")
        depth = int(sys.argv[1])
        start = time.perf_counter()
        positions_reached = bitboard_perft_sequences(gs, depth)
        end = time.perf_counter()
        runtime = end-start
        formatted_runtime = str(timedelta(seconds=runtime))

        logging.info(f"\n------ Depth {depth} ------")
        logging.info(f"Total Runtime        : {formatted_runtime}")
        logging.info(f"Positions found      : {positions_reached}")
        logging.info(f"Actual # positions   : {number_of_positions[depth]}")
        logging.info(f"Nodes found / second : {positions_reached/runtime:.0f}")
