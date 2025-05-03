import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
from datetime import datetime, timedelta
import logging

from numba import njit
from numba.typed import List
from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings
warnings.simplefilter('ignore', category=Warning, lineno=0, append=False)

from src.bitboard_game import BitboardGameState
from src.bitboard_gamestate_utils import is_check_numba, apply_move_numba, undo_move_numba, update_occupancies_numba, get_move_info
from src.generate_moves import generate_all_moves
from src.constants import move_state_type, number_of_positions
from src.bitboard_debugging import get_standard_algebraic

@njit
def _bitboard_perft(gs, depth, move_info):
    if depth == 0:
        return 1

    nodes = 0
    moves = generate_all_moves(gs)
    for move in moves:
        prev_state = apply_move_numba(gs, move)
        update_occupancies_numba(gs)
        gs.white_to_move = not gs.white_to_move

        if not is_check_numba(gs, not gs.white_to_move):
            move_info.append(prev_state)
            nodes += _bitboard_perft(gs, depth - 1, move_info)
            undo_move_numba(gs, move_info)
        else:
            undo_move_numba(gs, prev_state)

        update_occupancies_numba(gs)
        gs.white_to_move = not gs.white_to_move

    return nodes

def perft_worker(move, start_state, depth):
    gs = BitboardGameState()
    undo_move_numba(gs, start_state)
    prev = apply_move_numba(gs, move)
    update_occupancies_numba(gs)
    gs.white_to_move = not gs.white_to_move

    if is_check_numba(gs, not gs.white_to_move):
        undo_move_numba(gs, prev)
        update_occupancies_numba(gs)
        gs.white_to_move = not gs.white_to_move
        return 0

    move_info = List.empty_list(move_state_type)
    return _bitboard_perft(gs, depth - 1, move_info)

def parallel_bitboard_perft(gs, depth):
    root_moves = generate_all_moves(gs)
    logging.info(f"Parallelizing {len(root_moves)} root moves...")

    start_state = get_move_info(gs)
    total_nodes = 0

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(perft_worker, move, start_state, depth) for move in root_moves]
        for i, future in enumerate(as_completed(futures), 1):
            nodes = future.result()
            total_nodes += nodes
            logging.info(f"Move {i}/{len(root_moves)} ({get_standard_algebraic(root_moves[i-1])}): {nodes} nodes")

    return total_nodes


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = f"logs/perft_parallel_{timestamp}.log"

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
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    logging.info(f"\nRunning parallel perft to depth {depth}...")
    start = time.perf_counter()
    nodes = parallel_bitboard_perft(gs, depth)
    end = time.perf_counter()
    runtime = end - start
    formatted_runtime = str(timedelta(seconds=runtime))

    logging.info(f"\n------ Depth {depth} ------")
    logging.info(f"Total Runtime        : {formatted_runtime}")
    logging.info(f"Positions found      : {nodes}")
    logging.info(f"Expected positions   : {number_of_positions[depth]}")
    logging.info(f"Nodes/second         : {nodes/runtime:.0f}")

            