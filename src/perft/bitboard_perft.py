import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.bitboard_game import BitboardGameState
from src.bitboard_gamestate_utils import is_check_numba, apply_move_numba, undo_move_numba, update_occupancies_numba
from src.generate_moves import generate_all_moves
from src.constants import move_state_type, number_of_positions
from numba import njit
from numba.typed import List

import time
from datetime import datetime, timedelta
import logging


import warnings
warnings.simplefilter('ignore', category=Warning, lineno=0, append=False)


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
        # TODO: Do something for checkmate
        pass

    return nodes


def bitboard_perft(gs, depth):
    move_info = List.empty_list(move_state_type)
    return _bitboard_perft(gs, depth, move_info)

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
    for depth in range(10):
        start = time.perf_counter()
        positions_reached = bitboard_perft(gs, depth)
        end = time.perf_counter()
        runtime = end-start
        formatted_runtime = str(timedelta(seconds=runtime))

        logging.info(f"\n------ Depth {depth} ------")
        logging.info(f"Total Runtime        : {formatted_runtime}")
        logging.info(f"Positions found      : {positions_reached}")
        logging.info(f"Actual # positions   : {number_of_positions[depth]}")
        logging.info(f"Nodes found / second : {positions_reached/runtime:.0f}")

            