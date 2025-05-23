from bitboard_game import BitboardGameState
from generate_moves import generate_all_moves
import time
from functools import wraps

# Validation tools
import chess
import ast
from typing import List, Tuple

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        res = f"{func.__name__} executed in {end - start:.6f} seconds. {result}"
        return res
    return wrapper

counts_set = set()

def _bitboard_perft(gs, depth, verbose):
    if depth == 0:
        return 1
    
    nodes = 0
    moves = generate_all_moves(gs, verbose)
    for move in moves:
        gs.make_move(move)
        if not gs.is_check(not gs.white_to_move):
            nodes += _bitboard_perft(gs, depth-1, verbose)
            
        gs.undo_move()
        
    return nodes


@timeit
def bitboard_perft(gs, depth, verbose=False):
    return _bitboard_perft(gs, depth, verbose)

def bitboard_perft_sequences(gs, depth, outfile):
    sequence = []
    _bitboard_perft_sequences(gs, depth, sequence, outfile)


def _bitboard_perft_sequences(gs, depth, sequence, outfile):
    if depth == 0:
        outfile.write(str(sequence)+"\n")
        return

    moves = generate_all_moves(gs, verbose=False)
    for move in moves:
        gs.make_move(move)
        if not gs.is_check(not gs.white_to_move):
            sequence.append(move)
            _bitboard_perft_sequences(gs, depth - 1, sequence, outfile)
            sequence.pop()
        gs.undo_move()

    return

# VALIDATION

def parse_tuple_list(s: str) -> List[Tuple[int]]:
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
                msg = str(move) + " from "
                for m in moves:
                    msg += str(m) + ", "
                msg += "is not allowed"
                print(msg)
                n += 1
                break
            
            board.push(move)
            
    print(f"{n} wrong")

    
if __name__ == "__main__":
    correct_nodes = [20, 400, 8902, 197281, 4865609]
    gs = BitboardGameState()
    for depth in range(5):
        print(f"Depth {depth+1}: {bitboard_perft(gs, depth+1)}=={correct_nodes[depth]} nodes")
        # with open(f"logs/moves_depth{depth+1}.txt", "w") as f:
        #     bitboard_perft_sequences(gs, depth, f)
        
        # validate(depth+1)