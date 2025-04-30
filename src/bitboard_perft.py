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
        # for i in range(0, len(sequence), 2):
        #     move_num = i // 2 + 1
        #     if i + 1 < len(sequence):
        #         outfile.write(f"{move_num}. {sequence[i]}, {sequence[i+1]} ")
        #     else:
        #         outfile.write(f"{move_num}. {sequence[i]}\n")
        outfile.write(str(sequence)+"\n")
        return 1

    nodes = 0
    moves = generate_all_moves(gs, verbose=False)
    for move in moves:
        # move_str = str(move)
        gs.make_move(move)
        if not gs.is_check(not gs.white_to_move):
            sequence.append(move)
            nodes += _bitboard_perft_sequences(gs, depth - 1, sequence, outfile)
            sequence.pop()
        gs.undo_move()

    return nodes

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
    gs = BitboardGameState()
    print("Depth 1:", bitboard_perft(gs, 1), "==20 nodes")
    print("Depth 2:", bitboard_perft(gs, 2), "==400 nodes") 
    print("Depth 3:", bitboard_perft(gs, 3), "==8902 nodes")
    print("Depth 4:", bitboard_perft(gs, 4), "==197281 nodes")
    print("Depth 5:", bitboard_perft(gs, 5), "==4865609 nodes")
    
    depth = 5
    with open(f"logs/moves_depth{depth}.txt", "w") as f:
        bitboard_perft_sequences(gs, depth, f)
        
    validate(depth)