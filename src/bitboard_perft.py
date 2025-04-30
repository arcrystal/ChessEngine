from bitboard_game import BitboardGameState
from generate_moves import generate_all_moves_debug, generate_all_moves
import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        res = f"{func.__name__} executed in {end - start:.6f} seconds. {result} nodes. depth"
        return res
    return wrapper

counts_set = set()

def _bitboard_perft_debug(gs, depth):
    if depth == 0:
        return 1
    
    nodes = 0
    moves, moves_str = generate_all_moves_debug(gs, depth)
    counts_set.add(moves_str)
    for move in moves:
        gs.make_move(move)
        if not gs.is_in_check(not gs.white_to_move):
            nodes += _bitboard_perft_debug(gs, depth-1)
            
        gs.undo_move()
        
    return nodes

def _bitboard_perft(gs, depth):
    if depth == 0:
        return 1
    
    nodes = 0
    moves, moves_str = generate_all_moves(gs)
    counts_set.add(moves_str)
    for move in moves:
        gs.make_move(move)
        if not gs.is_in_check(not gs.white_to_move):
            nodes += _bitboard_perft(gs, depth-1)
            
        gs.undo_move()
        
    return nodes

@timeit
def bitboard_perft_debug(gs, depth):
    return _bitboard_perft_debug(gs, depth)


@timeit
def bitboard_perft(gs, depth):
    return _bitboard_perft(gs, depth)
    
if __name__ == "__main__":
    gs = BitboardGameState()
    print("\n")
    #print("Depth 1:", bitboard_perft(gs, 1), "nodes")  # Should print 20
    #print("Depth 2:", bitboard_perft(gs, 2), "nodes")  # Should print 400
    res = bitboard_perft_debug(gs, 3)  # Should print 8902
    # print("Depth 4:", bitboard_perft(gs, 4), "nodes")  # 
    # print("Depth 5:", bitboard_perft(gs, 5), "nodes")  # first en passant
    for moves_str in counts_set:
        print(moves_str, end="\n\n")
        
    print(f"{res} 3")