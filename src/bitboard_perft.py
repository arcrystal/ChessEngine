import numpy as np
from bitboard_game import BitboardGameState
from generate_moves import generate_all_legal_moves

def bitboard_perft(gs, depth):
    if depth == 0:
        return 1
    
    nodes = 0
    for move in generate_all_legal_moves(gs):
        print(move)
        gs_copy = gs.copy()
        gs_copy.make_move(move)
        nodes += bitboard_perft(gs_copy, depth-1)
        
    return nodes

if __name__ == "__main__":
    gs = BitboardGameState()
    print("Perft depth 1:", bitboard_perft(gs, 1))  # Should print 20
    # print("Perft depth 2:", bitboard_perft(gs, 2))  # Should print 400
    # print("Perft depth 3:", bitboard_perft(gs, 3))  # Should print 8902