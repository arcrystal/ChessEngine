import numpy as np
from bitboard_game import BitboardGameState
from generate_moves import generate_all_legal_moves

def bitboard_perft(gs, depth):
    if depth == 0:
        return 1
    
    nodes = 0
    for move in generate_all_legal_moves(gs):
        print(to_standard_algebraic(move))
        gs_copy = gs.copy()
        gs_copy.make_move(move)
        nodes += bitboard_perft(gs_copy, depth-1)
        
    return nodes

def index_to_square(index):
    """Convert a 0-63 index to a chessboard square in algebraic notation."""
    rank = 8 - (index // 8)  # Row is 8 minus the index divided by 8
    file = chr(index % 8 + ord('a'))  # File is 'a' to 'h', based on index % 8
    return f"{file}{rank}"

def to_standard_algebraic(move):
    """Convert a list of moves from index notation to algebraic notation."""
    from_sq, to_sq, promo = move
    from_square_algebraic = index_to_square(from_sq)
    to_square_algebraic = index_to_square(to_sq)
    # Adding promotion notation if needed
    if promo != 0:
        promo_piece = {1: 'N', 2: 'B', 3: 'R', 4: 'Q'}.get(promo, '')
        move_notation = f"{from_square_algebraic} {to_square_algebraic} {promo_piece}"
    else:
        move_notation = f"{from_square_algebraic} {to_square_algebraic}"
        
    return move_notation
    

if __name__ == "__main__":
    gs = BitboardGameState()
    print("Perft depth 1:", bitboard_perft(gs, 1))  # Should print 20
    # print("Perft depth 2:", bitboard_perft(gs, 2))  # Should print 400
    # print("Perft depth 3:", bitboard_perft(gs, 3))  # Should print 8902