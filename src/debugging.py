def print_board(gs, return_str=False):
    """Print the current board with pieces."""
    # Create a list of piece symbols
    piece_symbols = {
        "white_pawns": "P", "black_pawns": "p",
        "white_knights": "N", "black_knights": "n",
        "white_bishops": "B", "black_bishops": "b",
        "white_rooks": "R", "black_rooks": "r",
        "white_queens": "Q", "black_queens": "q",
        "white_king": "K", "black_king": "k"
    }

    # Iterate over the board squares (0 to 63)
    board_str = ""
    for rank in range(7, -1, -1):
        row = ""
        for file in range(8):
            square_index = (rank * 8) + file
            piece_found = False

            # Check if the piece is on this square
            for piece_type in piece_symbols:
                bb = getattr(gs, piece_type)
                # Ensure that square_index is cast to np.uint64
                if bb & (1 << square_index):
                    row += piece_symbols[piece_type] + " "
                    piece_found = True
                    break

            # If no piece found, print a dot for an empty square
            if not piece_found:
                row += "* "
                
        if return_str:
            
            board_str += row + "\n"
        else:
            print(row)
    
    if return_str:
        return board_str
    
def print_bitboard(bitboard: int, label: str = ""):
    if label:
        print(f"{label}")
    print("  +-----------------+")
    for rank in range(7, -1, -1):
        row = f"{rank + 1} |"
        for file in range(8):
            square = rank * 8 + file
            row += " X" if (int(bitboard) >> square) & 1 else " ."
        row += " |"
        print(row)
    print("  +-----------------+")
    print("    a b c d e f g h\n")
    
def index_to_square(index):
    """Convert a 0-63 index to a chessboard square in algebraic notation."""
    rank = (index // 8) + 1
    file = chr(index % 8 + ord('a'))
    return f"{file}{rank}"
    
def get_standard_algebraic(move_or_loc):
    """Convert a list of moves from index notation to algebraic notation."""
    if isinstance(move_or_loc, tuple):
        from_sq, to_sq, promo = move_or_loc
        from_square_algebraic = index_to_square(from_sq)
        to_square_algebraic = index_to_square(to_sq)
        return f"{from_square_algebraic}{to_square_algebraic}"
    else:
        return index_to_square(move_or_loc)
    

