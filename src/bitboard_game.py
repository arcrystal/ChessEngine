import numpy as np
from numba import uint64, boolean, int8, int32
from numba.experimental import jitclass


bitboard_spec = [
    # Piece bitboards
    ('white_pawns', uint64), ('white_knights', uint64), ('white_bishops', uint64),
    ('white_rooks', uint64), ('white_queens', uint64), ('white_king', uint64),
    ('black_pawns', uint64), ('black_knights', uint64), ('black_bishops', uint64),
    ('black_rooks', uint64), ('black_queens', uint64), ('black_king', uint64),

    # Occupancy
    ('white_occupancy', uint64),
    ('black_occupancy', uint64),
    ('occupied', uint64),

    # Game info
    ('white_to_move', boolean),
    ('en_passant_target', int32),
    ('halfmove_clock', int32),
    ('fullmove_number', int32),

    # Castling rights as a 4-element array: [wK, wQ, bK, bQ]
    ('castling_rights', int8[:]),
]


@jitclass(bitboard_spec)
class BitboardGameState:
    def __init__(self):
        # White pieces
        self.white_pawns   = np.uint64(0x000000000000FF00)
        self.white_knights = np.uint64(0x0000000000000042)
        self.white_bishops = np.uint64(0x0000000000000024)
        self.white_rooks   = np.uint64(0x0000000000000081)
        self.white_queens  = np.uint64(0x0000000000000008)
        self.white_king    = np.uint64(0x0000000000000010)

        # Black pieces
        self.black_pawns   = np.uint64(0x00FF000000000000)
        self.black_knights = np.uint64(0x4200000000000000)
        self.black_bishops = np.uint64(0x2400000000000000)
        self.black_rooks   = np.uint64(0x8100000000000000)
        self.black_queens  = np.uint64(0x0800000000000000)
        self.black_king    = np.uint64(0x1000000000000000)

        # Occupancy
        self.white_occupancy = (
            self.white_pawns | self.white_knights | self.white_bishops |
            self.white_rooks | self.white_queens | self.white_king
        )
        self.black_occupancy = (
            self.black_pawns | self.black_knights | self.black_bishops |
            self.black_rooks | self.black_queens | self.black_king
        )
        self.occupied = self.white_occupancy | self.black_occupancy

        # Game info
        self.white_to_move = True
        self.en_passant_target = -1
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.castling_rights = np.ones(4, dtype=np.int8)  # w_kingside, w_queenside, b_kingside, b_queenside
    
    def undo_move(self):
        """
        Undo a move by restoring saved game state.
        """
        (
            self.white_pawns,
            self.white_knights,
            self.white_bishops,
            self.white_rooks,
            self.white_queens,
            self.white_king,
            self.black_pawns,
            self.black_knights,
            self.black_bishops,
            self.black_rooks,
            self.black_queens,
            self.black_king,
            self.white_occupancy,
            self.black_occupancy,
            self.occupied,
            self.white_to_move,
            self.castling_rights,
            self.en_passant_target,
            self.halfmove_clock,
            self.fullmove_number
        ) = self.move_info.pop()
        
    # def copy(self):
    #     """Return a deep copy of the game state."""
    #     new_state = BitboardGameState()
    #     new_state.white_pawns = self.white_pawns
    #     new_state.white_knights = self.white_knights
    #     new_state.white_bishops = self.white_bishops
    #     new_state.white_rooks = self.white_rooks
    #     new_state.white_queens = self.white_queens
    #     new_state.white_king = self.white_king

    #     new_state.black_pawns = self.black_pawns
    #     new_state.black_knights = self.black_knights
    #     new_state.black_bishops = self.black_bishops
    #     new_state.black_rooks = self.black_rooks
    #     new_state.black_queens = self.black_queens
    #     new_state.black_king = self.black_king

    #     new_state.white_occupancy = self.white_occupancy
    #     new_state.black_occupancy = self.black_occupancy
    #     new_state.occupied = self.occupied

    #     new_state.white_to_move = self.white_to_move
    #     new_state.castling_rights = list(self.castling_rights)
    #     new_state.en_passant_target = self.en_passant_target
    #     new_state.halfmove_clock = self.halfmove_clock
    #     new_state.fullmove_number = self.fullmove_number
    #     return new_state
    
    # def print_board(self, return_str=False):
    #     """Print the current board with pieces."""
    #     # Create a list of piece symbols
    #     piece_symbols = {
    #         "white_pawns": "P", "black_pawns": "p",
    #         "white_knights": "N", "black_knights": "n",
    #         "white_bishops": "B", "black_bishops": "b",
    #         "white_rooks": "R", "black_rooks": "r",
    #         "white_queens": "Q", "black_queens": "q",
    #         "white_king": "K", "black_king": "k"
    #     }

    #     # Iterate over the board squares (0 to 63)
    #     board_str = ""
    #     for rank in range(7, -1, -1):
    #         row = ""
    #         for file in range(8):
    #             square_index = (rank * 8) + file
    #             piece_found = False

    #             # Check if the piece is on this square
    #             for piece_type in piece_symbols:
    #                 bb = getattr(self, piece_type)
    #                 # Ensure that square_index is cast to np.uint64
    #                 if bb & (np.uint64(1) << np.uint64(square_index)):
    #                     row += piece_symbols[piece_type] + " "
    #                     piece_found = True
    #                     break

    #             # If no piece found, print a dot for an empty square
    #             if not piece_found:
    #                 row += "* "
                    
    #         if return_str:
                
    #             board_str += row + "\n"
    #         else:
    #             print(row)
        
    #     if return_str:
    #         return board_str
        
    # def print_bitboard(self, bitboard: int, label: str = ""):
    #     if label:
    #         print(f"{label}")
    #     print("  +-----------------+")
    #     for rank in range(7, -1, -1):
    #         row = f"{rank + 1} |"
    #         for file in range(8):
    #             square = rank * 8 + file
    #             row += " X" if (int(bitboard) >> square) & 1 else " ."
    #         row += " |"
    #         print(row)
    #     print("  +-----------------+")
    #     print("    a b c d e f g h\n")
        
    # def index_to_square(self, index):
    #     """Convert a 0-63 index to a chessboard square in algebraic notation."""
    #     rank = (index // 8) + 1
    #     file = chr(index % 8 + ord('a'))
    #     return f"{file}{rank}"
        
    # def get_standard_algebraic(self, move_or_loc):
    #     """Convert a list of moves from index notation to algebraic notation."""
    #     if isinstance(move_or_loc, tuple):
    #         from_sq, to_sq, promo = move_or_loc
    #         from_square_algebraic = self.index_to_square(from_sq)
    #         to_square_algebraic = self.index_to_square(to_sq)
    #         # Adding promotion notation if needed
    #         if promo != 0:
    #             promo_piece = {KNIGHT: 'N', BISHOP: 'B', ROOK: 'R', QUEEN: 'Q'}
    #             move_notation = f"{from_square_algebraic}{to_square_algebraic}{promo_piece}"
    #         else:
    #             move_notation = f"{from_square_algebraic}{to_square_algebraic}"
                
    #         return move_notation
    #     else:
    #         return self.index_to_square(move_or_loc)
        

