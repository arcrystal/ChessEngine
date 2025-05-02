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

