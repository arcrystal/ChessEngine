import numpy as np
import bitboards
from bitboards import (
    rank_mask, square_mask,
    pawn_attacks, knight_attacks, king_attacks,
    bishop_attacks, rook_attacks, queen_attacks
)
from constants import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING


class BitboardGameState:
    def __init__(self):
        # Piece bitboards
        self.white_pawns = np.uint64(0)
        self.white_knights = np.uint64(0)
        self.white_bishops = np.uint64(0)
        self.white_rooks = np.uint64(0)
        self.white_queens = np.uint64(0)
        self.white_king = np.uint64(0)

        self.black_pawns = np.uint64(0)
        self.black_knights = np.uint64(0)
        self.black_bishops = np.uint64(0)
        self.black_rooks = np.uint64(0)
        self.black_queens = np.uint64(0)
        self.black_king = np.uint64(0)

        # Overall occupancy
        self.white_occupancy = np.uint64(0)
        self.black_occupancy = np.uint64(0)
        self.occupied = np.uint64(0)

        # Game state info
        self.white_to_move = True
        self.castling_rights = [1, 1, 1, 1]  # [wK, wQ, bK, bQ]
        self.en_passant_target = -1  # Square index or -1
        self.halfmove_clock = 0
        self.fullmove_number = 1

        self._init_starting_position()

    def _init_starting_position(self):
        # Setup initial position using bitboards
        self.white_pawns   = np.uint64(0x000000000000FF00)
        self.black_pawns   = np.uint64(0x00FF000000000000)

        self.white_rooks   = np.uint64(0x0000000000000081)
        self.black_rooks   = np.uint64(0x8100000000000000)

        self.white_knights = np.uint64(0x0000000000000042)
        self.black_knights = np.uint64(0x4200000000000000)

        self.white_bishops = np.uint64(0x0000000000000024)
        self.black_bishops = np.uint64(0x2400000000000000)

        self.white_queens  = np.uint64(0x0000000000000008)
        self.black_queens  = np.uint64(0x0800000000000000)

        self.white_king    = np.uint64(0x0000000000000010)
        self.black_king    = np.uint64(0x1000000000000000)

        self.update_occupancies()
        
    def is_in_check(self, white: bool) -> bool:
        """Check if side `white` is in check."""
        king_bb = self.white_king if white else self.black_king
        king_sq = int(king_bb).bit_length() - 1
        enemy_pawns = self.black_pawns if white else self.white_pawns
        enemy_knights = self.black_knights if white else self.white_knights
        enemy_bishops = self.black_bishops if white else self.white_bishops
        enemy_rooks = self.black_rooks if white else self.white_rooks
        enemy_queens = self.black_queens if white else self.white_queens
        enemy_king = self.black_king if white else self.white_king
        occ = self.white_occupancy | self.black_occupancy

        if np.uint64(pawn_attacks(king_sq, not white)) & np.uint64(enemy_pawns): return True
        if np.uint64(knight_attacks(king_sq)) & np.uint64(enemy_knights): return True
        if np.uint64(bishop_attacks(king_sq, occ)) & np.uint64(enemy_bishops | enemy_queens): return True
        if np.uint64(rook_attacks(king_sq, occ)) & np.uint64(enemy_rooks | enemy_queens): return True
        if np.uint64(king_attacks(king_sq)) & np.uint64(enemy_king): return True
        return False
    
    def attack_map(self, is_white):
        occupied = self.white_occupancy | self.black_occupancy
        attacks = np.uint64(0)

        if is_white:
            pawns = self.white_pawns
            knights = self.white_knights
            bishops = self.white_bishops
            rooks = self.white_rooks
            queens = self.white_queens
            king = self.white_king
        else:
            pawns = self.black_pawns
            knights = self.black_knights
            bishops = self.black_bishops
            rooks = self.black_rooks
            queens = self.black_queens
            king = self.black_king

        while pawns:
            sq = (pawns & -pawns).bit_length() - 1
            attacks |= pawn_attacks(sq, is_white)
            pawns &= pawns - 1

        while knights:
            sq = (knights & -knights).bit_length() - 1
            attacks |= knight_attacks(sq)
            knights &= knights - 1

        while bishops:
            sq = (bishops & -bishops).bit_length() - 1
            attacks |= bishop_attacks(sq, occupied)
            bishops &= bishops - 1

        while rooks:
            sq = (rooks & -rooks).bit_length() - 1
            attacks |= rook_attacks(sq, occupied)
            rooks &= rooks - 1

        while queens:
            sq = (queens & -queens).bit_length() - 1
            attacks |= queen_attacks(sq, occupied)
            queens &= queens - 1

        if king:
            sq = (king & -king).bit_length() - 1
            attacks |= king_attacks(sq)

        return attacks
        
    def make_move(self, move):
        """Apply a move to the current state."""
        from_sq, to_sq, promo = move

        mover_bb = np.uint64(1) << np.uint64(from_sq)
        to_bb = np.uint64(1) << np.uint64(to_sq)

        moving_side = 'white' if self.white_to_move else 'black'
        opponent_side = 'black' if self.white_to_move else 'white'

        # --- Find and move the piece
        for piece_type in ['pawns', 'knights', 'bishops', 'rooks', 'queens', 'king']:
            bb = getattr(self, f"{moving_side}_{piece_type}")
            if bb & mover_bb:
                new_bb = (bb & ~mover_bb) | to_bb
                setattr(self, f"{moving_side}_{piece_type}", new_bb)
                break
        else:
            raise ValueError(f"No moving piece found at {from_sq}")

        # --- Handle captures
        for piece_type in ['pawns', 'knights', 'bishops', 'rooks', 'queens', 'king']:
            opp_bb = getattr(self, f"{opponent_side}_{piece_type}")
            if opp_bb & to_bb:
                new_opp_bb = opp_bb & ~to_bb
                setattr(self, f"{opponent_side}_{piece_type}", new_opp_bb)
                break

        # --- Handle promotions
        if promo:
            pawn_attr = f"{moving_side}_pawns"
            promote_to = {KNIGHT: "knights", BISHOP: "bishops", ROOK: "rooks", QUEEN: "queens"}[promo]

            # Remove pawn
            pawn_bb = getattr(self, pawn_attr)
            setattr(self, pawn_attr, pawn_bb & ~to_bb)

            # Add promoted piece
            promote_attr = f"{moving_side}_{promote_to}"
            promote_bb = getattr(self, promote_attr)
            setattr(self, promote_attr, promote_bb | to_bb)

        # --- Update occupancies and side
        self.update_occupancies()
        self.white_to_move = not self.white_to_move

    def update_occupancies(self):
        self.white_occupancy = (
            self.white_pawns | self.white_knights | self.white_bishops |
            self.white_rooks | self.white_queens | self.white_king
        )
        self.black_occupancy = (
            self.black_pawns | self.black_knights | self.black_bishops |
            self.black_rooks | self.black_queens | self.black_king
        )
        self.occupied = self.white_occupancy | self.black_occupancy
        
    def update_bitboards(self):
        """Update all occupancy bitboards after a move."""
        self.update_occupancies()
        
    def copy(self):
        """Return a deep copy of the game state."""
        new_state = BitboardGameState()
        new_state.white_pawns = self.white_pawns
        new_state.white_knights = self.white_knights
        new_state.white_bishops = self.white_bishops
        new_state.white_rooks = self.white_rooks
        new_state.white_queens = self.white_queens
        new_state.white_king = self.white_king

        new_state.black_pawns = self.black_pawns
        new_state.black_knights = self.black_knights
        new_state.black_bishops = self.black_bishops
        new_state.black_rooks = self.black_rooks
        new_state.black_queens = self.black_queens
        new_state.black_king = self.black_king

        new_state.white_occupancy = self.white_occupancy
        new_state.black_occupancy = self.black_occupancy
        new_state.occupied = self.occupied

        new_state.white_to_move = self.white_to_move
        new_state.castling_rights = list(self.castling_rights)
        new_state.en_passant_target = self.en_passant_target
        new_state.halfmove_clock = self.halfmove_clock
        new_state.fullmove_number = self.fullmove_number
        return new_state
    
    