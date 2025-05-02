from bitboard_game import BitboardGameState
from bitboard_gamestate_utils import attack_map_numba, apply_move_numba
from bitboard_nomagic import knight_attacks, king_attacks
from bitboard_magic import bishop_attacks, rook_attacks, queen_attacks
from debugging import print_bitboard, print_board, get_standard_algebraic
from constants import board_str
from bitboard_utils import rank_mask, file_mask

from generate_moves import *

gs = BitboardGameState()
apply_move_numba(gs, (11,27,0))
print_board(gs)

empty = ~gs.occupied
enemy = gs.black_occupancy if gs.white_to_move else gs.white_occupancy
pawns = gs.white_pawns if gs.white_to_move else gs.black_pawns
occupancy = gs.white_occupancy if gs.white_to_move else gs.black_occupancy
if gs.white_to_move:
    single_push = (pawns << 8) & empty
    double_push = ((single_push & rank_mask(2)) << 8) & empty
    left_attacks = (pawns << 7) & enemy & ~file_mask(7)
    right_attacks = (pawns << 9) & enemy & ~file_mask(0)
else:
    single_push = (pawns >> 8) & empty
    double_push = ((single_push & rank_mask(5)) >> 8) & empty
    left_attacks = (pawns >> 9) & enemy & ~file_mask(7)
    right_attacks = (pawns >> 7) & enemy & ~file_mask(0)


print("Pawn moves bb:")
moves = (left_attacks | right_attacks | single_push | double_push)
squares = [8,9,10,11,12,13,14,15] if gs.white_to_move else [48,49,50,51,52,53,54,55]
print_bitboard(squares, moves)
print("Pawn generated moves:")
for m in generate_pawn_moves(gs, pawns, gs.white_to_move):
    print(get_standard_algebraic(m))
print()

rooks = gs.white_rooks if gs.white_to_move else gs.black_rooks
print("Rook moves bb:")
sq = 0 if gs.white_to_move else 56
moves = rook_attacks(sq, gs.occupied) & ~occupancy
print_bitboard([sq], moves)
print("Rook generated moves:")
for m in generate_rook_moves(gs, rooks, gs.white_to_move):
    print(get_standard_algebraic(m))
print()

knights = gs.white_knights if gs.white_to_move else gs.black_knights
print("Knight moves bb:")
sq = 1 if gs.white_to_move else 57
moves = knight_attacks(sq) & ~occupancy
print_bitboard([sq], moves)
for m in generate_knight_moves(gs, knights, gs.white_to_move):
    print(get_standard_algebraic(m))
print()

bishops = gs.white_bishops if gs.white_to_move else gs. black_bishops
print("Bishop moves bb:")
sq = 2 if gs.white_to_move else 58
moves = bishop_attacks(sq, gs.occupied) & ~occupancy
print_bitboard([sq], moves)
for m in generate_bishop_moves(gs, bishops, gs.white_to_move):
    print(get_standard_algebraic(m))
print()

queen = gs.white_queens if gs.white_to_move else gs.black_queens
print("Queen moves bb:")
sq = 3 if gs.white_to_move else 59
moves = queen_attacks(sq, gs.occupied) & ~occupancy
print_bitboard([sq], moves)
for m in generate_queen_moves(gs, queen, gs.white_to_move):
    print(get_standard_algebraic(m))
print()

king = gs.white_king if gs.white_to_move else gs.black_king
print("King moves bb:")
sq = 4 if gs.white_to_move else 60
enemy_attacks = attack_map_numba(gs, not gs.white_to_move)
moves = king_attacks(sq) & ~occupancy & ~enemy_attacks
print_bitboard([sq], moves)
for m in generate_king_moves(gs, king, gs.white_to_move):
    print(get_standard_algebraic(m))
print()

print(board_str)