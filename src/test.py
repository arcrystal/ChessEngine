from bitboard_game import BitboardGameState
from bitboard_gamestate_utils import attack_map_numba, apply_move_numba
from bitboard_nomagic import knight_attacks, king_attacks
from bitboard_magic import bishop_attacks, rook_attacks, queen_attacks
from debugging import print_bitboard, print_board
from constants import board_str
from bitboard_utils import rank_mask, file_mask

gs = BitboardGameState()
apply_move_numba(gs, (11,27,0))
apply_move_numba(gs, (51,35,0))
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


print("Pawn attacks:")
moves = (left_attacks | right_attacks | single_push | double_push)
print_bitboard(moves)

print("Rook attacks:")
sq = 0 if gs.white_to_move else 56
moves = rook_attacks(sq, gs.occupied) & ~occupancy
print_bitboard(moves)

print("Knight attacks:")
sq = 1 if gs.white_to_move else 57
moves = knight_attacks(sq) & ~occupancy
print_bitboard(moves)

print("Bishop attacks:")
sq = 2 if gs.white_to_move else 58
moves = bishop_attacks(sq, gs.occupied) & ~occupancy
print_bitboard(moves)

print("Queen attacks:")
sq = 3 if gs.white_to_move else 59
moves = queen_attacks(sq, gs.occupied) & ~occupancy
print_bitboard(moves)

print("King attacks:")
sq = 4 if gs.white_to_move else 60
enemy_attacks = attack_map_numba(gs, not gs.white_to_move)
moves = king_attacks(sq) & ~occupancy & ~enemy_attacks
print_bitboard(moves)

print(board_str)