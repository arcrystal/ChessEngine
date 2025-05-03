from bitboard_game import BitboardGameState
from bitboard_gamestate_utils import attack_map_numba, apply_move_numba, update_occupancies_numba, undo_move_numba
from bitboard_nomagic import knight_attacks, king_attacks
from bitboard_magic import bishop_attacks, rook_attacks, queen_attacks
from bitboard_debugging import print_bitboard, get_standard_algebraic, print_board
from constants import board_str
from bitboard_utils import rank_mask, file_mask, bb_to_squares
from numba import uint64, boolean
from numba import types
from numba.typed import List
from generate_moves import *
import warnings
warnings.simplefilter("ignore")

print()

gs = BitboardGameState()

move_state_type = types.Tuple((
    uint64, uint64, uint64, uint64, uint64, uint64,   # white pieces
    uint64, uint64, uint64, uint64, uint64, uint64,   # black pieces
    types.UniTuple(types.int8, 4),                    # castling_rights
    types.int32,                                      # en_passant_target
    types.int32,                                      # halfmove_clock
    types.int32                                       # fullmove_number
))

n = 0
move_info = List.empty_list(move_state_type)
for mv1 in generate_all_moves(gs):
    move = apply_move_numba(gs, mv1)
    move_info.append(move)
    update_occupancies_numba(gs)
    gs.white_to_move = not gs.white_to_move
    for mv2 in generate_all_moves(gs):
        move = apply_move_numba(gs, mv1)
        move_info.append(move)
        update_occupancies_numba(gs)
        gs.white_to_move = not gs.white_to_move
        for mv3 in generate_all_moves(gs):
            n += 1

        undo_move_numba(gs, move_info)
        update_occupancies_numba(gs)
        gs.white_to_move = not gs.white_to_move


    undo_move_numba(gs, move_info)
    update_occupancies_numba(gs)
    gs.white_to_move = not gs.white_to_move

print("Moves @ depth=3:", n)
print()

gs = BitboardGameState()
move = apply_move_numba(gs, (10,26,0)) #g1h3
update_occupancies_numba(gs)
gs.white_to_move = not gs.white_to_move
move = apply_move_numba(gs, (51,43,0)) #g1h3
update_occupancies_numba(gs)
gs.white_to_move = not gs.white_to_move

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

print()
print_board(gs)
print()

print("Pawn moves bb:")
moves = (left_attacks | right_attacks | single_push | double_push)
squares = bb_to_squares(pawns)
print_bitboard(squares, moves)
print("Pawn generated moves:")
for m in generate_pawn_moves(gs, pawns, gs.white_to_move):
    print(get_standard_algebraic(m))
print()

rooks = gs.white_rooks if gs.white_to_move else gs.black_rooks
print("Rook moves bb:")
moves = 0
squares = bb_to_squares(rooks)
for sq in squares:
    moves |= rook_attacks(sq, gs.occupied) & ~occupancy
print_bitboard(squares, moves)
print("Rook generated moves:")
for m in generate_rook_moves(gs, rooks, gs.white_to_move):
    print(get_standard_algebraic(m))
print()

knights = gs.white_knights if gs.white_to_move else gs.black_knights
print("Knight moves bb:")
moves = 0
squares = bb_to_squares(knights)
for sq in squares:
    moves |= knight_attacks(sq) & ~occupancy

print_bitboard(squares, moves)
for m in generate_knight_moves(gs, knights, gs.white_to_move):
    print(get_standard_algebraic(m))
print()

bishops = gs.white_bishops if gs.white_to_move else gs.black_bishops
print("Bishop moves bb:")
squares = bb_to_squares(bishops)
moves = 0
for sq in squares:
    moves |= bishop_attacks(sq, gs.occupied) & ~occupancy
print_bitboard(squares, moves)
for m in generate_bishop_moves(gs, bishops, gs.white_to_move):
    print(get_standard_algebraic(m))
print()

queens = gs.white_queens if gs.white_to_move else gs.black_queens
print("Queen moves bb:")
squares = bb_to_squares(queens)
moves = 0
for sq in squares:
    moves |= queen_attacks(sq, gs.occupied) & ~occupancy
print_bitboard(squares, moves)
for m in generate_queen_moves(gs, queens, gs.white_to_move):
    print(get_standard_algebraic(m))
print()

king = gs.white_king if gs.white_to_move else gs.black_king
print("King moves bb:")
squares = bb_to_squares(king)
enemy_attacks = attack_map_numba(gs, not gs.white_to_move)
moves = 0
for sq in squares:
    moves |= king_attacks(sq) & ~occupancy & ~enemy_attacks
print_bitboard(squares, moves)
for m in generate_king_moves(gs, king, gs.white_to_move):
    print(get_standard_algebraic(m))
print()

print(board_str)