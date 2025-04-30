from bitboard_game import BitboardGameState
from generate_moves import generate_bishop_moves

"""
8 56 57 58 59 60 61 62 63
7 48 59 50 51 52 53 54 55
6 40 42 42 43 44 45 46 47
5 32 33 34 35 36 37 38 39
4 24 25 26 27 28 29 30 31
3 16 17 18 19 20 21 22 23
2 08 09 10 11 12 13 14 15
1 00 01 02 03 04 05 06 07
  a  b  c  d  e  f  g  h
"""

gs = BitboardGameState()
from_sq = 2
gs.make_move((11, 27, 0))
gs.make_move((54, 38, 0))
generate_bishop_moves(gs, gs.white_bishops, gs.white_to_move, debug=True)
exit()
