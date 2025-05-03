from numba import types, uint64
from numba.typed import List

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6
EMPTY = 7

board_str = """
  +-------------------------+
8 | 56 57 58 59 60 61 62 63 |
  |                         |
7 | 48 49 50 51 52 53 54 55 |
  |                         |
6 | 40 41 42 43 44 45 46 47 |
  |                         |
5 | 32 33 34 35 36 37 38 39 |
  |                         |
4 | 24 25 26 27 28 29 30 31 |
  |                         |
3 | 16 17 18 19 20 21 22 23 |
  |                         |
2 | 08 09 10 11 12 13 14 15 |
  |                         |
1 | 00 01 02 03 04 05 06 07 |
  +-------------------------+
    a  b  c  d  e  f  g  h
"""
# Define move triples
f2f3 = (13, 21, 0)
f2f4 = (13, 29, 0)
g2g3 = (14, 22, 0)
g2g4 = (14, 30, 0)
e7e5 = (51, 35, 0)
e7e6 = (51, 43, 0)
d8h4 = (59, 31, 0)

# Helper to create Numba list from Python tuple of 3 moves
def make_sequence(a, b, c):
    l = List.empty_list(types.UniTuple(types.int64, 3))
    l.append(a)
    l.append(b)
    l.append(c)
    return l

# Compose test mates
move4_mates = List.empty_list(types.ListType(types.UniTuple(types.int64, 3)))
for seq in [
    (f2f3, e7e6, g2g4),
    (f2f3, e7e5, g2g4),
    (f2f4, e7e6, g2g4),
    (f2f4, e7e5, g2g4),
    (g2g4, e7e6, f2f3),
    (g2g4, e7e6, f2f4),
    (g2g4, e7e5, f2f3),
    (g2g4, e7e5, f2f4)]:
    move4_mates.append(make_sequence(*seq))

move_state_type = types.Tuple((
    uint64, uint64, uint64, uint64, uint64, uint64,   # white pieces
    uint64, uint64, uint64, uint64, uint64, uint64,   # black pieces
    types.UniTuple(types.int8, 4),                    # castling_rights
    types.int64,                                      # en_passant_target
    types.int32,                                      # halfmove_clock
    types.int32                                       # fullmove_number
))

number_of_positions  = [1, 20, 400, 8902, 197281, 4865609, 119060324, 3195901860, 84998978956, 2439530234167]
number_of_checkmates = [0, 0,  0,   0,    8,      347,     10828,     435767,     9852036,     400191963]