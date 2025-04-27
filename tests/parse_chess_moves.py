# parse_text_to_game_format.py
import re

file_to_col = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
rank_to_row = {'1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0}

def square_to_coords(square):
    file, rank = square[0], square[1]
    return rank_to_row[rank], file_to_col[file]

def parse_test_file(filepath):
    moves = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.search(r"[abcdefgh]", line).start()
        line = line[match:]
        turn = line.split(", ")
        white = turn[0]
        if len(turn) > 1:
            black = turn[1]
        else:
            black = None
                    
        from_sq, to_sq = white.split()
        from_row, from_col = square_to_coords(from_sq)
        to_row, to_col = square_to_coords(to_sq)
        moves.append((from_row, from_col, to_row, to_col))
        if black is not None:
            from_sq, to_sq = black.split()
            from_row, from_col = square_to_coords(from_sq)
            to_row, to_col = square_to_coords(to_sq)
            moves.append((from_row, from_col, to_row, to_col))
        
    return moves

def generate_test_code(moves):
    output = []
    output.append("    simulate_moves(new_game, [")
    for i, move in enumerate(moves):
        if i%2==0:
            output.append(f"        {move},")
        else:
            output[-1] += f" {move},"
    output.append("    ])")
    return "\n".join(output)

if __name__ == "__main__":
    filepath = "moves.txt"  # <-- your input file
    moves = parse_test_file(filepath)
    code = generate_test_code(moves)
    print(code)