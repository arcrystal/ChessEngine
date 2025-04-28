import numpy as np
from evaluate_board import evaluate_board
from engine_utils import generate_legal_moves, apply_move, undo_move
from game import GameState

class Engine:
    """
    A basic chess engine implementing Minimax search with alpha-beta pruning,
    iterative deepening, move ordering, and quiescence search for better evaluation
    of volatile positions.
    """
    def __init__(self, max_depth=5):
        """
        Initialize the engine.

        Args:
            max_depth (int): Maximum search depth (in plies) for full search.
        """
        self.max_depth = max_depth
        self.nodes_searched = 0

    def search(self, gs):
        """
        Perform iterative deepening search up to max_depth.

        Args:
            gs (GameState): Current board state.

        Returns:
            best_move (tuple): The best move found.
        """
        best_move = None
        best_eval = -np.inf if gs.white_to_move else np.inf

        for current_depth in range(1, self.max_depth + 1):
            eval_score, move = self.iterative_deepening(gs, current_depth)
            if move is not None:
                best_move = move
                best_eval = eval_score
            print(f"(Depth {current_depth}) Best Move: {self.to_standard_algebraic(best_move)}, Eval: {round(eval_score)}")

        print(f"Total nodes searched: {self.nodes_searched}")
        return best_move

    def iterative_deepening(self, gs, depth):
        """
        Search the position to a specified depth.

        Args:
            gs (GameState): Current board state.
            depth (int): Search depth.

        Returns:
            eval_score (float): Evaluation score.
            best_move (tuple): Best move at this depth.
        """
        self.nodes_searched = 0
        maximizing = gs.white_to_move
        eval_score, best_move = self._minimax(gs, depth, -np.inf, np.inf, maximizing)
        return eval_score, best_move

    def _minimax(self, gs, depth, alpha, beta, maximizing_player):
        """
        Minimax with alpha-beta pruning.

        Args:
            gs (GameState): Current board state.
            depth (int): Remaining depth.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            maximizing_player (bool): True if maximizing, False if minimizing.

        Returns:
            (eval_score, best_move) (tuple): Best evaluation and corresponding move.
        """
        self.nodes_searched += 1

        if depth == 0:
            return self.quiescence(gs, alpha, beta, maximizing_player), None

        moves = generate_legal_moves(gs)
        if not moves:
            return evaluate_board(gs), None

        moves = self.order_moves(gs, moves)

        best_move = None

        if maximizing_player:
            max_eval = -np.inf
            for move in moves:
                move_info = apply_move(gs, move)
                eval_score, _ = self._minimax(gs, depth - 1, alpha, beta, False)
                undo_move(gs, move, *move_info)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move

        else:
            min_eval = np.inf
            for move in moves:
                move_info = apply_move(gs, move)
                eval_score, _ = self._minimax(gs, depth - 1, alpha, beta, True)
                undo_move(gs, move, *move_info)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def quiescence(self, gs, alpha, beta, maximizing_player):
        """
        Quiescence search: extends the search on capture moves only,
        preventing horizon effect on tactical positions.

        Args:
            gs (GameState): Current board state.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            maximizing_player (bool): True if maximizing, False if minimizing.

        Returns:
            (float): Best evaluation score.
        """
        stand_pat = evaluate_board(gs)

        if maximizing_player:
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if stand_pat < beta:
                beta = stand_pat

        moves = self.generate_capture_moves(gs, include_checks=True)

        moves = self.order_moves(gs, moves)

        for move in moves:
            move_info = apply_move(gs, move)
            eval_score = -self.quiescence(gs, -beta, -alpha, not maximizing_player)
            undo_move(gs, move, *move_info)

            if maximizing_player:
                if eval_score >= beta:
                    return beta
                if eval_score > alpha:
                    alpha = eval_score
            else:
                if eval_score <= alpha:
                    return alpha
                if eval_score < beta:
                    beta = eval_score

        return alpha if maximizing_player else beta

    def generate_capture_moves(self, gs):
        """
        Generate only capture moves for quiescence search.

        Args:
            gs (GameState): Current board state.

        Returns:
            (list): List of capture moves.
        """
        moves = generate_legal_moves(gs)
        capture_moves = []
        king_pos = self.find_king(gs)

        for move in moves:
            from_r, from_c, to_r, to_c, promo = move
            captured = gs.board[to_r, to_c]
            if captured != 0:
                capture_moves.append(move)
            elif include_checks and self.is_check(gs, move, king_pos):
                capture_moves.append(move)

        return capture_moves

    def order_moves(self, gs, moves):
        """
        Heuristically order moves: prefer captures and promotions first.

        Args:
            gs (GameState): Current board state.
            moves (list): List of possible moves.

        Returns:
            (list): Moves ordered for better alpha-beta pruning.
        """
        scored_moves = []
        for move in moves:
            from_r, from_c, to_r, to_c, promo = move
            captured = gs.board[to_r, to_c]
            score = 0
            if captured != 0:
                score += 1000 + abs(captured)
            if promo != 0:
                score += 900 + promo
            scored_moves.append((score, move))
        scored_moves.sort(reverse=True)
        return [move for score, move in scored_moves]
    
    def to_standard_algebraic(self, move):
        """
        Convert a move tuple into human-readable chess notation (e.g., e2e4).

        Args:
            move (tuple): Move in (from_row, from_col, to_row, to_col, promotion_piece).

        Returns:
            (str): Move in algebraic notation.
        """
        files = 'abcdefgh'
        from_row, from_col, to_row, to_col, _ = move
        from_square = f"{files[from_col]}{8 - from_row}"
        to_square = f"{files[to_col]}{8 - to_row}"
        return f"{from_square} {to_square}"
    
    def order_moves(self, gs, moves):
    """Order moves: captures > promotions > checks > others."""
    scored_moves = []
    king_pos = self.find_king(gs)

    for move in moves:
        from_r, from_c, to_r, to_c, promo = move
        captured = gs.board[to_r, to_c]
        score = 0

        if captured != 0:
            score += 10_000 + abs(captured) - abs(gs.board[from_r, from_c])  # MVV-LVA
        elif promo != 0:
            score += 9_000 + promo
        elif self.is_check(gs, move, king_pos):
            score += 8_000
        else:
            score += 0  # quiet move

        scored_moves.append((score, move))

    scored_moves.sort(reverse=True)
    return [move for score, move in scored_moves]

def find_king(self, gs):
    """Find the position of the opponent's king for faster is_check detection."""
    king = KING if not gs.white_to_move else -KING
    for r in range(8):
        for c in range(8):
            if gs.board[r, c] == king:
                return (r, c)
    return (-1, -1)

def is_check(self, gs, move, king_pos):
    """Rough check detection: does this move attack opponent king square?"""
    _, _, to_r, to_c, _ = move
    return (to_r, to_c) == king_pos
        

# --- Example Runner ---
if __name__ == "__main__":
    import time
    game = GameState()
    start = time.time()
    engine = Engine(max_depth=7)  # Deeper search
    best_move = engine.search(game)
    print(f"Best move found in {round(time.time() - start)}s")