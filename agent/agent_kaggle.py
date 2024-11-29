%%writefile submission.py
from Chessnut import Game
import random
import math

def memoize(func):
    """
    A decorator to implement memoization with a configurable cache size.
    
    Args:
        func (callable): The function to be memoized
    
    Returns:
        callable: Memoized version of the function
    """
    cache = {}
    max_cache_size = 1000  # Configurable cache size limit
    
    def memoized_func(*args, **kwargs):
        # Create a hashable key from arguments
        key = str(args) + str(kwargs)
        
        # Check if result is in cache
        if key in cache:
            return cache[key]
        
        # Compute and cache result
        result = func(*args, **kwargs)
        
        # Manage cache size
        if len(cache) >= max_cache_size:
            cache.popitem()  # Remove oldest entry
        
        cache[key] = result
        return result
    
    return memoized_func


class StrategicChessAgent:
    """
    A resource-constrained chess agent implementing strategic decision-making
    with efficient computational approaches, equipped with deeper analysis
    techniques to identify zugzwang scenarios, create aggressive bottlenecks, 
    and incorporate long-term aggressive strategic planning to dominate the match.
    """
    # Piece value constants
    PIECE_VALUES = {
        'P': 100, 'N': 1320, 'B': 1330, 'R': 1500, 'Q': 1900, 'K': 20000,
        'p': -100, 'n': -1320, 'b': -1330, 'r': -1500, 'q': -1900, 'k': -20000
    }

    def __init__(self, max_search_depth=4):
        """
        Initialize the chess agent with configurable search parameters.
        
        Args:
            max_search_depth (int): Maximum depth for move evaluation
        """
        # Memoization cache for move evaluations
        self.move_cache = {}  
        # Probabilistic move prediction
        self.opponent_move_probability = {}  
        # Memoization cache specifically for move sequences
        self.move_sequence_cache = {}    
        self.max_search_depth = max_search_depth
        self.opening_moves = {
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1': 
            ['e2e4', 'd2d4', 'c2c4', 'g1f3']
        }

    def evaluate_move(self, game, move):
        """
        Evaluate a potential move based on multiple strategic factors,
        emphasizing immediate tactical advantages, aggressive bottleneck creation,
        long-term strategic planning, and aggressive defense from the opening to the endgame.
        
        Args:
            game (Game): Current game state
            move (str): Move to evaluate
        
        Returns:
            float: Move evaluation score
        """
        score = 0

        try:
            test_game = Game(game.get_fen())
            test_game.apply_move(move)

            # Detect if the game is in the endgame stage
            in_endgame = self.is_endgame(test_game, game.turn)

            # 1. Checkmate detection (highest priority)
            if test_game.status == Game.CHECKMATE:
                return float('inf') if test_game.turn == game.turn else -float('inf')

            # 2. Capture evaluation with increased weights
            captured_piece = game.board.get_piece(Game.xy2i(move[2:4]))
            if captured_piece != ' ':
                capture_bonus = 500 + self.PIECE_VALUES[captured_piece]
                score += capture_bonus  # Significant bonus for captures

            # 3. Threats evaluation with increased emphasis
            threats_score = self.evaluate_threats(test_game, game.turn)
            score += threats_score * (1.5 if not in_endgame else 2.0)  # Increased weight for threats, more in endgame

            # 4. Tactical Patterns: Forks, Pins, Skewers, Double Threats, Discovered Attacks
            tactical_score = self.evaluate_tactical_patterns(test_game, game.turn)
            score += tactical_score * (2 if not in_endgame else 3)  # Higher weight for tactical patterns, more in endgame

            # 5. Check and Checkmate threats
            if self.is_check(test_game, game.turn):
                score += 800 if not in_endgame else 1600  # Higher bonus in endgame

            # 6. Piece Activity: Mobility with increased weight for active pieces
            mobility_score = self.evaluate_mobility(test_game, game.turn)
            score += mobility_score * (1.2 if not in_endgame else 1.5)  # Slightly higher weight in endgame

            # 7. King Safety: Maintain sufficient consideration
            king_safety_score = self.evaluate_king_safety(test_game, game.turn)
            score += king_safety_score  # Maintain original weight

            # 8. Pawn Structure: Further reduced emphasis in endgame
            pawn_structure_score = self.evaluate_pawn_structure(test_game, game.turn, in_endgame)
            score += pawn_structure_score * (0.8 if not in_endgame else 1.2)  # Lower weight normally, higher in endgame

            # 9. Positional Factors: Enhanced evaluation in endgame
            positional_score = self.evaluate_positional_factors(test_game, game.turn, in_endgame)
            score += positional_score * (0.8 if not in_endgame else 1.5)  # Lower weight normally, higher in endgame

            # 10. Endgame Considerations: Enhanced in endgame
            endgame_score = self.evaluate_endgame(test_game, game.turn, in_endgame)
            score += endgame_score

            # 11. Minimax Search Evaluation: Incorporate deeper analysis
            minimax_score = self.minimax(test_game, self.max_search_depth, -math.inf, math.inf, False)
            score += minimax_score * 1.0  # Weight of minimax evaluation

            # 12. Bottleneck Creation: Aggressively reward moves that create bottlenecks
            bottleneck_score = self.evaluate_bottleneck(test_game, game.turn)
            score += bottleneck_score * (8.0 if not in_endgame else 16.0)  # Higher weight for aggressive bottlenecking

            # 13. Aggressive Defense: Reward moves that strengthen defenses
            aggressive_defense_score = self.evaluate_aggressive_defense(test_game, game.turn, in_endgame)
            score += aggressive_defense_score * (1.5 if in_endgame else 1.0)  # Higher weight in endgame

            # 14. Long-term Strategic Planning: Encourage moves that set up future dominance
            strategic_planning_score = self.evaluate_strategic_planning(test_game, game.turn)
            score += strategic_planning_score * 2.0  # Significant weight for long-term planning

            # 15. Piece value consideration
            piece = game.board.get_piece(Game.xy2i(move[0:2]))
            score += self.PIECE_VALUES.get(piece, 0)

            # 16. Random factor to prevent predictability
            score += random.random() * 20

        except Exception as e:
            # Fallback scoring if move evaluation fails
            score = random.random() * 100

        return score

    def minimax(self, game, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with Alpha-Beta pruning to evaluate move strength.
        
        Args:
            game (Game): Current game state
            depth (int): Current depth in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing, False if minimizing
        
        Returns:
            float: Evaluation score of the game state
        """
        if depth == 0 or game.status in [Game.CHECKMATE, Game.DRAW]:
            return self.evaluate_game_state(game, not maximizing_player)

        if maximizing_player:
            max_eval = -math.inf
            for move in game.get_moves():
                new_game = Game(game.get_fen())
                new_game.apply_move(move)
                eval = self.minimax(new_game, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval
        else:
            min_eval = math.inf
            for move in game.get_moves():
                new_game = Game(game.get_fen())
                new_game.apply_move(move)
                eval = self.minimax(new_game, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval

    def evaluate_game_state(self, game, maximizing_player):
        """
        Evaluate the game state from the perspective of the agent.
        
        Args:
            game (Game): Current game state
            maximizing_player (bool): True if evaluating for maximizing player, else False
        
        Returns:
            float: Evaluation score
        """
        in_endgame = self.is_endgame(game, game.turn)
        score = 0

        # Material evaluation
        material = self.calculate_material(game, game.turn)
        for piece, value in self.PIECE_VALUES.items():
            score += game.board.pieces.count(piece) * value

        # King Safety
        king_safety = self.evaluate_king_safety(game, game.turn)
        score += king_safety

        # Pawn Structure
        pawn_structure = self.evaluate_pawn_structure(game, game.turn, in_endgame)
        score += pawn_structure * (0.8 if not in_endgame else 1.2)

        # Positional Factors
        positional = self.evaluate_positional_factors(game, game.turn, in_endgame)
        score += positional * (0.8 if not in_endgame else 1.5)

        # Endgame Factors
        endgame = self.evaluate_endgame(game, game.turn, in_endgame)
        score += endgame

        # Bottleneck Creation
        bottleneck = self.evaluate_bottleneck(game, game.turn)
        score += bottleneck * (2.0 if not in_endgame else 3.0)

        # Aggressive Defense
        aggressive_defense = self.evaluate_aggressive_defense(game, game.turn, in_endgame)
        score += aggressive_defense * (1.5 if in_endgame else 1.0)

        # Long-term Strategic Planning
        strategic_planning = self.evaluate_strategic_planning(game, game.turn)
        score += strategic_planning * 2.0  # Significant weight for long-term planning

        return score

    def evaluate_strategic_planning(self, game, turn):
        """
        Evaluate long-term strategic planning factors such as pawn chain formation,
        piece coordination, and potential for controlling multiple key squares over several moves.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            float: Strategic planning score
        """
        strategic_score = 0

        # Pawn Chain Strength
        pawn_chains = self.evaluate_pawn_chains(game, turn)
        strategic_score += pawn_chains * 50  # Reward for strong pawn chains

        # Piece Coordination: How well pieces support each other
        piece_coordination = self.evaluate_piece_coordination(game, turn)
        strategic_score += piece_coordination * 40  # Reward for good coordination

        # Control of Multiple Key Squares
        key_control = self.evaluate_multiple_key_control(game, turn)
        strategic_score += key_control * 60  # Significant reward for controlling multiple key squares

        return strategic_score

    def evaluate_pawn_chains(self, game, turn):
        """
        Evaluate the strength of pawn chains.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            int: Pawn chain strength score
        """
        score = 0
        pawns = [pos for pos in game.board.positions('P' if turn == 'w' else 'p')]
        pawn_positions = sorted(pawns, key=lambda pos: Game.i2xy(pos)[1], reverse=(turn == 'w'))

        for pawn in pawn_positions:
            x, y = Game.i2xy(pawn)
            # Check for connected pawns
            adjacent_files = [x - 1, x + 1]
            for af in adjacent_files:
                if 0 <= af < 8:
                    behind_pos = Game.xy2i(f"{chr(af + ord('a'))}{y - 1 + (1 if turn == 'w' else -1)}")
                    if behind_pos in pawns:
                        score += 20  # Reward for connected pawns
        return score

    def evaluate_piece_coordination(self, game, turn):
        """
        Evaluate how well the pieces support each other.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            int: Piece coordination score
        """
        score = 0
        if turn == 'w':
            pieces = [pos for pos in game.board.positions('N') + game.board.positions('B') + 
                      game.board.positions('R') + game.board.positions('Q') + game.board.positions('K')]
        else:
            pieces = [pos for pos in game.board.positions('n') + game.board.positions('b') + 
                      game.board.positions('r') + game.board.positions('q') + game.board.positions('k')]
        
        for piece in pieces:
            attacks = game.get_piece_attacks(game.board.get_piece(piece), piece)
            for target in attacks:
                target_piece = game.board.get_piece(target)
                if target_piece.lower() in ['n', 'b', 'r', 'q', 'k']:
                    score += 10  # Reward for attacking and supporting other pieces
        return score

    def evaluate_multiple_key_control(self, game, turn):
        """
        Evaluate control over multiple key squares.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            int: Multiple key squares control score
        """
        controlled_keys = self.get_key_squares(game, turn)
        if len(controlled_keys) >= 6:
            return 150  # Major reward for controlling multiple key squares
        elif len(controlled_keys) == 5:
            return 120
        elif len(controlled_keys) == 4:
            return 100
        elif len(controlled_keys) == 3:
            return 75
        elif len(controlled_keys) == 2:
            return 50
        elif len(controlled_keys) == 1:
            return 25
        return 0

    def is_endgame(self, game, turn):
        """
        Determine if the current game state is in the endgame phase.
        A simple heuristic: few pieces on the board, especially queens and rooks.

        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b' indicating which player's turn it is

        Returns:
            bool: True if in endgame, else False
        """
        # Count the number of major and minor pieces
        major_pieces = ['Q', 'R', 'q', 'r']
        minor_pieces = ['N', 'B', 'n', 'b']
        queens = sum(1 for piece in game.board.pieces if piece in ['Q', 'q'])
        rooks = sum(1 for piece in game.board.pieces if piece in ['R', 'r'])
        bishops = sum(1 for piece in game.board.pieces if piece in ['B', 'b'])
        knights = sum(1 for piece in game.board.pieces if piece in ['N', 'n'])
        # Define endgame as having few major pieces and limited minor pieces
        if queens + rooks <= 2 and bishops + knights <= 2:
            return True
        return False

    def is_check(self, game, turn):
        """
        Check if the opponent is in check after the move.

        Args:
            game (Game): Current game state after the move
            turn (str): 'w' or 'b' indicating which player's turn it is

        Returns:
            bool: True if opponent is in check, else False
        """
        opponent = 'b' if turn == 'w' else 'w'
        return game.is_in_check(opponent)

    def evaluate_tactical_patterns(self, game, turn):
        """
        Evaluate tactical patterns such as forks, pins, skewers, double threats, and discovered attacks.

        Args:
            game (Game): Current game state after the move
            turn (str): 'w' or 'b' indicating which player's turn it is

        Returns:
            float: Tactical patterns score
        """
        tactical_score = 0
        opponent = 'b' if turn == 'w' else 'w'
        
        # Iterate through all pieces to find tactical motifs
        for piece_type in ['N', 'B', 'R', 'Q'] if turn == 'w' else ['n', 'b', 'r', 'q']:
            for pos in game.board.positions(piece_type):
                attacks = game.get_piece_attacks(piece_type, pos)
                for target in attacks:
                    target_piece = game.board.get_piece(target)
                    if target_piece != ' ' and (target_piece.islower() if turn == 'w' else target_piece.isupper()):
                        # Potential for double threat
                        if self.is_double_threat(game, pos, target, turn):
                            tactical_score += 120
                        # Potential for fork
                        if self.is_fork(game, pos, target, turn):
                            tactical_score += 100
                        # Potential for pin
                        if self.is_pin(game, pos, target, turn):
                            tactical_score += 80
                        # Potential for skewer
                        if self.is_skewer(game, pos, target, turn):
                            tactical_score += 80
                        # Potential for discovered attack
                        if self.is_discovered_attack(game, pos, target, turn):
                            tactical_score += 90
        return tactical_score

    def is_fork(self, game, attacker_pos, target_pos, turn):
        """
        Determine if moving to target_pos results in a fork.

        Args:
            game (Game): Current game state
            attacker_pos (int): Position of the attacking piece
            target_pos (int): Position being attacked

        Returns:
            bool: True if it's a fork, else False
        """
        # Fork detection: One piece attacking two or more valuable targets
        attacking_piece = game.board.get_piece(attacker_pos)
        if attacking_piece.lower() not in ['n', 'b', 'r', 'q']:
            return False

        # Get all attacks from the attacking piece
        attacks = game.get_piece_attacks(attacking_piece, attacker_pos)
        # Count how many high-value pieces are under attack
        high_value_targets = 0
        for pos in attacks:
            piece = game.board.get_piece(pos)
            if piece != ' ' and (piece.islower() if turn == 'w' else piece.isupper()):
                if self.PIECE_VALUES.get(piece.upper(), 0) >= 300:  # Considering N, B, R, Q as high-value
                    high_value_targets += 1
        return high_value_targets >= 2

    def is_pin(self, game, attacker_pos, target_pos, turn):
        """
        Determine if moving to target_pos results in a pin.

        Args:
            game (Game): Current game state
            attacker_pos (int): Position of the attacking piece
            target_pos (int): Position being attacked

        Returns:
            bool: True if it's a pin, else False
        """
        # Pin detection: Attacker is a bishop, rook, or queen aligning with a more valuable piece
        attacker_piece = game.board.get_piece(attacker_pos)
        target_piece = game.board.get_piece(target_pos)
        if attacker_piece.lower() not in ['b', 'r', 'q']:
            return False

        # Determine direction from attacker to target
        direction = self.get_direction(attacker_pos, target_pos)
        if direction is None:
            return False

        # Look beyond the target to see if a higher-value piece is aligned
        next_pos = target_pos + direction
        while 0 <= next_pos < 64:
            beyond_piece = game.board.get_piece(next_pos)
            if beyond_piece == ' ':
                next_pos += direction
                continue
            if (beyond_piece.isupper() and turn == 'w') or (beyond_piece.islower() and turn == 'b'):
                # If beyond_piece is a higher-value piece, it's a pin
                if self.PIECE_VALUES.get(beyond_piece.upper(), 0) > self.PIECE_VALUES.get(target_piece.upper(), 0):
                    return True
            break
        return False

    def is_skewer(self, game, attacker_pos, target_pos, turn):
        """
        Determine if moving to target_pos results in a skewer.

        Args:
            game (Game): Current game state
            attacker_pos (int): Position of the attacking piece
            target_pos (int): Position being attacked

        Returns:
            bool: True if it's a skewer, else False
        """
        # Skewer detection: More valuable piece is in front and is attacked, exposing a less valuable piece behind
        attacker_piece = game.board.get_piece(attacker_pos)
        target_piece = game.board.get_piece(target_pos)
        if attacker_piece.lower() not in ['b', 'r', 'q']:
            return False

        # Determine direction from attacker to target
        direction = self.get_direction(attacker_pos, target_pos)
        if direction is None:
            return False

        # Check if target_piece is more valuable than any piece behind it in the same line
        next_pos = target_pos + direction
        while 0 <= next_pos < 64:
            beyond_piece = game.board.get_piece(next_pos)
            if beyond_piece == ' ':
                next_pos += direction
                continue
            if (beyond_piece.isupper() and turn == 'w') or (beyond_piece.islower() and turn == 'b'):
                # If beyond_piece is a higher-value piece, it's a skewer
                if self.PIECE_VALUES.get(beyond_piece.upper(), 0) > self.PIECE_VALUES.get(target_piece.upper(), 0):
                    return True
            break
        return False

    def is_discovered_attack(self, game, attacker_pos, target_pos, turn):
        """
        Determine if moving to target_pos results in a discovered attack.

        Args:
            game (Game): Current game state
            attacker_pos (int): Position of the attacking piece before move
            target_pos (int): Position being attacked after move

        Returns:
            bool: True if it's a discovered attack, else False
        """
        # Discovered attack detection: After moving a piece, another piece reveals an attack
        # For simplicity, compare attacks before and after the move

        # Capture the current attacks before the move
        current_attacks = self.get_all_attacks(game, turn)

        # Simulate the move
        new_game = Game(game.get_fen())
        new_game.apply_move(move=target_pos)  # Assuming 'move' leads to 'target_pos'

        # Get attacks after the move
        new_attacks = self.get_all_attacks(new_game, turn)

        # Discovered attack if new attacks include additional targets
        discovered_attacks = new_attacks - current_attacks
        return len(discovered_attacks) > 0

    def get_all_attacks(self, game, turn):
        """
        Get all squares currently under attack by the player.

        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'

        Returns:
            set: Set of squares under attack
        """
        attacks = set()
        for piece_type in ['P', 'N', 'B', 'R', 'Q', 'K'] if turn == 'w' else ['p', 'n', 'b', 'r', 'q', 'k']:
            for pos in game.board.positions(piece_type):
                piece = game.board.get_piece(pos)
                attacks.update(game.get_piece_attacks(piece, pos))
        return attacks

    def is_double_threat(self, game, attacker_pos, target_pos, turn):
        """
        Determine if moving to target_pos creates a double threat.

        Args:
            game (Game): Current game state
            attacker_pos (int): Position of the attacking piece
            target_pos (int): Position being attacked

        Returns:
            bool: True if it's a double threat, else False
        """
        # Double threat: A move that threatens two different targets simultaneously
        attacking_piece = game.board.get_piece(attacker_pos)
        if attacking_piece.lower() not in ['n', 'b', 'r', 'q']:
            return False

        # Get all attacks from the attacking piece
        attacks = game.get_piece_attacks(attacking_piece, attacker_pos)
        # Count how many high-value pieces are under attack
        high_value_targets = 0
        for pos in attacks:
            piece = game.board.get_piece(pos)
            if piece != ' ' and (piece.islower() if turn == 'w' else piece.isupper()):
                if self.PIECE_VALUES.get(piece.upper(), 0) >= 300:  # Considering N, B, R, Q as high-value
                    high_value_targets += 1
        return high_value_targets >= 2

    def evaluate_mobility(self, game, turn):
        """
        Evaluate the mobility of pieces, especially knights and bishops.
        Mobility is the number of legal moves available to a player's pieces.
        
        Args:
            game (Game): Current game state after the move
            turn (str): 'w' or 'b' indicating which player's turn it is
        
        Returns:
            float: Mobility score
        """
        mobility_score = 0
        for piece_type in ['N', 'B', 'R', 'Q', 'K'] if turn == 'w' else ['n', 'b', 'r', 'q', 'k']:
            for pos in game.board.positions(piece_type):
                piece = game.board.get_piece(pos)
                moves = game.get_piece_moves(piece, pos)
                mobility_score += len(moves) * 10  # Weight mobility
        return mobility_score

    def evaluate_threats(self, game, turn):
        """
        Evaluate threats created by the move, such as attacking opponent's pieces or king.
        
        Args:
            game (Game): Current game state after the move
            turn (str): 'w' or 'b' indicating which player's turn it is
        
        Returns:
            float: Threats score
        """
        threats_score = 0
        opponent = 'b' if turn == 'w' else 'w'
        for piece_type in ['N', 'B', 'R', 'Q', 'P'] if turn == 'w' else ['n', 'b', 'r', 'q', 'p']:
            for pos in game.board.positions(piece_type):
                threats = game.get_piece_attacks(piece_type, pos)
                for target in threats:
                    target_piece = game.board.get_piece(target)
                    if target_piece != ' ' and (target_piece.islower() if turn == 'w' else target_piece.isupper()):
                        threats_score += 50  # Bonus for threatening a piece
                    if game.is_attacked(target, opponent):
                        threats_score += 100  # Bonus for threatening the king
        return threats_score

    def evaluate_king_safety(self, game, turn):
        """
        Evaluate the safety of the king by assessing exposure and pawn shield.
        
        Args:
            game (Game): Current game state after the move
            turn (str): 'w' or 'b' indicating which player's turn it is
        
        Returns:
            float: King safety score
        """
        safety_score = 0
        king_pos = game.find_king(turn)
        if not king_pos:
            return safety_score  # Should not happen, but safe guard

        # King Shelter: Reward if king has more pawn shields
        shield_pawns = self.get_pawn_shield(game, king_pos, turn)
        safety_score += len(shield_pawns) * 20  # Reward for each pawn in shield

        # Exposure: Penalize if king is under attack or has open squares around
        if game.is_attacked(king_pos, 'b' if turn == 'w' else 'w'):
            safety_score -= 300  # High penalty if king is in check

        # Additional exposure assessment (e.g., open squares around king)
        surrounding_squares = self.get_surrounding_squares(game, king_pos)
        open_squares = [sq for sq in surrounding_squares if game.board.get_piece(sq) == ' ']
        safety_score -= len(open_squares) * 10  # Penalize for each open square around king

        return safety_score

    def get_pawn_shield(self, game, king_pos, turn):
        """
        Get the pawns that are part of the king's shield.
        
        Args:
            game (Game): Current game state
            king_pos (int): Position of the king
            turn (str): 'w' or 'b'
        
        Returns:
            list: Positions of shield pawns
        """
        # Simplistic approach: pawns directly in front and diagonally
        x, y = Game.i2xy(king_pos)
        directions = [(-1, 1), (0, 1), (1, 1)] if turn == 'w' else [(-1, -1), (0, -1), (1, -1)]
        shield_pawns = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                pos = Game.xy2i(f"{chr(nx + ord('a'))}{ny + 1}")
                piece = game.board.get_piece(pos)
                expected_pawn = 'P' if turn == 'w' else 'p'
                if piece == expected_pawn:
                    shield_pawns.append(pos)
        return shield_pawns

    def get_surrounding_squares(self, game, pos):
        """
        Get squares surrounding a given position.
        
        Args:
            game (Game): Current game state
            pos (int): Position to find surrounding squares
        
        Returns:
            list: Positions surrounding the given square
        """
        surrounding = []
        x, y = Game.i2xy(pos)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    surrounding.append(Game.xy2i(f"{chr(nx + ord('a'))}{ny + 1}"))
        return surrounding

    def evaluate_pawn_structure(self, game, turn, in_endgame):
        """
        Evaluate the pawn structure, prioritizing passed pawns and avoiding weak squares.
        Enhanced prioritization in the endgame.
        
        Args:
            game (Game): Current game state after the move
            turn (str): 'w' or 'b' indicating which player's turn it is
            in_endgame (bool): True if in endgame phase, else False
        
        Returns:
            float: Pawn structure score
        """
        pawn_score = 0
        pawns = [pos for pos in game.board.positions('P' if turn == 'w' else 'p')]
        
        for pawn in pawns:
            # Passed Pawns: No opposing pawns can block or capture it
            if self.is_passed_pawn(game, pawn, turn):
                pawn_score += 100  # Significant bonus for passed pawn
                if in_endgame:
                    pawn_score += 50  # Additional bonus in endgame

            # Avoid creating weak squares (doubled, isolated, backward)
            if self.is_doubled_pawn(game, pawn, turn):
                pawn_score -= 50  # Penalty for doubled pawn
            if self.is_isolated_pawn(game, pawn, turn):
                pawn_score -= 50  # Penalty for isolated pawn
            if self.is_backward_pawn(game, pawn, turn):
                pawn_score -= 50  # Penalty for backward pawn

        # Pawn Majority on a Side
        pawn_majority_score = self.evaluate_pawn_majority(game, turn)
        pawn_score += pawn_majority_score

        return pawn_score

    def is_passed_pawn(self, game, pawn_pos, turn):
        """
        Check if a pawn is a passed pawn.
        
        Args:
            game (Game): Current game state
            pawn_pos (int): Position of the pawn
            turn (str): 'w' or 'b'
        
        Returns:
            bool: True if passed pawn, else False
        """
        x, y = Game.i2xy(pawn_pos)
        if turn == 'w':
            for dy in range(y+1, 8):
                for dx in [x-1, x, x+1]:
                    if 0 <= dx < 8:
                        pos = Game.xy2i(f"{chr(dx + ord('a'))}{dy + 1}")
                        if game.board.get_piece(pos).lower() == 'p':
                            return False
        else:
            for dy in range(y-1, -1, -1):
                for dx in [x-1, x, x+1]:
                    if 0 <= dx < 8:
                        pos = Game.xy2i(f"{chr(dx + ord('a'))}{dy + 1}")
                        if game.board.get_piece(pos).lower() == 'p':
                            return False
        return True

    def is_doubled_pawn(self, game, pawn_pos, turn):
        """
        Check if a pawn is doubled.
        
        Args:
            game (Game): Current game state
            pawn_pos (int): Position of the pawn
            turn (str): 'w' or 'b'
        
        Returns:
            bool: True if doubled pawn, else False
        """
        x, y = Game.i2xy(pawn_pos)
        file = x
        pawns = [pos for pos in game.board.positions('P' if turn == 'w' else 'p') if Game.i2xy(pos)[0] == file]
        return len(pawns) > 1

    def is_isolated_pawn(self, game, pawn_pos, turn):
        """
        Check if a pawn is isolated.
        
        Args:
            game (Game): Current game state
            pawn_pos (int): Position of the pawn
            turn (str): 'w' or 'b'
        
        Returns:
            bool: True if isolated pawn, else False
        """
        x, y = Game.i2xy(pawn_pos)
        adjacent_files = [x-1, x+1]
        has_support = False
        for af in adjacent_files:
            if 0 <= af < 8:
                for dy in range(y, 8 if turn == 'w' else -1, 1 if turn == 'w' else -1):
                    pos = Game.xy2i(f"{chr(af + ord('a'))}{dy + 1}")
                    if game.board.get_piece(pos).lower() == 'p':
                        has_support = True
                        break
            if has_support:
                break
        return not has_support

    def is_backward_pawn(self, game, pawn_pos, turn):
        """
        Check if a pawn is backward.
        
        Args:
            game (Game): Current game state
            pawn_pos (int): Position of the pawn
            turn (str): 'w' or 'b'
        
        Returns:
            bool: True if backward pawn, else False
        """
        x, y = Game.i2xy(pawn_pos)
        target_y = y + 1 if turn == 'w' else y - 1
        if not (0 <= target_y < 8):
            return False
        # Check if pawn can be advanced
        forward_pos = Game.xy2i(f"{chr(x + ord('a'))}{target_y + 1}")
        if game.board.get_piece(forward_pos) == ' ':
            return False
        return True

    def evaluate_pawn_majority(self, game, turn):
        """
        Evaluate pawn majority on a particular side of the board.

        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'

        Returns:
            float: Pawn majority score
        """
        majority_score = 0
        # Define left, center, and right zones
        zones = {
            'left': range(0, 3),    # Files a, b, c
            'center': range(3, 6),  # Files d, e, f
            'right': range(6, 8)    # Files g, h
        }
        # Count pawns in each zone
        for zone_name, files in zones.items():
            own_pawns = len([pos for pos in game.board.positions('P' if turn == 'w' else 'p') 
                             if Game.i2xy(pos)[0] in files])
            opponent_pawns = len([pos for pos in game.board.positions('p' if turn == 'w' else 'P') 
                                  if Game.i2xy(pos)[0] in files])
            if own_pawns > opponent_pawns:
                majority_score += (own_pawns - opponent_pawns) * 10  # Reward for pawn majority
            elif own_pawns < opponent_pawns:
                majority_score -= (opponent_pawns - own_pawns) * 10  # Penalize for pawn deficit
        return majority_score

    def evaluate_positional_factors(self, game, turn, in_endgame):
        """
        Evaluate positional factors such as control of the center, open files, diagonals, space advantage,
        control of key squares, and potential for creating zugzwang. Enhanced evaluation in endgame.

        Args:
            game (Game): Current game state after the move
            turn (str): 'w' or 'b' indicating which player's turn it is
            in_endgame (bool): True if in endgame phase, else False

        Returns:
            float: Positional factors score
        """
        positional_score = 0

        # Control of the Center
        center_squares = ['d4', 'd5', 'e4', 'e5']
        for square in center_squares:
            pos = Game.xy2i(square)
            piece = game.board.get_piece(pos)
            if (turn == 'w' and piece.isupper()) or (turn == 'b' and piece.islower()):
                positional_score += 100  # Bonus for controlling center squares

        # Open Files and Diagonals
        open_files = self.get_open_files(game)
        positional_score += len(open_files) * 50  # Bonus for each open file

        open_diagonals = self.get_open_diagonals(game)
        positional_score += len(open_diagonals) * 50  # Bonus for each open diagonal

        # Space Advantage: Count the number of squares controlled
        controlled_squares = self.count_controlled_squares(game, turn)
        positional_score += controlled_squares * (1 if not in_endgame else 2)  # Slightly higher in endgame

        # Control of Key Squares
        key_squares = self.get_key_squares(game, turn)
        positional_score += len(key_squares) * 75  # Significant bonus for control of key squares

        # Potential for Zugzwang
        zugzwang_score = self.evaluate_zugzwang_potential(game, turn)
        positional_score += zugzwang_score * 100  # High bonus for potential zugzwang

        return positional_score

    def get_open_files(self, game):
        """
        Get the number of open files (no pawns) controlled by the player.

        Args:
            game (Game): Current game state

        Returns:
            int: Number of open files
        """
        open_files = 0
        for file in range(8):
            has_pawn = False
            for rank in range(8):
                pos = Game.xy2i(f"{chr(file + ord('a'))}{rank + 1}")
                piece = game.board.get_piece(pos)
                if piece.lower() == 'p':
                    has_pawn = True
                    break
            if not has_pawn:
                open_files += 1
        return open_files

    def get_open_diagonals(self, game):
        """
        Get the number of open diagonals controlled by the player.

        Args:
            game (Game): Current game state

        Returns:
            int: Number of open diagonals
        """
        # Simplistic approach: count diagonals without pawns
        open_diagonals = 0
        # Main diagonals
        for d in range(-7, 8):
            has_pawn = False
            for x in range(8):
                y = x + d
                if 0 <= y < 8:
                    pos = Game.xy2i(f"{chr(x + ord('a'))}{y + 1}")
                    piece = game.board.get_piece(pos)
                    if piece.lower() == 'p':
                        has_pawn = True
                        break
            if not has_pawn:
                open_diagonals += 1
        return open_diagonals

    def count_controlled_squares(self, game, turn):
        """
        Count the number of squares controlled by the player.

        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'

        Returns:
            int: Number of controlled squares
        """
        controlled = set()
        for piece_type in ['P', 'N', 'B', 'R', 'Q', 'K'] if turn == 'w' else ['p', 'n', 'b', 'r', 'q', 'k']:
            for pos in game.board.positions(piece_type):
                piece = game.board.get_piece(pos)
                controlled.update(game.get_piece_attacks(piece, pos))
        return len(controlled)

    def get_key_squares(self, game, turn):
        """
        Identify and return key squares controlled by the player.
        Key squares are typically central and strategically important squares.

        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'

        Returns:
            set: Set of key squares under control
        """
        key_squares = set(['d4', 'd5', 'e4', 'e5', 'c3', 'c6', 'f3', 'f6',
                          'c4', 'f4', 'c5', 'f5'])  # Expanded for aggressive bottlenecking
        controlled_key_squares = set()
        for square in key_squares:
            pos = Game.xy2i(square)
            piece = game.board.get_piece(pos)
            if (turn == 'w' and piece.isupper()) or (turn == 'b' and piece.islower()):
                controlled_key_squares.add(pos)
        return controlled_key_squares

    def evaluate_zugzwang_potential(self, game, turn):
        """
        Evaluate the potential for creating zugzwang for the opponent.

        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'

        Returns:
            float: Zugzwang potential score
        """
        # Zugzwang is a situation where any move a player makes worsens their position
        # Detecting potential zugzwang requires deep analysis; here, we use heuristics

        opponent = 'b' if turn == 'w' else 'w'
        opponent_moves = list(game.get_moves(opponent))
        
        # If opponent has very few moves, potential for zugzwang
        if len(opponent_moves) == 0:
            return 300  # Checkmate already handled elsewhere
        elif len(opponent_moves) <= 2:
            return 200  # High potential
        elif len(opponent_moves) <= 4:
            return 100  # Moderate potential
        return 0

    def evaluate_bottleneck(self, game, turn):
        """
        Evaluate the potential for creating bottlenecks that confine the opponent's pieces.

        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'

        Returns:
            float: Bottleneck creation score
        """
        bottleneck_score = 0
        opponent = 'b' if turn == 'w' else 'w'

        # Define key bottleneck squares (expanded for aggressive strategy)
        bottleneck_squares = ['c4', 'f4', 'c5', 'f5', 'd4', 'd5', 'e4', 'e5',
                              'c3', 'c6', 'f3', 'f6']  # Include central squares

        for square in bottleneck_squares:
            pos = Game.xy2i(square)
            piece = game.board.get_piece(pos)
            if (turn == 'w' and piece.isupper()) or (turn == 'b' and piece.islower()):
                # Control of a bottleneck square
                bottleneck_score += 100

            # Additionally, check if opponent's pieces are confined around these squares
            opponent_attacks = self.get_controlled_squares(game, opponent)
            if pos in opponent_attacks:
                # Opponent's pieces are restricted from controlling bottleneck squares
                bottleneck_score += 50

        return bottleneck_score

    def evaluate_aggressive_defense(self, game, turn, in_endgame):
        """
        Evaluate and reward moves that strengthen the agent's defensive position aggressively.

        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
            in_endgame (bool): True if in endgame phase, else False

        Returns:
            float: Aggressive defense score
        """
        defense_score = 0

        # Identify vulnerable pieces
        vulnerable_pieces = self.get_vulnerable_pieces(game, turn)

        for piece_pos in vulnerable_pieces:
            # Reward moves that defend vulnerable pieces
            if self.is_defended(game, piece_pos, turn):
                defense_score += 100  # Significant reward for defending a vulnerable piece
            else:
                defense_score -= 100  # Penalty for leaving a piece undefended

        # Additionally, reward moves that block opponent's attacks
        blocking_moves = self.get_blocking_moves(game, turn)
        defense_score += len(blocking_moves) * 50  # Reward for each blocking move

        # Penalize moves that leave own pieces exposed
        exposed_pieces = self.get_exposed_pieces(game, turn)
        defense_score -= len(exposed_pieces) * 75  # Penalize for each exposed piece

        return defense_score

    def get_vulnerable_pieces(self, game, turn):
        """
        Identify own pieces that are vulnerable to being captured.

        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'

        Returns:
            list: Positions of vulnerable pieces
        """
        vulnerable_pieces = []
        for piece_type in ['P', 'N', 'B', 'R', 'Q', 'K'] if turn == 'w' else ['p', 'n', 'b', 'r', 'q', 'k']:
            for pos in game.board.positions(piece_type):
                # If the piece is under attack and not sufficiently defended
                if game.is_attacked(pos, 'b' if turn == 'w' else 'w'):
                    if not self.is_defended(game, pos, turn):
                        vulnerable_pieces.append(pos)
        return vulnerable_pieces

    def is_defended(self, game, pos, turn):
        """
        Check if a piece at a given position is defended by another piece.

        Args:
            game (Game): Current game state
            pos (int): Position of the piece
            turn (str): 'w' or 'b'

        Returns:
            bool: True if defended, else False
        """
        opponent = 'b' if turn == 'w' else 'w'
        attackers = game.get_attackers(pos, opponent)
        for attacker_pos in attackers:
            # Check if any of the agent's pieces attack the attacker
            if game.is_attacked(attacker_pos, turn):
                return True
        return False

    def get_blocking_moves(self, game, turn):
        """
        Identify moves that block opponent's attacks.

        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'

        Returns:
            list: Blocking moves
        """
        blocking_moves = []
        opponent = 'b' if turn == 'w' else 'w'
        for pos in game.board.positions('K' if turn == 'w' else 'k'):
            attackers = game.get_attackers(pos, opponent)
            for attacker_pos in attackers:
                # Determine if moving a piece can block the attack
                direction = self.get_direction(attacker_pos, pos)
                if direction is None:
                    continue
                blocker_pos = attacker_pos + direction
                if 0 <= blocker_pos < 64:
                    blocker_piece = game.board.get_piece(blocker_pos)
                    if blocker_piece == ' ' or blocker_piece.isupper() != (turn == 'w'):
                        blocking_moves.append(blocker_pos)
        return blocking_moves

    def get_exposed_pieces(self, game, turn):
        """
        Identify own pieces that are exposed to attacks.

        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'

        Returns:
            list: Positions of exposed pieces
        """
        exposed_pieces = []
        for piece_type in ['P', 'N', 'B', 'R', 'Q', 'K'] if turn == 'w' else ['p', 'n', 'b', 'r', 'q', 'k']:
            for pos in game.board.positions(piece_type):
                if game.is_attacked(pos, 'b' if turn == 'w' else 'w'):
                    exposed_pieces.append(pos)
        return exposed_pieces

    def evaluate_endgame(self, game, turn, in_endgame):
        """
        Evaluate endgame-specific factors such as king and pawn endgames or rook endgames.
        Enhanced eagerness in endgame to win immediately.
        
        Args:
            game (Game): Current game state after the move
            turn (str): 'w' or 'b'
            in_endgame (bool): True if in endgame phase, else False
        
        Returns:
            float: Endgame score
        """
        endgame_score = 0
        if in_endgame:
            # Prioritize promotion of passed pawns
            endgame_score += self.evaluate_pawn_promotion(game, turn) * 2  # Higher weight in endgame
            # Encourage king activity
            endgame_score += self.evaluate_king_activity(game, turn) * 1.5
            # Rook and king coordination
            endgame_score += self.evaluate_rook_king_coordination(game, turn) * 1.5
            # Control of key squares in endgame
            endgame_score += self.evaluate_control_of_key_squares(game, turn) * 2
            # Potential for zugzwang
            endgame_score += self.evaluate_zugzwang_potential(game, turn) * 1.5
        else:
            # Continue aggressive bottlenecking in the middlegame
            # No additional endgame-specific factors
            pass
        return endgame_score

    def evaluate_pawn_promotion(self, game, turn):
        """
        Evaluate the potential for pawn promotion.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            float: Pawn promotion score
        """
        promotion_score = 0
        pawns = [pos for pos in game.board.positions('P' if turn == 'w' else 'p')]
        
        for pawn in pawns:
            x, y = Game.i2xy(pawn)
            if turn == 'w' and y == 6:  # Pawn on 7th rank
                promotion_score += 150
            elif turn == 'b' and y == 1:  # Pawn on 2nd rank
                promotion_score += 150
        return promotion_score

    def evaluate_king_activity(self, game, turn):
        """
        Evaluate the activity of the king in the endgame.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            float: King activity score
        """
        king_pos = game.find_king(turn)
        if not king_pos:
            return 0
        x, y = Game.i2xy(king_pos)
        # Centralize the king
        distance_to_center = abs(x - 3.5) + abs(y - 3.5)
        activity_score = (7 - distance_to_center) * 20  # Higher when closer to center
        return activity_score

    def evaluate_rook_king_coordination(self, game, turn):
        """
        Evaluate the coordination between rooks and the king in the endgame.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            float: Rook and king coordination score
        """
        coordination_score = 0
        rooks = [pos for pos in game.board.positions('R' if turn == 'w' else 'r')]
        king_pos = game.find_king(turn)
        if not king_pos:
            return coordination_score
        for rook in rooks:
            # Rook should support the king's advancement
            distance = self.distance(rook, king_pos)
            if distance <= 4:
                coordination_score += 50  # Reward rooks near the king
        return coordination_score

    def evaluate_control_of_key_squares(self, game, turn):
        """
        Evaluate the control of key squares in the endgame.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            float: Control of key squares score
        """
        key_squares = set(['d4', 'd5', 'e4', 'e5', 'c3', 'c6', 'f3', 'f6',
                          'c4', 'f4', 'c5', 'f5'])  # Expanded for aggressive bottlenecking
        control_score = 0
        for square in key_squares:
            pos = Game.xy2i(square)
            piece = game.board.get_piece(pos)
            if (turn == 'w' and piece.isupper()) or (turn == 'b' and piece.islower()):
                control_score += 75  # Significant bonus for controlling key squares
        return control_score

    def distance(self, pos1, pos2):
        """
        Calculate Manhattan distance between two positions.
        
        Args:
            pos1 (int): First position
            pos2 (int): Second position
        
        Returns:
            int: Manhattan distance
        """
        x1, y1 = Game.i2xy(pos1)
        x2, y2 = Game.i2xy(pos2)
        return abs(x1 - x2) + abs(y1 - y2)

    def calculate_material(self, game, turn):
        """
        Calculate the material on the board.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            dict: Material counts
        """
        material = {
            'minor_pieces': 0,
            'rooks': 0,
            'queens': 0
        }
        for piece in game.board.pieces:
            if piece.isupper() if turn == 'w' else piece.islower():
                if piece.lower() in ['n', 'b']:
                    material['minor_pieces'] += 1
                elif piece.lower() == 'r':
                    material['rooks'] += 1
                elif piece.lower() == 'q':
                    material['queens'] += 1
        return material

    def evaluate_king_and_pawns_endgame(self, game, turn):
        """
        Evaluate king and pawn endgame factors.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            float: King and pawn endgame score
        """
        endgame_score = 0
        # Centralize the king
        king_pos = game.find_king(turn)
        if king_pos:
            x, y = Game.i2xy(king_pos)
            distance_to_center = abs(x - 3.5) + abs(y - 3.5)
            endgame_score += (7 - distance_to_center) * 10  # Closer to center is better
        # Advance passed pawns
        pawns = [pos for pos in game.board.positions('P' if turn == 'w' else 'p')]
        for pawn in pawns:
            x, y = Game.i2xy(pawn)
            advancement = y if turn == 'w' else 7 - y
            endgame_score += advancement * 10  # Encourage advancing pawns
        return endgame_score

    def evaluate_rook_endgame(self, game, turn):
        """
        Evaluate rook endgame factors.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            float: Rook endgame score
        """
        endgame_score = 0
        rooks = [pos for pos in game.board.positions('R' if turn == 'w' else 'r')]
        for rook in rooks:
            x, y = Game.i2xy(rook)
            # Rook on open file
            file = chr(x + ord('a'))
            is_open = all(game.board.get_piece(Game.xy2i(f"{file}{rank}")) == ' ' for rank in range(1, 9))
            if is_open:
                endgame_score += 50  # Bonus for rook on open file
            # Rook active (centralized)
            distance_to_center = abs(x - 3.5) + abs(y - 3.5)
            endgame_score += (7 - distance_to_center) * 5  # Closer to center is better
        # Assess pawn structure in rook endgame
        endgame_score += self.evaluate_pawn_structure(game, turn, in_endgame=True)
        return endgame_score

    def evaluate_game_state(self, game, maximizing_player):
        """
        Evaluate the game state from the perspective of the agent.
        
        Args:
            game (Game): Current game state
            maximizing_player (bool): True if evaluating for maximizing player, else False
        
        Returns:
            float: Evaluation score
        """
        in_endgame = self.is_endgame(game, game.turn)
        score = 0

        # Material evaluation
        material = self.calculate_material(game, game.turn)
        for piece, value in self.PIECE_VALUES.items():
            score += game.board.pieces.count(piece) * value

        # King Safety
        king_safety = self.evaluate_king_safety(game, game.turn)
        score += king_safety

        # Pawn Structure
        pawn_structure = self.evaluate_pawn_structure(game, game.turn, in_endgame)
        score += pawn_structure * (0.8 if not in_endgame else 1.2)

        # Positional Factors
        positional = self.evaluate_positional_factors(game, game.turn, in_endgame)
        score += positional * (0.8 if not in_endgame else 1.5)

        # Endgame Factors
        endgame = self.evaluate_endgame(game, game.turn, in_endgame)
        score += endgame

        # Bottleneck Creation
        bottleneck = self.evaluate_bottleneck(game, game.turn)
        score += bottleneck * (2.0 if not in_endgame else 3.0)

        # Aggressive Defense
        aggressive_defense = self.evaluate_aggressive_defense(game, game.turn, in_endgame)
        score += aggressive_defense * (1.5 if in_endgame else 1.0)

        # Long-term Strategic Planning
        strategic_planning = self.evaluate_strategic_planning(game, game.turn)
        score += strategic_planning * 2.0  # Significant weight for long-term planning

        return score

    def select_strategic_move_old(self, board_fen):
        """
        Select the most strategic move for the current board state.
        
        Args:
            board_fen (str): Current board state in FEN notation
        
        Returns:
            str: Selected move
        """
        # Check opening book first
        if board_fen in self.opening_moves:
            return random.choice(self.opening_moves[board_fen])

        # Create game instance
        game = Game(board_fen)
        
        # Get all legal moves
        moves = list(game.get_moves())
        
        # Ensure moves exist
        if not moves:
            return 'e2e4'  # Fallback move
        
        # Evaluate moves
        evaluated_moves = [
            (self.evaluate_move(game, move), move) 
            for move in moves[:10]  # Increased limit for deeper search
        ]
        
        # Sort moves by score, descending
        evaluated_moves.sort(reverse=True)
        
        # Return top move
        return evaluated_moves[0][1]

    def select_strategic_move(self, board_fen):
        """
        Advanced strategic move selection with multi-tier evaluation.
        
        Args:
            board_fen (str): Current board state in FEN notation
        
        Returns:
            str: Selected strategic move
        """
        # 1. Opening Book Priority
        if board_fen in self.opening_moves:
            return random.choice(self.opening_moves[board_fen])

        # 2. Create game instance
        game = Game(board_fen)
        moves = list(game.get_moves())
        
        if not moves:
            return 'e2e4'  # Fallback move
        
        # 3. Advanced Probabilistic Move Selection
        try:
            # Attempt advanced move prediction
            predicted_moves = self.predict_opponent_moves(game)
            
            if predicted_moves:
                # Evaluate moves considering opponent prediction
                move_evaluations = [
                    (
                        self.evaluate_move(game, move['initial_move']), 
                        move['initial_move'],
                        move['probability'],
                        move['strategic_score']
                    ) 
                    for move in predicted_moves[:5]  # Top 5 predicted moves
                ]
                
                # Multi-factor sorting
                move_evaluations.sort(
                    key=lambda x: (x[0], x[2], x[3]), 
                    reverse=True
                )
                
                return move_evaluations[0][1]
        
        except Exception as e:
            # Fallback to original evaluation if advanced prediction fails
            pass
        
        # 4. Traditional Move Evaluation
        evaluated_moves = [
            (self.evaluate_move(game, move), move) 
            for move in moves[:10]
        ]
        
        # Sort and return top move
        evaluated_moves.sort(reverse=True)
        return evaluated_moves[0][1]


    def evaluate_piece_coordination(self, game, turn):
        """
        Evaluate how well the pieces support each other.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            int: Piece coordination score
        """
        score = 0
        if turn == 'w':
            pieces = [pos for pos in game.board.positions('N') + game.board.positions('B') + 
                      game.board.positions('R') + game.board.positions('Q') + game.board.positions('K')]
        else:
            pieces = [pos for pos in game.board.positions('n') + game.board.positions('b') + 
                      game.board.positions('r') + game.board.positions('q') + game.board.positions('k')]
        
        for piece in pieces:
            attacks = game.get_piece_attacks(game.board.get_piece(piece), piece)
            for target in attacks:
                target_piece = game.board.get_piece(target)
                if target_piece.lower() in ['n', 'b', 'r', 'q', 'k']:
                    score += 10  # Reward for attacking and supporting other pieces
        return score

# => <= #

    def predict_opponent_moves(self, game, look_ahead_depth=4):
        """
        Predict and evaluate potential opponent moves with probabilistic weighting.
        
        Args:
            game (Game): Current game state
            look_ahead_depth (int): Number of future moves to analyze
        
        Returns:
            list: Ranked list of probable opponent move sequences
        """
        # Check memoization cache first
        cache_key = (game.get_fen(), look_ahead_depth)
        if cache_key in self.move_cache:
            return self.move_cache[cache_key]

        probable_move_sequences = []
        
        # Generate all legal opponent moves
        opponent_moves = game.get_moves()
        
        for move in opponent_moves:
            # Create a hypothetical game state after opponent's move
            hypothetical_game = Game(game.get_fen())
            hypothetical_game.apply_move(move)
            
            # Evaluate move sequences recursively
            move_sequence = self.explore_move_sequences(
                hypothetical_game, 
                depth=look_ahead_depth-1, 
                is_maximizing=False
            )
            
            # Calculate move probability and strategic value
            move_probability = self.calculate_move_probability(move, hypothetical_game)
            move_sequence_score = self.evaluate_move_sequence(move_sequence)
            
            probable_move_sequences.append({
                'initial_move': move,
                'sequence': move_sequence,
                'probability': move_probability,
                'strategic_score': move_sequence_score
            })
        
        # Sort by strategic score and probability
        probable_move_sequences.sort(
            key=lambda x: (x['strategic_score'], x['probability']), 
            reverse=True
        )
        
        # Cache results
        self.move_cache[cache_key] = probable_move_sequences
        
        return probable_move_sequences

    def calculate_move_probability(self, move, game):
        """
        Calculate the probability of a move based on strategic factors.
        
        Args:
            move (str): Chess move notation
            game (Game): Game state
        
        Returns:
            float: Probability score for the move
        """
        # Implement probabilistic move assessment
        probability_factors = [
            self.evaluate_tactical_patterns(game, game.turn),
            self.evaluate_threats(game, game.turn),
            self.evaluate_positional_factors(game, game.turn, False)
        ]
        
        return sum(probability_factors) / len(probability_factors)

    def evaluate_move_sequence(self, move_sequence):
        """
        Evaluate the strategic value of a move sequence.
        
        Args:
            move_sequence (list): Sequence of moves and their evaluations
        
        Returns:
            float: Strategic score for the move sequence
        """
        if not move_sequence:
            return 0
        
        # Calculate cumulative strategic score
        strategic_score = sum(
            move['evaluation'] * (0.9 ** idx) 
            for idx, move in enumerate(move_sequence)
        )
        
        return strategic_score

    def select_optimal_move(self, game):
        """
        Select the most strategic move considering opponent's probable responses.
        
        Args:
            game (Game): Current game state
        
        Returns:
            str: Best move to execute
        """
        # Predict and analyze opponent's probable moves
        opponent_move_predictions = self.predict_opponent_moves(game)
        
        # Evaluate current possible moves
        current_moves = game.get_moves()
        move_evaluations = [
            {
                'move': move, 
                'score': self.evaluate_move(game, move),
                'opponent_response_risk': self.assess_opponent_response_risk(game, move)
            } 
            for move in current_moves
        ]
        
        # Sort moves by comprehensive strategic score
        optimal_moves = sorted(
            move_evaluations, 
            key=lambda x: (x['score'], -x['opponent_response_risk']), 
            reverse=True
        )
        
        return optimal_moves[0]['move'] if optimal_moves else None

    def assess_opponent_response_risk(self, game, move):
        """
        Assess the potential risk of a move based on opponent's likely responses.
        
        Args:
            game (Game): Current game state
            move (str): Proposed move
        
        Returns:
            float: Risk score
        """
        hypothetical_game = Game(game.get_fen())
        hypothetical_game.apply_move(move)
        
        opponent_moves = hypothetical_game.get_moves()
        response_risks = [
            self.evaluate_move(hypothetical_game, response) 
            for response in opponent_moves
        ]
        
        return max(response_risks) if response_risks else 0

    def explore_move_sequences(self, game, depth, is_maximizing):
        """
        Recursively explore potential move sequences with memoization.
        
        Args:
            game (Game): Current game state
            depth (int): Remaining look-ahead depth
            is_maximizing (bool): Whether current player is maximizing
        
        Returns:
            list: Sequence of moves and their evaluations
        """
        # Create a unique cache key
        cache_key = (
            game.get_fen(),  # Current board state
            depth,           # Remaining depth
            is_maximizing    # Current player's turn
        )

        # Check if result is already cached
        if cache_key in self.move_sequence_cache:
            return self.move_sequence_cache[cache_key]
        
        # Termination conditions
        if depth == 0 or game.status in [Game.CHECKMATE, Game.DRAW]:
            return []
        
        move_sequences = []
        moves = game.get_moves()
        
        for move in moves:
            # Create a copy of the game to avoid state mutation
            hypothetical_game = Game(game.get_fen())
            hypothetical_game.apply_move(move)
            
            # Recursive exploration with alternating players
            sub_sequences = self.explore_move_sequences(
                hypothetical_game, 
                depth - 1, 
                not is_maximizing
            )
            
            move_evaluation = self.evaluate_move(game, move)
            
            move_sequence_entry = {
                'move': move,
                'evaluation': move_evaluation,
                'sub_sequences': sub_sequences
            }
            
            move_sequences.append(move_sequence_entry)
        
        # Cache and return the results
        self.move_sequence_cache[cache_key] = move_sequences
        return move_sequences

    def clear_move_sequence_cache(self):
        """
        Clear the memoization cache to prevent memory overflow.
        Useful between different game states or after significant changes.
        """
        self.move_sequence_cache.clear()
# => <= #
def chess_bot(obs):
    """
    Chess bot interface that provides strategic move selection.
    
    Args:
        obs: An object with a 'board' attribute representing the board state
    
    Returns:
        str: Selected move in UCI notation
    """
    agent = StrategicChessAgent(max_search_depth=1500)
    return agent.select_strategic_move(obs.board)

