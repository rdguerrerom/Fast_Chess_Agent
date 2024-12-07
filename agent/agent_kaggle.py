%%writefile submission.py
from Chessnut import Game
import random
import math
import copy

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
    BASE_PIECE_VALUES = {
        'P': 100, 'N': 1320, 'B': 1330, 'R': 1500, 'Q': 1900, 'K': 20000,
        'p': -110, 'n': -1520, 'b': -1530, 'r': -1700, 'q': -2100, 'k': -23000
    }

    def __init__(self, max_search_depth=16):
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
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1': ['e2e4', 'd2d4', 'c2c4', 'g1f3'],
            # Open Games
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1': ['e7e5', 'c7c5'],
            'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1': ['e7e6', 'd7d5'],
            
            # Semi-Open Games
            'rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1': ['e7e6', 'd7d5'],
            
            # Closed Games
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1': ['e7e6', 'd7d5'],
            
            # Sicilian Defense
            'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2': ['d2d4'],

            # Ruy-Lopez
            '1e4 e5 2Nf3 Nc6 3Bb5': ['a7a6', 'd7d6', 'f8e7'],

            # Italiam Game
            '1e4 e5 2Nf3 Nc6 3Bc4': ['g8f6', 'f8c5'],

            # Scotch Game
            '1e4 e5 2Nf3 Nc6 3d4': ['e5d4', 'c6d4'],

            # Sicilian Defense
            '1e4 c5': ['g1f3', 'c2c3'],

            # French defense
            '1e4 e6': ['d2d4', 'g1f3'],

            # Caro-Kann Defense
            '1e4 c6': ['d2d4', 'g1f3'],

            # Queen’s Gambit
            '1d4 d5 2c4': ['e7e6', 'c7c6'],

            # King’s Indian Defense
            '1d4 Nf6 2c4 g6': ['g2g3', 'b1c3'],

            # Grünfeld defense
            '1d4 Nf6 2c4 g6 3Nc3 d5': ['c4d5', 'e2e4'],

            #English opening
            '1c4': ['e7e5', 'g8f6'],

            # Reti Openingi
            '1Nf3': ['d7d5', 'g8f6'],

            # King’s Fianchetto
            '1g3': ['d7d5', 'g8f6'],

            # Responses to irregular openings
            '1e4': ['c5', 'e6', 'c6', 'g6'],
            '1d4': ['d5', 'Nf6', 'e6'],
            '1c4': ['e5', 'c5'],
            '1Nf3': ['d5', 'Nf6']
        }

    def get_piece_value(self, piece, perspective='w'):
        val = BASE_PIECE_VALUES.get(piece, 0)
        if perspective == 'b':
            # Flip the sign for all values to mirror perspective
            val = -val
        return val

    def simplify_fen(self, fen):
        """
        Simplify a FEN string by removing unnecessary details.
        Keeps:
            - Piece placement
            - Active color
            - Castling rights
        Removes:
            - En passant square
            - Halfmove clock
            - Fullmove number
        
        Args:
            fen (str): Full FEN string.
        
        Returns:
            str: Simplified FEN string.
        """
        parts = fen.split()
        if len(parts) < 4:
            raise ValueError("Invalid FEN format")

        # Retain only the first three components (piece placement, active color, castling rights)
        simplified = " ".join(parts[:3])
        return simplified

    def get_game_turn(self, game):
        # FEN format: 
        # [0] Piece placement
        # [1] Active color ('w' or 'b')
        # [2] Castling availability
        # [3] En passant target square
        # [4] Halfmove clock
        # [5] Fullmove number
        fen_parts = game.get_fen().split()
        return fen_parts[1]  # 'w' or 'b'


    def get_opening_move(self, board_fen):
        """
        Retrieve a move from the opening book or generate a dynamic response.
        """
        # Simplify FEN string for matching
        key = self.simplify_fen(board_fen)

        if key in self.opening_moves:
            return random.choice(self.opening_moves[key])

        # If no opening move found, dynamically generate a strategic response
        game = Game(board_fen)
        legal_moves = game.get_moves()
        return self.dynamic_opening_strategy(game, legal_moves)


    def evaluate_move_old(self, game, move):
        """
        Evaluate a potential move with an aggressive strategy, emphasizing tactical advantages,
        offensive capabilities, and strategic long-term winning potential.
        
        Args:
            game (Game): Current game state
            move (str): Move to evaluate
        
        Returns:
            float: Move evaluation score with increased aggression and tactical focus

        TODO: Port this to the new function
        """
        score = 0

        try:
            test_game = Game(game.get_fen())
            test_game.apply_move(move)

            # Detect game stage
            in_endgame = self.is_endgame(test_game, game.turn)

            # Collaborative checkmate evaluation
            score += self.evaluate_checkmate_coordination(test_game, game.turn)*5.0


            # Detect queen moves and evaluate potential
            moving_piece = game.board.get_piece(Game.xy2i(move[:2]))
            if moving_piece.lower() == 'q':  # Queen move
                queen_pos = Game.xy2i(move[2:4])
                score += self.evaluate_queen_potential(test_game, queen_pos, game.turn)

            # Detect second-rank piece moves
            moving_piece = game.board.get_piece(Game.xy2i(move[:2]))
            new_pos = Game.xy2i(move[2:4])

            if moving_piece.lower() == 'r':  # Rook
                score += self.evaluate_rook_potential(test_game, new_pos, game.turn)
            elif moving_piece.lower() == 'n':  # Knight
                score += self.evaluate_knight_potential(test_game, new_pos, game.turn)
            elif moving_piece.lower() == 'b':  # Bishop
                score += self.evaluate_bishop_potential(test_game, new_pos, game.turn)


            # 1. Key square coordination for high-rank pieces
            key_square_score = self.evaluate_piece_on_key_squares(test_game, game.turn)
            score += key_square_score * (2.0 if in_endgame else 1.5)
            # 2. Immediate Pawn Promotion (Highest Priority)
            if move[1] in ['7', '2'] and move[3] in ['8', '1']:
                score += 2000  # High reward for immediate promotion

            score += self.evaluate_piece_coordination(test_game, game.turn) * 3.0
            score += self.evaluate_control_of_key_squares(test_game, game.turn) * 5.5
            score += self.evaluate_checkmate_coordination(test_game, game.turn) * 4.0

            if in_endgame:
                score += self.evaluate_rook_potential(test_game, new_pos, game.turn) * 2.0
                score += self.evaluate_knight_potential(test_game, new_pos, game.turn) * 1.5
                score += self.evaluate_bishop_potential(test_game, new_pos, game.turn) * 1.5
            else:
                score += self.evaluate_rook_potential(test_game, new_pos, game.turn) * 1.5
                score += self.evaluate_knight_potential(test_game, new_pos, game.turn) * 2.0
                score += self.evaluate_bishop_potential(test_game, new_pos, game.turn) * 2.0


            # 3. Evaluate pawn advancement and coordination
            moving_piece = game.board.get_piece(Game.xy2i(move[:2]))
            if moving_piece.lower() == 'p':  # If the move involves a pawn
                pawn_advancement_score = self.evaluate_coordinated_pawn_advancement(test_game, game.turn)
                score += pawn_advancement_score * (2.0 if in_endgame else 1.5)

            # 4. Aggressive Capture Evaluation
            captured_piece = game.board.get_piece(Game.xy2i(move[2:4]))
            if captured_piece != ' ':
                capture_bonus = 1500 + self.PIECE_VALUES[captured_piece]
                score += capture_bonus

            # 5. Enhanced Threats Evaluation
            threats_score = self.evaluate_threats(test_game, game.turn)
            score += threats_score * (3.0 if not in_endgame else 4.0)

            # 6. Dynamic Piece Mobility
            mobility_score = self.evaluate_mobility(test_game, game.turn)
            score += mobility_score * (2.0 if not in_endgame else 2.5)

            # 7. King Safety
            king_safety_score = self.evaluate_king_safety(test_game, game.turn)
            score += king_safety_score * 2.0

            # 8. King Safety (Balanced Consideration)
            king_safety_score = self.evaluate_king_safety(test_game, game.turn)
            score += king_safety_score * 1.2

            # 9. Pawn Structure with Strategic Nuance
            pawn_structure_score = self.evaluate_pawn_structure(test_game, game.turn, in_endgame)
            score += pawn_structure_score * (1.0 if not in_endgame else 1.5)

            # 10. Positional Strategy with Offensive Orientation
            positional_score = self.evaluate_positional_factors(test_game, game.turn, in_endgame)
            score += positional_score * (1.2 if not in_endgame else 2.0)

            # 11. Enhanced Endgame Considerations
            endgame_score = self.evaluate_endgame(test_game, game.turn, in_endgame)
            score += endgame_score * 1.5

            # 12. Deep Search Minimax Evaluation
            if in_endgame:
                minimax_score = self.minimax(test_game, self.choose_search_depth(), -math.inf, math.inf, False)
            else:
                minimax_score = self.minimax(test_game, self.choose_search_depth(), -math.inf, math.inf, False)
            score += minimax_score * 3.0

            # 13. Ultra-Aggressive Bottleneck Creation
            bottleneck_score = self.evaluate_bottleneck(test_game, game.turn)* (20.0 if not in_endgame else 40.0)
            score += bottleneck_score * (12.0 if not in_endgame else 24.0)

            # 14. Strategic Offensive Defense
            aggressive_defense_score = self.evaluate_aggressive_defense(test_game, game.turn, in_endgame)
            score += aggressive_defense_score * (3.0 if in_endgame else 2.5)

            # 15. Long-Term Strategic Dominance Planning
            strategic_planning_score = self.evaluate_strategic_planning(test_game, game.turn)
            score += strategic_planning_score * 5.0

            # 16. Piece Value Consideration with Offensive Weighting
            piece = game.board.get_piece(Game.xy2i(move[0:2]))
            score += self.PIECE_VALUES.get(piece, 0) * 1.5

            # 17. Strategic Unpredictability Factor
            score += random.random() * 50

            # Integrate proactivity evaluation
            score += self.evaluate_proactivity(game, move, game.turn)*1.2

            # Update the endgame scoring section
            if in_endgame:
                # Dramatically increase weights for checkmate-oriented metrics
                score += self.calculate_checkmate_acceleration_bonus(test_game, game.turn) * 10.0
                score += endgame_score * 3.0  # Increased endgame score weight
                score += threats_score * 4.0  # More emphasis on threatening moves
                score += tactical_score * 5.0  # Higher tactical pattern importance

                # Endgame-specific evaluations
                score += self.evaluate_endgame_checkmate_potential(test_game, game.turn)
                score += self.evaluate_king_pawn_coordination(test_game, game.turn)


        except Exception as e:
            # Robust Fallback Scoring
            score = random.random() * 10

        return score

    def evaluate_move(self, game, move):
        """
        Evaluate a potential move with an aggressive strategy, emphasizing tactical advantages,
        offensive capabilities, and strategic long-term winning potential.
        """
        score = 0
        turn = self.get_game_turn(game)
        try:
            test_game = Game(game.get_fen())
            test_game.apply_move(move)

            in_endgame = self.is_endgame(test_game, turn)

            # Boost checkmate-related factors even more
            checkmate_coord = self.evaluate_checkmate_coordination(test_game, turn)
            score += checkmate_coord * 10.0

            moving_piece = game.board.get_piece(Game.xy2i(move[:2]))
            new_pos = Game.xy2i(move[2:4])

            # Increase value for queen, rook, knight, bishop potential again
            if moving_piece.lower() == 'q':
                score += self.evaluate_queen_potential(test_game, new_pos, turn) * 2.0
            elif moving_piece.lower() == 'r':
                score += self.evaluate_rook_potential(test_game, new_pos, turn) * 2.0
            elif moving_piece.lower() == 'n':
                score += self.evaluate_knight_potential(test_game, new_pos, turn) * 2.0
            elif moving_piece.lower() == 'b':
                score += self.evaluate_bishop_potential(test_game, new_pos, turn) * 2.0

            # 3. Evaluate pawn advancement and coordination
            moving_piece = game.board.get_piece(Game.xy2i(move[:2]))
            if moving_piece.lower() == 'p':  # If the move involves a pawn
                pawn_advancement_score = self.evaluate_coordinated_pawn_advancement(test_game, game.turn)
                score += pawn_advancement_score * (2.0 if in_endgame else 1.5)

            # Key squares, coordination, etc.
            key_square_score = self.evaluate_piece_on_key_squares(test_game, turn)
            score += key_square_score * (3.0 if in_endgame else 2.5)

            # Immediate promotion extremely rewarded
            if move[1] in ['7','2'] and move[3] in ['8','1']:
                score += 3500

            # Increase piece coordination importance
            piece_coordination_score = self.evaluate_piece_coordination(test_game, turn)
            score += piece_coordination_score * 5.0

            control_key_squares_score = self.evaluate_control_of_key_squares(test_game, turn)
            score += control_key_squares_score * 8.0

            # Incorporate multi-piece coordination, even if not used before
            multi_piece_coord = self.evaluate_multi_piece_coordination(test_game, turn)
            score += multi_piece_coord * 4.0

            # Threats, mobility, king safety
            threats_score = self.evaluate_threats(test_game, turn)
            score += threats_score * (5.0 if in_endgame else 4.0)

            mobility_score = self.evaluate_mobility(test_game, turn)
            score += mobility_score * (3.0 if in_endgame else 2.5)

            king_safety_score = self.evaluate_king_safety(test_game, turn)
            score += king_safety_score * 3.0

            pawn_structure_score = self.evaluate_pawn_structure(test_game, turn, in_endgame)
            score += pawn_structure_score * (1.5 if in_endgame else 1.2)

            positional_score = self.evaluate_positional_factors(test_game, turn, in_endgame)
            score += positional_score * (2.5 if in_endgame else 1.8)

            endgame_score = self.evaluate_endgame(test_game, turn, in_endgame)
            score += endgame_score * (2.5 if in_endgame else 1.0)

            # Add rook king coordination, king activity, king-pawn coordination at all times, especially in endgame
            king_activity_score = self.evaluate_king_activity(test_game, turn)
            score += king_activity_score * (5.0 if in_endgame else 2.0)

            rook_king_coord_score = self.evaluate_rook_king_coordination(test_game, turn)
            score += rook_king_coord_score * (4.0 if in_endgame else 1.5)

            king_pawn_coord_score = self.evaluate_king_pawn_coordination(test_game, turn)
            score += king_pawn_coord_score * (4.0 if in_endgame else 1.5)

            # If in endgame, also integrate previously unused endgame heuristics:
            if in_endgame:
                king_pawns_endgame_score = self.evaluate_king_and_pawns_endgame(test_game, turn)
                score += king_pawns_endgame_score * 3.0

                rook_endgame_score = self.evaluate_rook_endgame(test_game, turn)
                score += rook_endgame_score * 2.5

            # Add previously unused checkmate patterns and endgame checkmate potential strongly
            checkmate_patterns_score = self.evaluate_checkmate_patterns(test_game, turn)
            score += checkmate_patterns_score * (10.0 if in_endgame else 5.0)

            endgame_checkmate_potential_score = self.evaluate_endgame_checkmate_potential(test_game, turn)
            score += endgame_checkmate_potential_score * (10.0 if in_endgame else 5.5)

            # Minimax remains
            minimax_score = self.minimax(test_game, self.choose_search_depth(test_game), -math.inf, math.inf, False)
            score += minimax_score * 3.0

            # Aggressive Defense and Bottleneck still matter
            aggressive_defense_score = self.evaluate_aggressive_defense(test_game, turn, in_endgame)
            score += aggressive_defense_score * (6.0 if in_endgame else 5.0)

            bottleneck_score = self.evaluate_bottleneck(test_game, turn)* (40.0 if in_endgame else 25.0)
            score += bottleneck_score * (15.0 if in_endgame else 10.0)

            strategic_planning_score = self.evaluate_strategic_planning(test_game, turn)
            score += strategic_planning_score * (6.0 if in_endgame else 3.0)

            # Piece value
            piece_value = self.get_piece_value(moving_piece, turn)
            score += piece_value * 6.0

            # Reduce randomness further; we want stable mate-seeking behavior
            score += random.random() * 5

            # Proactivity
            proactivity_score = self.evaluate_proactivity(game, move, turn)
            score += proactivity_score * 3.0

            # Endgame checkmate acceleration
            if in_endgame:
                score += self.calculate_checkmate_acceleration_bonus(test_game, turn) * 25.0
                score += endgame_score * 4.0
                score += threats_score * 7.0
                tactical_score = self.evaluate_tactical_patterns(test_game, turn)
                score += tactical_score * 10.0

        except Exception:
            score = random.random() * 10

        return score


    def calculate_checkmate_acceleration_bonus(self, test_game, turn):
        """
        Calculate an aggressive bonus for moves that push towards checkmate in endgame scenarios.
        
        Args:
            test_game (Game): Projected game state
            turn (str): Current player's turn
        
        Returns:
            float: Checkmate acceleration bonus
        """
        own_king_pos = test_game.board.get_king_position(turn)
        opp_king_pos = test_game.board.get_king_position('b' if turn == 'w' else 'w')
        
        king_distance = abs(own_king_pos[0] - opp_king_pos[0]) + abs(own_king_pos[1] - opp_king_pos[1])
        material_diff = self.calculate_material_difference(test_game, turn)

        checkmate_acceleration_score = (
            (8 - king_distance) * 500 +  # closer kings => higher bonus
            (material_diff * 250) +       # material superiority
            (1000 if self.is_check(test_game, turn) else 0)
        )

        return checkmate_acceleration_score

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
        Evaluate coordination between pieces of the current side (turn). This involves
        how well pieces support each other, attack the same targets, and create compound threats.

        Args:
            game (Game): Current game state.
            turn (str): 'w' or 'b'.

        Returns:
            float: Score for piece coordination.
        """
        score = 0
        # Identify friendly pieces depending on turn
        if turn == 'w':
            pieces = (game.board.positions('N') + game.board.positions('B') +
                      game.board.positions('R') + game.board.positions('Q') +
                      game.board.positions('K'))
        else:
            pieces = (game.board.positions('n') + game.board.positions('b') +
                      game.board.positions('r') + game.board.positions('q') +
                      game.board.positions('k'))

        for p_pos in pieces:
            piece_type = game.board.get_piece(p_pos)
            attacks = game.get_piece_attacks(piece_type, p_pos)
            for target in attacks:
                target_piece = game.board.get_piece(target)
                # Check if target is an opponent's piece
                if target_piece != ' ' and ((piece_type.isupper() and target_piece.islower()) or
                                            (piece_type.islower() and target_piece.isupper())):
                    # Base reward for threatening enemy pieces
                    score += 20

                    # If another friendly piece also attacks this target => coordinated attack
                    if self.is_attacked_by_ally(game, target, turn):
                        score += 90

                    # If the target square is also defended by a friendly piece => control synergy
                    if self.is_defended(game, target, turn):
                        score += 50

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
        attacking_piece = game.board.get_piece(attacker_pos)
        if attacking_piece.lower() not in ['n', 'b', 'r', 'q']:
            return False

        attacks = game.get_piece_attacks(attacking_piece, attacker_pos)
        high_value_targets = 0
        for pos in attacks:
            piece = game.board.get_piece(pos)
            if piece != ' ' and ((piece.islower() if turn == 'w' else piece.isupper())):
                # Check if piece is high-value by absolute value:
                if abs(self.get_piece_value(piece, turn)) >= 300:
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
        attacker_piece = game.board.get_piece(attacker_pos)
        target_piece = game.board.get_piece(target_pos)
        if attacker_piece.lower() not in ['b', 'r', 'q']:
            return False

        direction = self.get_direction(attacker_pos, target_pos)
        if direction is None:
            return False

        next_pos = target_pos + direction
        while 0 <= next_pos < 64:
            beyond_piece = game.board.get_piece(next_pos)
            if beyond_piece == ' ':
                next_pos += direction
                continue
            # Check if beyond_piece is higher value than target_piece
            if ((beyond_piece.isupper() and turn == 'w') or (beyond_piece.islower() and turn == 'b')):
                # Compare absolute values
                beyond_val = abs(self.get_piece_value(beyond_piece, turn))
                target_val = abs(self.get_piece_value(target_piece, turn))
                if beyond_val > target_val:
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
        attacking_piece = game.board.get_piece(attacker_pos)
        if attacking_piece.lower() not in ['n', 'b', 'r', 'q']:
            return False

        attacks = game.get_piece_attacks(attacking_piece, attacker_pos)
        high_value_targets = 0
        for pos in attacks:
            piece = game.board.get_piece(pos)
            if piece != ' ' and ((piece.islower() if turn == 'w' else piece.isupper())):
                if abs(self.get_piece_value(piece, turn)) >= 300:
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

        if in_endgame:
            # Encourage king to move closer to opponent's king
            own_king_pos = test_game.board.get_king_position(turn)
            opp_king_pos = test_game.board.get_king_position('b' if turn == 'w' else 'w')
            
            king_proximity_bonus = max(0, 14 - (abs(own_king_pos[0] - opp_king_pos[0]) + 
                                                 abs(own_king_pos[1] - opp_king_pos[1]))) * 200
            
            safety_score += king_proximity_bonus

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
            if self.is_passed_pawn(game, pawn):
                pawn_score += 100  # Significant bonus for passed pawn
                if in_endgame:
                    pawn_score += 50  # Additional bonus in endgame

            # Avoid creating weak squares (doubled, isolated, backward)
            if self.is_doubled_pawn(game, pawn, turn):
                pawn_score -= 50  # Penalty for doubled pawn
            if self.is_isolated_pawn(game, pawn):
                pawn_score -= 50  # Penalty for isolated pawn
            if self.is_backward_pawn(game, pawn):
                pawn_score -= 50  # Penalty for backward pawn

        # Pawn Majority on a Side
        pawn_majority_score = self.evaluate_pawn_majority(game, turn)
        pawn_score += pawn_majority_score

        return pawn_score

    def is_passed_pawn(self, game, pawn_pos):
        """
        Determine if a pawn is a passed pawn. A passed pawn is one that has
        no opposing pawns that can stop it from advancing to the promotion rank.

        Args:
            game (Game): Current game state.
            pawn_pos (int): Position of the pawn.

        Returns:
            bool: True if the pawn is passed, False otherwise.
        """
        piece = game.board.get_piece(pawn_pos)
        if piece == ' ':
            return False
        
        # White pawns move upward (increasing rank), black pawns move downward (decreasing rank).
        is_white = piece.isupper()
        x, y = Game.i2xy(pawn_pos)

        # Define the direction of forward movement for this pawn
        if is_white:
            forward_ranks = range(y+1, 8)  # Check ranks above
            enemy_pawn = 'p'
        else:
            forward_ranks = range(y-1, -1, -1)  # Check ranks below for black
            enemy_pawn = 'P'

        # Check all squares ahead (forward) in the same file and adjacent files
        for rank in forward_ranks:
            for dx in [x-1, x, x+1]:
                if 0 <= dx < 8:
                    pos = Game.xy2i(f"{chr(dx + ord('a'))}{rank + 1}")
                    if game.board.get_piece(pos) == enemy_pawn:
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

    def is_isolated_pawn(self, game, pawn_pos):
        """
        Check if a pawn is isolated. An isolated pawn has no friendly pawns in the
        adjacent files on its forward-moving side of the board.

        Args:
            game (Game): Current game state.
            pawn_pos (int): Position of the pawn.

        Returns:
            bool: True if the pawn is isolated, False otherwise.
        """
        piece = game.board.get_piece(pawn_pos)
        if piece == ' ':
            return False

        is_white = piece.isupper()
        x, y = Game.i2xy(pawn_pos)

        # Adjacent files to check
        adjacent_files = [x-1, x+1]

        # Determine forward direction for scanning
        # White pawns move up towards higher ranks, black pawns move down.
        rank_range = range(y, 8) if is_white else range(y, -1, -1)
        friendly_pawn = 'P' if is_white else 'p'

        has_support = False
        for af in adjacent_files:
            if 0 <= af < 8:
                for dy in rank_range:
                    pos = Game.xy2i(f"{chr(af + ord('a'))}{dy + 1}")
                    if game.board.get_piece(pos) == friendly_pawn:
                        has_support = True
                        break
            if has_support:
                break

        return not has_support

    def is_backward_pawn(self, game, pawn_pos):
        """
        Check if a pawn is backward. A backward pawn is one that cannot advance due
        to a blocked forward square and lacks pawns behind it that could support an advance.

        This simple heuristic checks if the immediate forward square is blocked.

        Args:
            game (Game): Current game state.
            pawn_pos (int): Position of the pawn.

        Returns:
            bool: True if the pawn is backward, False otherwise.
        """
        piece = game.board.get_piece(pawn_pos)
        if piece == ' ':
            return False

        is_white = piece.isupper()
        x, y = Game.i2xy(pawn_pos)
        # Determine the square in front of the pawn
        target_y = y + 1 if is_white else y - 1

        if not (0 <= target_y < 8):
            # Pawn is at the last rank or invalid forward position, can't be "backward" in the usual sense
            return False

        forward_pos = Game.xy2i(f"{chr(x + ord('a'))}{target_y + 1}")
        # If the forward square is occupied, pawn can't move forward. This can hint at being backward.
        # However, for a fully robust backward definition, you'd also consider the inability
        # to be supported by pawns. For simplicity, we follow the original logic:
        if game.board.get_piece(forward_pos) != ' ':
            return True
        return False


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
        positional_score += len(key_squares) * 100  # Significant bonus for control of key squares

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
            # Reward key square control by high-rank pieces
            score += self.evaluate_piece_on_key_squares(game, turn) * 2.0

            # Encourage king activity and pawn advancement
            score += self.evaluate_king_activity(game, turn) * 1.5
            score += self.evaluate_rook_king_coordination(game, turn) * 1.5

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
        Evaluate the potential of pawns promoting. Pawns close to the last rank receive a bonus.

        Args:
            game (Game): Current game state.
            turn (str): 'w' or 'b' - perspective for evaluation.

        Returns:
            float: Score reflecting the pawn promotion potential.
        """
        score = 0
        # From the perspective of 'turn', we consider the pawns of that color.
        is_white = (turn == 'w')
        pawns = game.board.positions('P' if is_white else 'p')

        for pawn in pawns:
            x, y = Game.i2xy(pawn)
            # For white, a pawn at y=6 (7th rank) can promote next move
            # For black, a pawn at y=1 (2nd rank) can promote next move
            if is_white and y == 6:
                score += 250
            elif (not is_white) and y == 1:
                score += 250

        return score


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
        turn = self.get_game_turn(game)
        turn = self.get_game_turn(game)
        in_endgame = self.is_endgame(game, turn)
        score = 0

        # Material
        for piece in game.board.pieces:
            score += self.get_piece_value(piece, turn)

        # King Safety
        king_safety = self.evaluate_king_safety(game, turn)
        score += king_safety * 2.5

        # Pawn Structure
        pawn_structure = self.evaluate_pawn_structure(game, turn, in_endgame)
        score += pawn_structure * (1.5 if in_endgame else 1.0)

        # Positional Factors
        positional = self.evaluate_positional_factors(game, turn, in_endgame)
        score += positional * (2.0 if in_endgame else 1.5)

        # Endgame Factors
        endgame = self.evaluate_endgame(game, turn, in_endgame)
        score += endgame * (3.0 if in_endgame else 1.0)

        # Bottleneck Creation
        bottleneck = self.evaluate_bottleneck(game, turn)
        score += bottleneck * (4.0 if in_endgame else 2.0)

        # Aggressive Defense
        aggressive_defense = self.evaluate_aggressive_defense(game, turn, in_endgame)
        score += aggressive_defense * (2.0 if in_endgame else 1.5)

        # Strategic Planning
        strategic_planning = self.evaluate_strategic_planning(game, turn)
        score += strategic_planning * (4.0 if in_endgame else 2.0)

        # Integrate king proactivity and coordination even in static eval
        king_activity_score = self.evaluate_king_activity(game, turn)
        score += king_activity_score * (3.0 if in_endgame else 1.0)

        rook_king_coord_score = self.evaluate_rook_king_coordination(game, turn)
        score += rook_king_coord_score * (3.0 if in_endgame else 1.0)

        king_pawn_coord_score = self.evaluate_king_pawn_coordination(game, turn)
        score += king_pawn_coord_score * (3.0 if in_endgame else 1.0)

        # Add multi-piece coordination to static eval
        multi_piece_coord = self.evaluate_multi_piece_coordination(game, turn)
        score += multi_piece_coord * (3.0 if in_endgame else 1.0)

        # If in endgame, integrate these endgame-specific checks:
        if in_endgame:
            king_pawns_endgame_score = self.evaluate_king_and_pawns_endgame(game, turn)
            score += king_pawns_endgame_score * 3.0

            rook_endgame_score = self.evaluate_rook_endgame(game, turn)
            score += rook_endgame_score * 2.0

        # Checkmate patterns & potential
        checkmate_coordination = self.evaluate_checkmate_coordination(game, turn)
        score += checkmate_coordination * (6.0 if in_endgame else 3.0)

        checkmate_patterns_score = self.evaluate_checkmate_patterns(game, turn)
        score += checkmate_patterns_score * (5.0 if in_endgame else 2.0)

        endgame_checkmate_potential_score = self.evaluate_endgame_checkmate_potential(game, turn)
        score += endgame_checkmate_potential_score * (6.0 if in_endgame else 1.5)

        # Encourage quicker mate with acceleration bonus in endgame static eval
        if in_endgame:
            score += self.calculate_checkmate_acceleration_bonus(game, turn) * 15.0

        return score

    def select_strategic_move(self, board_fen, look_ahead_depth=16):
        """
        Advanced strategic move selection with multi-tier evaluation.
        
        Args:
            board_fen (str): Current board state in FEN notation
        
        Returns:
            str: Selected strategic move
        """
        """
        Select a strategic move from the given board position.

        Steps:
        1. Check if position is in opening book. If so, pick a random opening move.
        2. Otherwise, create a Game instance and get all legal moves.
        3. If no legal moves, fallback to a symmetrical default based on turn.
        4. Attempt advanced probabilistic opponent move prediction:
           - Evaluate predicted moves and pick the best one.
           - If no predictions available or an error occurs, fallback to traditional move evaluation.
        5. Traditional evaluation:
           - Evaluate all legal moves with evaluate_move, pick the highest scoring one.

        This method now fully accounts for both White ('w') and Black ('b') perspectives.
        """
        # Create a Game instance to determine turn and moves
        test_game = Game(board_fen)
        turn = self.get_game_turn(test_game)  # 'w' or 'b'
        
        # 1. Opening Book Priority
        if board_fen in self.opening_moves:
            return random.choice(self.opening_moves[board_fen])

        moves = list(test_game.get_moves())
        depth = 12
        if not moves:
            # 2. No legal moves fallback:
            # White fallback: 'e2e4'
            # Black fallback: 'e7e5'
            if turn == 'w':
                return 'e2e4'
            else:
                return 'e7e5'

        # 3. Attempt Advanced Probabilistic Move Selection
        try:
            predicted_moves = self.predict_opponent_moves(test_game, look_ahead_depth=depth)
            if predicted_moves:
                # Evaluate top predicted moves. Perspective is handled inside evaluate_move.
                move_evaluations = [
                    (self.evaluate_move(test_game, move_info['initial_move']), 
                     move_info['initial_move'],
                     move_info['probability'],
                     move_info['strategic_score'])
                    for move_info in predicted_moves[:5]
                ]
                # Sort by score, probability, and strategic_score
                move_evaluations.sort(key=lambda x: (x[0], x[2], x[3]), reverse=True)
                return move_evaluations[0][1]
        except Exception:
            # If any error in prediction, fallback to traditional evaluation
            pass

        # 4. Traditional Move Evaluation
        evaluated_moves = [
            (self.evaluate_move(test_game, move), move) 
            for move in moves
        ]
        evaluated_moves.sort(reverse=True, key=lambda x: x[0])

        # Return the best move after thorough perspective-based evaluation
        return evaluated_moves[0][1]

    def evaluate_piece_coordination(self, game, turn):
        """
        Evaluate coordination between pieces, prioritizing collaborative attacks
        and control over critical squares.
        """
        score = 0
        pieces = (
            game.board.positions('N') + game.board.positions('B') +
            game.board.positions('R') + game.board.positions('Q') + 
            game.board.positions('K')
        ) if turn == 'w' else (
            game.board.positions('n') + game.board.positions('b') +
            game.board.positions('r') + game.board.positions('q') +
            game.board.positions('k')
        )

        for piece in pieces:
            attacks = game.get_piece_attacks(game.board.get_piece(piece), piece)
            for target in attacks:
                target_piece = game.board.get_piece(target)
                if target_piece != ' ' and (
                    target_piece.islower() if turn == 'w' else target_piece.isupper()
                ):
                    # Reward attacks and mutual support
                    score += 20  # Base reward
                    if self.is_attacked_by_ally(game, target, turn):
                        score += 90  # Extra for coordinated attacks
                    if self.is_defended(game, target, turn):
                        score += 50  # Reward supported squares
        return score


    def is_attacked_by_ally(self, game, pos, turn):
        """
        Check if a square is attacked by another friendly piece.
        """
        for piece_type in ['P', 'N', 'B', 'R', 'Q', 'K'] if turn == 'w' else ['p', 'n', 'b', 'r', 'q', 'k']:
            for piece in game.board.positions(piece_type):
                if pos in game.get_piece_attacks(game.board.get_piece(piece), piece):
                    return True
        return False



# => <= #

    def predict_opponent_moves(self, game, look_ahead_depth=32):
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

    def evaluate_multi_piece_coordination(self, game, turn):
        """
        Evaluate coordination between all available pieces for a powerful coordinated attack.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            float: Multi-piece coordination score
        """
        score = 0
        opponent_king = game.find_king('b' if turn == 'w' else 'w')
        if not opponent_king:
            return score  # No king found, skip evaluation

        # Get all attacking pieces
        attacking_pieces = []
        for piece_type in ['N', 'B', 'R', 'Q', 'K'] if turn == 'w' else ['n', 'b', 'r', 'q', 'k']:
            for pos in game.board.positions(piece_type):
                if opponent_king in game.get_piece_attacks(game.board.get_piece(pos), pos):
                    attacking_pieces.append(pos)
        
        # Reward multiple attackers
        num_attackers = len(attacking_pieces)
        if num_attackers >= 2:
            score += num_attackers * 300  # Reward for each attacking piece
            score += num_attackers * (num_attackers - 1) * 100  # Extra synergy reward for multiple attackers

        # Add score for restricting king's mobility
        restricted_squares = len(self.get_surrounding_squares(game, opponent_king)) - len(
            [sq for sq in self.get_surrounding_squares(game, opponent_king) if game.board.get_piece(sq) == ' ']
        )
        score += restricted_squares * 50  # Reward for limiting king movement

        # Evaluate pawn support around the opponent king
        pawns_supporting = self.get_pawn_support(game, opponent_king, turn)
        score += len(pawns_supporting) * 100  # Reward pawns controlling key squares around the king

        return score


    def get_pawn_support(self, game, king_pos, turn):
        """
        Identify pawns that contribute to restricting the opponent king's mobility.
        
        Args:
            game (Game): Current game state
            king_pos (int): Position of the opponent king
            turn (str): 'w' or 'b'
        
        Returns:
            list: List of pawns supporting the attack
        """
        pawns = game.board.positions('P' if turn == 'w' else 'p')
        supporting_pawns = []
        for pawn in pawns:
            if king_pos in game.get_piece_attacks(game.board.get_piece(pawn), pawn):
                supporting_pawns.append(pawn)
        return supporting_pawns


    def evaluate_checkmate_patterns(self, game, turn):
        """
        Detect and prioritize known checkmate patterns involving multiple pieces.
        
        Args:
            game (Game): Current game state
            turn (str): 'w' or 'b'
        
        Returns:
            float: Checkmate pattern score
        """
        score = 0
        opponent_king = game.find_king('b' if turn == 'w' else 'w')
        if not opponent_king:
            return score

        # Back-rank mate check (already done in original)
        if self.is_back_rank_mate_possible(game, opponent_king, turn):
            score += 500

        # Smothered mate check
        if self.is_smothered_mate_possible(game, opponent_king, turn):
            score += 500

        # Add multi-piece coordination score (already a function)
        score += self.evaluate_multi_piece_coordination(game, turn)

        return score

    def is_back_rank_mate_possible(self, game, king_pos, turn):
        """
        Check if a back-rank mate is possible in the current state.
        """
        x, y = Game.i2xy(king_pos)
        # White tries to mate on opponent’s back rank (which is White’s 1st rank, y=0)
        # Black tries to mate on opponent’s back rank (which is White’s 8th rank from White perspective, y=7)
        if turn == 'w' and y == 0:  # Opponent (black) king on black's back rank
            return True
        elif turn == 'b' and y == 7: # Opponent (white) king on white's back rank
            return True
        return False

    def is_smothered_mate_possible(self, game, king_pos, turn):
        """
        Check if a smothered mate is possible.
        Conditions (simplified):
        - Opponent’s king is in check.
        - The checking piece is a knight.
        - The king has no escape squares because all are occupied by its own pieces or off-board.
        """
        opponent = 'b' if turn == 'w' else 'w'
        if not game.is_in_check(opponent):
            return False

        # Find attackers of the opponent king
        attackers = game.get_attackers(king_pos, turn)
        # Smothered mate is almost always from a single knight
        if len(attackers) != 1:
            return False

        attacker_pos = attackers[0]
        attacker_piece = game.board.get_piece(attacker_pos)
        if attacker_piece.lower() != 'n':
            return False

        # Check if king has any escape squares
        surrounding = self.get_surrounding_squares(game, king_pos)
        for sq in surrounding:
            piece = game.board.get_piece(sq)
            # If an escape square is free or occupied by opponent’s piece (which could be captured),
            # it's not a classic smothered mate scenario.
            if piece == ' ' or (piece.isupper() if opponent == 'w' else piece.islower()):
                return False

        return True



# => <= #
    def evaluate_pawn_advancement(self, game, pawn_pos):
        """
        Evaluate the advancement potential of a single pawn, rewarding proximity
        to promotion, passed pawn status, and penalizing blockages.

        Args:
            game (Game): Current game state.
            pawn_pos (int): Position of the pawn.

        Returns:
            float: Score reflecting pawn advancement potential.
        """
        """
        Evaluate the advancement potential of a single pawn, rewarding:
        - Closeness to promotion
        - Being a passed pawn
        - Being unobstructed by enemy pieces (not blocked)
        
        Consider perspective: White pawns advance upward (y increasing), Black pawns downward (y decreasing).
        """
        piece = game.board.get_piece(pawn_pos)
        if piece == ' ':
            return 0

        is_white = piece.isupper()
        x, y = Game.i2xy(pawn_pos)

        # Distance to promotion:
        # For White: promotion rank is y=7 (8th rank in human terms)
        # Distance to promotion = 7 - y
        # For Black: promotion rank is y=0 (1st rank in human terms)
        # Distance to promotion = y
        distance_to_promotion = (7 - y) if is_white else y

        score = 0
        # Reward greater advancement towards promotion
        # Pawns closer to promotion rank get exponentially larger rewards
        # Encourage aggressive push for promotion
        score += (7 - distance_to_promotion) * 60

        # Check if passed pawn
        if self.is_passed_pawn(game, pawn_pos):
            # Passed pawns are extremely valuable especially as they get closer to promotion
            score += 250

        # Check if the pawn is blocked
        if self.is_blocked_pawn(game, pawn_pos):
            # Penalize blocked pawns as they cannot advance
            score -= 100

        # Additional minor reward if the pawn is protected or supported by another pawn
        # This encourages stable pawn structures advancing together
        if self.is_defended(game, pawn_pos, 'w' if is_white else 'b'):
            score += 40

        return score

    def is_blocked_pawn(self, game, pawn_pos):
        """
        Check if a pawn is blocked by another piece directly ahead.

        Args:
            game (Game): Current game state.
            pawn_pos (int): Position of the pawn.

        Returns:
            bool: True if the pawn is blocked, False otherwise.
        """
        piece = game.board.get_piece(pawn_pos)
        if piece == ' ':
            return False

        is_white = piece.isupper()
        x, y = Game.i2xy(pawn_pos)
        forward_y = y + 1 if is_white else y - 1

        if 0 <= forward_y < 8:
            forward_pos = Game.xy2i(f"{chr(x + ord('a'))}{forward_y + 1}")
            return game.board.get_piece(forward_pos) != ' '
        return False


    def evaluate_pawn_coordination(self, game, pawns, turn):
        """
        Evaluate the coordination of pawns, rewarding clusters and penalizing isolation.
        
        Args:
            game (Game): Current game state.
            pawns (list): List of pawn positions.
            turn (str): 'w' or 'b'.
        
        Returns:
            int: Coordination score for pawns.
        """
        score = 0
        for pawn in pawns:
            x, y = Game.i2xy(pawn)

            # Check for connected pawns
            connected_pawns = 0
            directions = [(-1, 1), (1, 1)] if turn == 'w' else [(-1, -1), (1, -1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    neighbor_pos = Game.xy2i(f"{chr(nx + ord('a'))}{ny + 1}")
                    if neighbor_pos in pawns:
                        connected_pawns += 1

            # Reward connected pawns
            score += connected_pawns * 50

            # Penalize isolated pawns
            if connected_pawns == 0:
                score -= 30

            # Bonus for supported pawns
            if self.is_defended(game, pawn, turn):
                score += 30

        return score


    def evaluate_coordinated_pawn_advancement(self, game, turn):
        """
        Evaluate all pawns for advancement with coordination considerations.
        
        Args:
            game (Game): Current game state.
            turn (str): 'w' or 'b'.
        
        Returns:
            int: Total pawn advancement score.
        """
        """
        Evaluate all pawns for advancement and coordination.
        
        This includes:
        - Summing up individual advancement potentials from evaluate_pawn_advancement.
        - Encouraging pawn chains, connected pawns, and avoiding isolated or backward pawns.
        - Considering synergy with the king in the endgame to push for promotion-based checkmates.
        
        Args:
            game (Game): Current game state.
            turn (str): 'w' or 'b', perspective of the current player.

        Returns:
            int: Total pawn advancement and coordination score.
        """
        pawns = game.board.positions('P' if turn == 'w' else 'p')
        score = 0
        in_endgame = self.is_endgame(game, turn)

        # First, evaluate each pawn’s individual advancement potential
        for pawn in pawns:
            score += self.evaluate_pawn_advancement(game, pawn)

        # Evaluate coordination among pawns:
        # Connected pawns, pawn chains, and supported pawns get bonuses
        # Conversely, isolated and backward pawns are penalized (already accounted in their advancement scores)
        # Here we provide additional synergy bonuses.

        # Reward clusters of connected pawns
        # Connected pawns often appear in "chains" diagonally supporting each other.
        # We'll scan for connected pairs to give synergy bonuses.
        # For White, pawns often support each other going upward, for Black downward.

        # The following logic checks for connected pawns:
        is_white = (turn == 'w')
        directions = [(1, 1), (-1, 1)] if is_white else [(1, -1), (-1, -1)]
        
        # Convert pawns positions to a set for quick lookup
        pawns_set = set(pawns)
        connected_bonus = 0
        for pawn in pawns:
            x, y = Game.i2xy(pawn)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    neighbor = Game.xy2i(f"{chr(nx + ord('a'))}{ny + 1}")
                    if neighbor in pawns_set:
                        # Connected pawns reinforcing each other
                        connected_bonus += 50

        score += connected_bonus

        # Additional factor: if in endgame, encourage king support of pawns
        # This encourages pushing pawns with the king’s help to achieve promotion.
        if in_endgame:
            king_pos = game.find_king(turn)
            if king_pos is not None:
                # Reward pawns close to the king, as the king can help escort them
                for pawn in pawns:
                    distance = self.distance(king_pos, pawn)
                    # Closer pawns to the king are easier to promote with king’s help
                    # The smaller the distance, the greater the synergy bonus
                    if distance <= 3:
                        score += (4 - distance) * 30  # Encourage king-pawn synergy significantly

        return score

    def evaluate_key_square_control(self, game, piece_pos, turn):
        """
        Evaluate control of key squares by high-rank pieces.
        
        Args:
            game (Game): Current game state.
            piece_pos (int): Position of the high-rank piece.
            turn (str): 'w' or 'b'.
        
        Returns:
            int: Key square control score.
        """
        score = 0
        key_squares = self.get_key_squares(game, turn)
        piece_attacks = game.get_piece_attacks(game.board.get_piece(piece_pos), piece_pos)

        for square in key_squares:
            if square in piece_attacks:
                score += 50  # Reward for controlling a key square

                # Bonus if the square is already controlled by another piece
                if self.is_attacked_by_ally(game, square, turn):
                    score += 30

                # Penalize if the square is undefended
                if not self.is_defended(game, square, turn):
                    score -= 20

        return score


    def evaluate_piece_on_key_squares(self, game, turn):
        """
        Evaluate high-rank pieces for placement on key squares.
        
        Args:
            game (Game): Current game state.
            turn (str): 'w' or 'b'.
        
        Returns:
            int: Total score for high-rank pieces on key squares.
        """
        high_rank_pieces = ['Q', 'R', 'B', 'N'] if turn == 'w' else ['q', 'r', 'b', 'n']
        score = 0

        for piece_type in high_rank_pieces:
            for pos in game.board.positions(piece_type):
                score += self.evaluate_key_square_control(game, pos, turn)

        return score

    def get_key_squares(self, game, turn):
        """
        Define key squares dynamically based on the game phase and perspective.

        In the endgame, focus on squares near promotion.
        In earlier stages, center and critical squares are key.

        Args:
            game (Game): Current game state.
            turn (str): 'w' or 'b'

        Returns:
            list: Positions of key squares.
        """
        if self.is_endgame(game, turn):
            # In the endgame, key squares might be those close to promotion
            # White: ranks 6,7 are crucial (7th and 8th rank from white's POV)
            # Black: ranks 0,1 are crucial (1st and 2nd rank from white's POV, top for black)
            ranks = [6, 7] if turn == 'w' else [0, 1]
            key_positions = [
                Game.xy2i(f"{chr(x + ord('a'))}{rank + 1}")
                for x in range(8) for rank in ranks
            ]
        else:
            # Middle or early stage: classical central and important squares
            central_squares = ['d4', 'd5', 'e4', 'e5', 'c3', 'c6', 'f3', 'f6']
            key_positions = [Game.xy2i(sq) for sq in central_squares]

        return key_positions


    def evaluate_endgame_checkmate_potential(self, game, turn):
        """
        Evaluate the endgame potential for checkmate using known patterns and king restrictions.
        Integrates is_ladder_mate_possible, is_rook_king_endgame and also checks restricted squares.
        """
        score = 0
        opponent_king = game.find_king('b' if turn == 'w' else 'w')
        if not opponent_king:
            return score

        # Restrict opponent king's mobility
        restricted_squares = len(self.get_surrounding_squares(game, opponent_king)) - len(
            [sq for sq in self.get_surrounding_squares(game, opponent_king) if game.board.get_piece(sq) == ' ']
        )
        score += restricted_squares * 50

        # Ladder mate possibility
        if self.is_ladder_mate_possible(game, opponent_king, turn):
            score += 500

        # Rook+king endgame pattern
        if self.is_rook_king_endgame(game, opponent_king, turn):
            score += 300

        return score


    def evaluate_king_pawn_coordination(self, game, turn):
        """
        Evaluate coordination between the king and pawns.
        """
        score = 0
        pawns = game.board.positions('P' if turn == 'w' else 'p')
        king_pos = game.find_king(turn)
        for pawn in pawns:
            distance = self.distance(king_pos, pawn)
            if distance <= 2:
                score += 50  # Reward proximity to pawn
        return score

    def evaluate_queen_potential(self, game, queen_pos, turn):
        """
        Evaluate the queen's attack potential based on its influence, threats, and safety.
        
        Args:
            game (Game): Current game state.
            queen_pos (int): Position of the queen.
            turn (str): 'w' or 'b'.
        
        Returns:
            int: Queen's attack potential score.
        """
        score = 0
        queen_piece = game.board.get_piece(queen_pos)
        attacks = game.get_piece_attacks(queen_piece, queen_pos)

        # Reward for attacking multiple squares
        score += len(attacks) * 10

        # Bonus for attacking high-value targets
        for target in attacks:
            target_piece = game.board.get_piece(target)
            if target_piece != ' ':
                # Use absolute value to determine how valuable the target is
                piece_value = abs(self.get_piece_value(target_piece, turn))
                score += piece_value * 2

        # Reward queen's influence on key squares
        key_squares = self.get_key_squares(game, turn)
        for square in key_squares:
            if square in attacks:
                score += 20

        # Penalize queen exposure
        if self.is_attacked(game, queen_pos, turn):
            score -= 50

        return score

    def evaluate_rook_potential(self, game, rook_pos, turn):
        """
        Evaluate a rook's activity, control of files/ranks, and synergy.
        """
        score = 0
        attacks = game.get_piece_attacks(game.board.get_piece(rook_pos), rook_pos)

        # Reward for open file control
        file = rook_pos % 8
        is_open_file = all(
            game.board.get_piece(file + i * 8) in [' ', 'P', 'p']
            for i in range(8)
        )
        if is_open_file:
            score += 50

        # Reward for controlling ranks
        score += len(attacks) * 10

        # Synergy with other rooks
        rooks = game.board.positions('R' if turn == 'w' else 'r')
        if len(rooks) > 1 and rook_pos in rooks:
            score += 30  # Reward for rook coordination

        return score

    def evaluate_knight_potential(self, game, knight_pos, turn):
        """
        Evaluate a knight's centralization, threats, and synergy.
        """
        score = 0
        knight_piece = game.board.get_piece(knight_pos)
        attacks = game.get_piece_attacks(knight_piece, knight_pos)

        # Centralization
        central_squares = [Game.xy2i(sq) for sq in ['d4', 'e4', 'd5', 'e5']]
        if knight_pos in central_squares:
            score += 40

        # Attacking high-value targets
        for square in attacks:
            target = game.board.get_piece(square)
            if target != ' ' and ((target.islower() if turn == 'w' else target.isupper())):
                score += abs(self.get_piece_value(target, turn)) * 2

        # Penalize edge positions
        if knight_pos % 8 in [0, 7] or knight_pos // 8 in [0, 7]:
            score -= 30

        return score

    def evaluate_bishop_potential(self, game, bishop_pos, turn):
        """
        Evaluate a bishop's diagonal control, long-range influence, and synergy.
        """
        score = 0
        attacks = game.get_piece_attacks(game.board.get_piece(bishop_pos), bishop_pos)

        # Reward long-range influence
        score += len(attacks) * 10

        # Reward diagonal control of key squares
        key_diagonals = [
            Game.xy2i(sq)
            for sq in ['c1', 'f1', 'a4', 'h5', 'd4', 'e5', 'a1', 'h8']
        ]
        for square in attacks:
            if square in key_diagonals:
                score += 20

        # Synergy with pawns
        pawns = game.board.positions('P' if turn == 'w' else 'p')
        for pawn in pawns:
            if self.distance(bishop_pos, pawn) <= 2:
                score += 15  # Reward synergy with pawns

        return score
    def evaluate_checkmate_coordination(self, game, turn):
        """
        Evaluate scenarios where pieces coordinate to checkmate.
        This includes applying pressure around the opponent's king and limiting
        its mobility, already defined above.
        """
        score = 0
        opponent = 'b' if turn == 'w' else 'w'
        opponent_king = game.find_king(opponent)
        if not opponent_king:
            return score  # No king found, skip evaluation
        
        king_attacks = 0
        key_supporting_pieces = 0
        key_squares_near_king = self.get_surrounding_squares(game, opponent_king)

        # Evaluate pressure around the opponent's king
        for square in key_squares_near_king:
            if self.is_attacked_by_ally(game, square, turn):
                king_attacks += 1
            if self.is_defended(game, square, turn):
                key_supporting_pieces += 1

        # Collaborative checkmate bonus
        score += king_attacks * 200  # Attacks around king
        score += key_supporting_pieces * 100  # Support near king

        # Extra reward if opponent is currently in check
        if game.is_in_check(opponent):
            score += 500

        return score


    def evaluate_proactivity(self, game, move, turn):
        """
        Evaluate the proactivity of a move, rewarding:
        - Pieces moving out from their starting ranks (developing moves)
        - Moves controlling key squares
        - Captures that favor the current player

        Args:
            game (Game): Current game state.
            move (str): Move in standard chess notation (e.g., 'e2e4').
            turn (str): 'w' or 'b'

        Returns:
            float: Proactivity score for the move.
        """
        score = 0
        move_piece = game.board.get_piece(Game.xy2i(move[:2]))
        target_square = move[2:4]
        initial_rank = int(move[1])

        # Encourage development
        if move_piece.isupper():
            if initial_rank in [1, 2, 3]:
                score += 50
        else:
            if initial_rank in [6, 7, 8]:
                score += 50

        # Key squares
        KEY_SQUARES = ['d4', 'd5', 'e4', 'e5', 'c3', 'c6', 'f3', 'f6']
        key_positions = [Game.xy2i(sq) for sq in KEY_SQUARES]
        target_pos = Game.xy2i(target_square)
        if target_pos in key_positions:
            score += 100

        # Beneficial captures
        target_piece = game.board.get_piece(target_pos)
        if target_piece != ' ' and ((move_piece.isupper() and target_piece.islower()) or
                                    (move_piece.islower() and target_piece.isupper())):
            target_val = abs(self.get_piece_value(target_piece, turn))
            score += target_val * 1.2

            # If it's a favorable trade
            move_piece_val = abs(self.get_piece_value(move_piece, turn))
            if target_val > move_piece_val:
                score += 150

        return score

    def get_piece_moves(self, game, position):
        """
        Retrieve all legal moves for the piece at the given position.
        """
        moves = []
        piece = game.board.get_piece(position)
        if not piece:
            return moves  # No piece at the given position
        
        potential_moves = self.get_piece_attacks(piece, position, game)
        for target in potential_moves:
            simulated_game = Game(game.get_fen())
            simulated_game.apply_move(Game.i2xy(position) + Game.i2xy(target))
            if not simulated_game.is_in_check(game.turn):
                moves.append(Game.i2xy(position) + Game.i2xy(target))
        
        return moves


    def get_piece_attacks(self, piece, position, game):
        """
        Calculate all possible attack squares for a piece at a given position.

        Args:
            piece (str): The piece character (e.g., 'P', 'N', 'B', 'R', 'Q', 'K').
            position (int): The index of the piece's position on the board (0-63).
            game (Game): The current game state.

        Returns:
            list: A list of target positions (indices) where the piece can move or attack.
        """
        piece = piece.upper()  # Handle piece type uniformly
        x, y = Game.i2xy(position)  # Convert position to 2D coordinates
        attacks = []

        # Define movement rules for each piece type
        directions = {
            'P': [(0, 1), (1, 1), (-1, 1)],  # Pawns move forward and capture diagonally
            'N': [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)],
            'B': [(1, 1), (1, -1), (-1, -1), (-1, 1)],  # Diagonals
            'R': [(0, 1), (1, 0), (0, -1), (-1, 0)],  # Straights
            'Q': [(1, 1), (1, -1), (-1, -1), (-1, 1), (0, 1), (1, 0), (0, -1), (-1, 0)],  # Queen is B+R
            'K': [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)],  # Adjacent squares
        }

        # Get piece's movement rules
        moves = directions.get(piece, [])
        if piece.isupper():  # White piece
            # Handle pawn's unique move rules (direction depends on color)
            moves = [(dx, dy) for dx, dy in original_pawn_moves]
        else:  # Black piece
            moves = [(dx, -dy) for dx, dy in original_pawn_moves]


        # Calculate attack positions based on movement rules
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:  # Stay within board bounds
                target_pos = Game.xy2i(f"{chr(nx + ord('a'))}{ny + 1}")
                if piece in 'P':
                    # Pawns attack diagonally only if an opponent's piece is present
                    target_piece = game.board.get_piece(target_pos)
                    if target_piece and (target_piece.islower() if game.turn == 'w' else target_piece.isupper()):
                        attacks.append(target_pos)
                elif piece in 'BQR':
                    # Sliding pieces (bishop, rook, queen) move multiple squares
                    for step in range(1, 8):
                        sx, sy = x + step * dx, y + step * dy
                        if 0 <= sx < 8 and 0 <= sy < 8:
                            slide_pos = Game.xy2i(f"{chr(sx + ord('a'))}{sy + 1}")
                            target_piece = game.board.get_piece(slide_pos)
                            attacks.append(slide_pos)
                            if target_piece:  # Stop sliding if hitting a piece
                                break
                        else:
                            break
                else:
                    # Knights, kings move to their calculated target
                    attacks.append(target_pos)

        return attacks

    def calculate_material_counts(self, game):
        """
        Calculate the material count for each type of piece on the board.

        Args:
            game (Game): The current game state.

        Returns:
            dict: Counts of each piece type on the board.
        """
        counts = {piece: 0 for piece in 'PNBRQKpnbrqk'}
        for piece in game.board.pieces:
            if piece in counts:
                counts[piece] += 1
        return counts

    def determine_game_stage(self, game):
        """
        Determine the current game stage by comprehensively analyzing board state for both players.
    
        Args:
            game (Game): The current game state.
    
        Returns:
            str: The current game stage ('opening', 'midgame', 'endgame').
        """
        # Calculate material counts for both players
        material_counts = self.calculate_material_counts(game)
        
        # Piece presence and counts
        def count_pieces(prefix):
            return {
                'queens': material_counts.get(prefix.upper() + 'Q', 0),
                'rooks': material_counts.get(prefix.upper() + 'R', 0),
                'bishops': material_counts.get(prefix.upper() + 'B', 0),
                'knights': material_counts.get(prefix.upper() + 'N', 0),
                'total_major': (
                    material_counts.get(prefix.upper() + 'Q', 0) + 
                    material_counts.get(prefix.upper() + 'R', 0)
                ),
                'total_minor': (
                    material_counts.get(prefix.upper() + 'B', 0) + 
                    material_counts.get(prefix.upper() + 'N', 0)
                )
            }
        
        white_pieces = count_pieces('')
        black_pieces = count_pieces('q')
        
        # Mobility analysis for both players
        def calculate_mobility(color_upper):
            return sum(
                len(self.get_piece_moves(game, pos)) 
                for pos in game.board.positions() 
                if game.board.get_piece(pos).upper() == color_upper
            )
        
        white_mobility = calculate_mobility('W')
        black_mobility = calculate_mobility('B')
        
        # Comprehensive game stage determination
        def analyze_stage():
            # Opening criteria: abundant pieces, limited development
            if (
                (white_pieces['queens'] > 0 or black_pieces['queens'] > 0) and
                (white_pieces['total_major'] >= 4 and black_pieces['total_major'] >= 4) and
                (white_pieces['total_minor'] >= 4 and black_pieces['total_minor'] >= 4) and
                (white_mobility < 50 or black_mobility < 50)  # Limited initial mobility
            ):
                return 'opening'
            
            # Midgame criteria: balanced material, active pieces
            elif (
                (white_pieces['queens'] > 0 and black_pieces['queens'] > 0) and
                (white_pieces['total_major'] > 2 and black_pieces['total_major'] > 2) and
                (white_pieces['total_minor'] > 2 and black_pieces['total_minor'] > 2) and
                (50 <= white_mobility <= 100 and 50 <= black_mobility <= 100)
            ):
                return 'midgame'
            
            # Endgame criteria: reduced material, limited piece presence
            else:
                return 'endgame'
        
        # Calculate total material for both players
        total_white_material = sum(
            material_counts.get(piece, 0) 
            for piece in ['Q', 'R', 'B', 'N', 'P']
        )
        total_black_material = sum(
            material_counts.get(piece, 0) 
            for piece in ['q', 'r', 'b', 'n', 'p']
        )
        
        # Fine-tune stage detection with material comparison
        game_stage = analyze_stage()
        
        # Additional refinement based on total material
        if game_stage == 'opening' and total_white_material + total_black_material < 24:
            return 'midgame'
        elif game_stage == 'midgame' and total_white_material + total_black_material < 16:
            return 'endgame'
        
        return game_stage

    def choose_search_depth(self, game):
        """
        Adaptive depth strategy based on the current game stage.

        Args:
            game (Game): The current game state.

        Returns:
            int: Chosen search depth.
        """
        game_stage = self.determine_game_stage(game)
        if game_stage == 'opening':
            return 16   # Faster decision-making
        elif game_stage == 'midgame':
            return 24  # Balanced depth
        elif game_stage == 'endgame':
            return 32  # More thorough analysis
        else:
            return self.max_search_depth

    def is_ladder_mate_possible(self, game, king_pos, turn):
        """
        Heuristic check for ladder mate:
        - Opponent’s king on a back rank (depending on turn).
        - Two rooks or a rook and queen confine the king.
        - No escape squares for the king.
        """
        opponent = 'b' if turn == 'w' else 'w'
        x, y = Game.i2xy(king_pos)

        # Check if king is on back rank from perspective of the attacker
        # For white attacking black: black king on y=0
        # For black attacking white: white king on y=7
        if turn == 'w' and y != 0:
            return False
        if turn == 'b' and y != 7:
            return False

        # Find rooks/queens of attacking side
        friendly_heavy = (game.board.positions('R') + game.board.positions('Q')) if turn == 'w' else (game.board.positions('r') + game.board.positions('q'))
        if len(friendly_heavy) < 2:
            return False

        # Check if king has no escape squares
        surrounding = self.get_surrounding_squares(game, king_pos)
        for sq in surrounding:
            piece = game.board.get_piece(sq)
            # If escape square is empty or occupied by opponent’s piece that can be captured,
            # it's not a closed ladder mate position.
            if piece == ' ' or (piece.isupper() if opponent == 'w' else piece.islower()):
                return False

        # This is a very simplistic check. If conditions met, assume ladder mate possible.
        return True

    def is_rook_king_endgame(self, game, king_pos, turn):
        """
        Check if conditions for a rook+king vs king endgame checkmate pattern exist:
        - Opponent has only king left.
        - The attacking side has at least one rook.
        - Opponent king is near edge and trapped by the attacking king and rook.
        """
        opponent = 'b' if turn == 'w' else 'w'

        # Check opponent's material: only a king?
        opp_pieces = [p for p in game.board.pieces if (p.isupper() if opponent == 'w' else p.islower()) and p.lower() != 'k']
        if len(opp_pieces) > 0:
            return False

        # Attacker must have a rook
        if turn == 'w':
            rooks = game.board.positions('R')
        else:
            rooks = game.board.positions('r')
        if len(rooks) == 0:
            return False

        # Check if opponent king is on an edge and restricted
        x, y = Game.i2xy(king_pos)
        if x not in [0,7] and y not in [0,7]:
            return False

        # If the king is on edge and we have rook+king, assume potential for rook+king checkmate pattern
        return True


def chess_bot(obs):
    """
    Chess bot interface that provides strategic move selection.
    
    Args:
        obs: An object with a 'board' attribute representing the board state
    
    Returns:
        str: Selected move in UCI notation
    """
    agent = StrategicChessAgent(max_search_depth=32)
    return agent.select_strategic_move(obs.board)
