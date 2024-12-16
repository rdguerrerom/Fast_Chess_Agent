%%writefile submission.py
from Chessnut import Game
import random
import math
import copy
from collections import Counter


class StrategicChessAgent:
    # Piece value constants
    BASE_PIECE_VALUES = {
        "P": 100,
        "N": 1320,
        "B": 1330,
        "R": 1500,
        "Q": 1900,
        "K": 20000,
        "p": -110,
        "n": -1452,
        "b": -1463,
        "r": -1650,
        "q": -2090,
        "k": -22000,
    }

    def __init__(self, max_search_depth=16):
        self.move_cache = {}
        self.opponent_move_probability = {}
        self.move_sequence_cache = {}
        self.max_search_depth = max_search_depth
        self.original_pawn_moves = [(0, 1), (1, 1), (-1, 1)]
        self.opening_moves = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": [
                "e2e4",
                "d2d4",
                "c2c4",
                "g1f3",
            ],
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": [
                "e7e5",
                "c7c5",
            ],
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1": [
                "e7e6",
                "d7d5",
            ],
            "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1": [
                "e7e6",
                "d7d5",
            ],
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1": [
                "e7e6",
                "d7d5",
            ],
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2": ["d2d4"],
            "1e4 e5 2Nf3 Nc6 3Bb5": ["a7a6", "d7d6", "f8e7"],
            "1e4 e5 2Nf3 Nc6 3Bc4": ["g8f6", "f8c5"],
            "1e4 e5 2Nf3 Nc6 3d4": ["e5d4", "c6d4"],
            "1e4 c5": ["g1f3", "c2c3"],
            "1e4 e6": ["d2d4", "g1f3"],
            "1e4 c6": ["d2d4", "g1f3"],
            "1d4 d5 2c4": ["e7e6", "c7c6"],
            "1d4 Nf6 2c4 g6": ["g2g3", "b1c3"],
            "1d4 Nf6 2c4 g6 3Nc3 d5": ["c4d5", "e2e4"],
            "1c4": ["e7e5", "g8f6"],
            "1Nf3": ["d7d5", "g8f6"],
            "1g3": ["d7d5", "g8f6"],
            "1e4": ["c5", "e6", "c6", "g6"],
            "1d4": ["d5", "Nf6", "e6"],
            "1c4": ["e5", "c5"],
            "1Nf3": ["d5", "Nf6"],
        }

    def get_piece_value(self, piece, perspective="w"):
        val = BASE_PIECE_VALUES.get(piece, 0)
        if perspective == "b":
            # Flip the sign for all values to mirror perspective
            val = -val
        return val

    def simplify_fen(self, fen):
        parts = fen.split()
        if len(parts) < 4:
            raise ValueError("Invalid FEN format")

        simplified = " ".join(parts[:3])
        return simplified

    def get_game_turn(self, game):
        fen_parts = game.get_fen().split()
        return fen_parts[1]  # 'w' or 'b'

    def evaluate_move(self, game, move):
        score = 0
        turn = self.get_game_turn(game)
        game_stage = self.determine_game_stage(game)

        try:
            test_game = Game(game.get_fen())
            test_game.apply_move(move)

            # Baseline checkmate detection
            if test_game.status == test_game.CHECKMATE:
                return 9999999

            # Weighted evaluation factors
            weights = {
                "checkmate_acceleration": 1900.0,  # Boosted for direct wins
                "tactical_potential": 800.0,  # Stronger focus on tactics
                "material_advantage": 110.0,  # Balanced but important
                "positional_advantage": 95.0,  # Slightly increased
                "king_safety": 40.0,  # Reduced to encourage risk-taking
                "pawn_structure": 80.0,  # Encourage pawn play
                "positional_factor": 100.0,  # Increased for strategic positioning
                "bottleneck": 50.0,  # Penalize restricted positions
                "aggressive_defense": 200.0,  # Counterattack emphasis
                "strategic_planning": 310.0,  # Better long-term planning
                "king_activity": 900.0,  # Active king in endgame
                "rook_king_coordination": 700.0,  # Promote teamwork
                "king_pawn_coordination": 800.0,  # Endgame synergy
                "multi_piece_coordination": 1900.0,  # Improved coordination
                "checkmate_coordination": 1200.0,  # Prioritize checkmate setups
                "minimax": 0.0,  # Dynamically adjusted (e.g., 50.0 in endgame)
            }

            # Stage-specific weight adjustments
            if game_stage == "opening":
                weights["piece_coordination"] *= 1.3
                weights["material_advantage"] *= 1.2
                weights["minimax"] = 15.0
            elif game_stage == "midgame":
                weights["tactical_potential"] *= 1.8
                weights["king_safety"] *= 1.1
                weights["minimax"] = 25.0
            else:  # endgame
                weights["checkmate_acceleration"] *= 2.5
                weights["minimax"] = 50.0

            # Accumulate weighted scores
            score += self.calculate_checkmate_acceleration_bonus(test_game, turn) * weights["checkmate_acceleration"]
            score += self.evaluate_piece_coordination(test_game, turn) * weights["piece_coordination"]
            score += self.evaluate_material_advantage(game, move) * weights["material_advantage"]
            score += self.evaluate_positional_advantage(game, move) * weights["positional_advantage"]
            score += self.evaluate_tactical_potential(test_game, turn) * weights["tactical_potential"]
            score += self.evaluate_king_safety(test_game, turn) * weights["king_safety"]
            score += self.evaluate_pawn_structure(test_game, turn, self.is_endgame(test_game, turn)) * weights["pawn_structure"]
            score += self.evaluate_positional_factors(test_game, turn, self.is_endgame(test_game, turn)) * weights["positional_factor"]
            score += self.evaluate_bottleneck(test_game, turn) * weights["bottleneck"]
            score += self.evaluate_aggressive_defense(test_game, turn, self.is_endgame(test_game, turn)) * weights["aggressive_defense"]
            score += self.evaluate_strategic_planning(test_game, turn) * weights["strategic_planning"]
            score += self.evaluate_king_activity(test_game, turn) * weights["king_activity"]
            score += self.evaluate_rook_king_coordination(test_game, turn) * weights["rook_king_coordination"]
            score += self.evaluate_king_pawn_coordination(test_game, turn) * weights["king_pawn_coordination"]
            score += self.evaluate_multi_piece_coordination(test_game, turn) * weights["multi_piece_coordination"]
            score += self.evaluate_checkmate_coordination(test_game, turn) * weights["checkmate_coordination"]
            score += weights["minimax"] * self.minimax(
                test_game,
                self.choose_search_depth(game),
                -math.inf,
                math.inf,
                False,
            )

            # Add randomness to prevent predictability
            score += random.random() * 10

        except Exception:
            score = random.random() * 10

        return score

    def choose_search_depth(self, game):
        turn = self.get_game_turn(game)  # 'w' or 'b'
        game_stage = self.determine_game_stage(game)  # 'opening', 'midgame', 'endgame'

        # Base depths for each stage
        if game_stage == "opening":
            base_depth = 16
        elif game_stage == "midgame":
            base_depth = 24
        elif game_stage == "endgame":
            base_depth = 32
        else:
            base_depth = 40

        moves = game.get_moves()
        num_moves = len(moves)

        material_diff = self.calculate_material_difference(game, turn)
        pawn_promotion_potential = self.evaluate_pawn_promotion(game, turn)

        if num_moves > 30:
            base_depth += 2
        elif num_moves < 10:
            base_depth += 3

        if game_stage == "endgame" and material_diff > 2:
            base_depth += 4

        if game_stage == "midgame" and material_diff < 0:
            base_depth += 3

        if pawn_promotion_potential > 200:
            base_depth += 2

        base_depth = min(base_depth, 40)

        return base_depth

    def calculate_material_difference(self, game, turn):
        # Define piece values
        values = {
            "P": 1,
            "N": 3,
            "B": 3,
            "R": 5,
            "Q": 9,
            "K": 0,
            "p": -1,
            "n": -3,
            "b": -3,
            "r": -5,
            "q": -9,
            "k": 0,
        }

        # Sum material values based on the board state
        board = game.get_fen().split()[0].replace("/", "")
        material_diff = sum(values.get(piece, 0) for piece in board)

        # Adjust sign if the agent plays as black
        return material_diff if turn == "white" else -material_diff

    def calculate_checkmate_acceleration_bonus(self, test_game, turn):
        own_king_pos = test_game.board.get_king_position(turn)
        opp_king_pos = test_game.board.get_king_position("b" if turn == "w" else "w")

        king_distance = abs(own_king_pos[0] - opp_king_pos[0]) + abs(own_king_pos[1] - opp_king_pos[1])
        material_diff = self.calculate_material_difference(test_game, turn)

        checkmate_acceleration_score = (8 - king_distance) * 1000 + (material_diff * 500) + (2000 if self.is_check(test_game, turn) else 0) + (5000 if self.can_force_checkmate(test_game, turn) else 0)

        return checkmate_acceleration_score

    def minimax(self, game, depth, alpha, beta, maximizing_player):
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
        strategic_score = 0

        pawn_chains = self.evaluate_pawn_chains(game, turn)
        strategic_score += pawn_chains * 50  # Reward for strong pawn chains

        piece_coordination = self.evaluate_piece_coordination(game, turn)
        strategic_score += piece_coordination * 40  # Reward for good coordination

        key_control = self.evaluate_multiple_key_control(game, turn)
        strategic_score += key_control * 60  # Significant reward for controlling multiple key squares

        return strategic_score

    def evaluate_pawn_chains(self, game, turn):
        score = 0
        pawns = [pos for pos in game.board.positions("P" if turn == "w" else "p")]
        pawn_positions = sorted(
            pawns,
            key=lambda pos: Game.i2xy(pos)[1],
            reverse=(turn == "w"),
        )

        for pawn in pawn_positions:
            x, y = Game.i2xy(pawn)
            for af in adjacent_files:
                if 0 <= af < 8:
                    behind_pos = Game.xy2i(f"{chr(af + ord('a'))}{y - 1 + (1 if turn == 'w' else -1)}")
                    if behind_pos in pawns:
                        score += 20  # Reward for connected pawns
        return score

    def evaluate_piece_coordination(self, game, turn):
        score = 0
        if turn == "w":
            pieces = game.board.positions("N") + game.board.positions("B") + game.board.positions("R") + game.board.positions("Q") + game.board.positions("K")
        else:
            pieces = game.board.positions("n") + game.board.positions("b") + game.board.positions("r") + game.board.positions("q") + game.board.positions("k")

        for p_pos in pieces:
            piece_type = game.board.get_piece(p_pos)
            attacks = game.get_piece_attacks(piece_type, p_pos)
            for target in attacks:
                target_piece = game.board.get_piece(target)
                if target_piece != " " and ((piece_type.isupper() and target_piece.islower()) or (piece_type.islower() and target_piece.isupper())):
                    score += 20

                    if self.is_attacked_by_ally(game, target, turn):
                        score += 90

                    if self.is_defended(game, target, turn):
                        score += 50

        return score

    def evaluate_multiple_key_control(self, game, turn):
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
        major_pieces = ["Q", "R", "q", "r"]
        minor_pieces = ["N", "B", "n", "b"]
        queens = sum(1 for piece in game.board.pieces if piece in ["Q", "q"])
        rooks = sum(1 for piece in game.board.pieces if piece in ["R", "r"])
        bishops = sum(1 for piece in game.board.pieces if piece in ["B", "b"])
        knights = sum(1 for piece in game.board.pieces if piece in ["N", "n"])
        if queens + rooks <= 2 and bishops + knights <= 2:
            return True
        return False

    def is_check(self, game, turn):
        opponent = "b" if turn == "w" else "w"
        return game.is_in_check(opponent)

    def evaluate_tactical_patterns(self, game, turn):
        tactical_score = 0
        opponent = "b" if turn == "w" else "w"

        # Iterate through all pieces to find tactical motifs
        for piece_type in ["N", "B", "R", "Q"] if turn == "w" else ["n", "b", "r", "q"]:
            for pos in game.board.positions(piece_type):
                attacks = game.get_piece_attacks(piece_type, pos)
                for target in attacks:
                    target_piece = game.board.get_piece(target)
                    if target_piece != " " and (target_piece.islower() if turn == "w" else target_piece.isupper()):
                        if self.is_double_threat(game, pos, target, turn):
                            tactical_score += 120
                        if self.is_fork(game, pos, target, turn):
                            tactical_score += 100
                        if self.is_pin(game, pos, target, turn):
                            tactical_score += 80
                        if self.is_skewer(game, pos, target, turn):
                            tactical_score += 80
                        if self.is_discovered_attack(game, pos, target, turn):
                            tactical_score += 90
        return tactical_score

    def is_fork(self, game, attacker_pos, target_pos, turn):
        attacking_piece = game.board.get_piece(attacker_pos)
        if attacking_piece.lower() not in ["n", "b", "r", "q"]:
            return False

        attacks = game.get_piece_attacks(attacking_piece, attacker_pos)
        high_value_targets = 0
        for pos in attacks:
            piece = game.board.get_piece(pos)
            if piece != " " and ((piece.islower() if turn == "w" else piece.isupper())):
                if abs(self.get_piece_value(piece, turn)) >= 300:
                    high_value_targets += 1
        return high_value_targets >= 2

    def is_pin(self, game, attacker_pos, target_pos, turn):
        attacker_piece = game.board.get_piece(attacker_pos)
        target_piece = game.board.get_piece(target_pos)
        if attacker_piece.lower() not in ["b", "r", "q"]:
            return False

        direction = self.get_direction(attacker_pos, target_pos)
        if direction is None:
            return False

        next_pos = target_pos + direction
        while 0 <= next_pos < 64:
            beyond_piece = game.board.get_piece(next_pos)
            if beyond_piece == " ":
                next_pos += direction
                continue
            if (beyond_piece.isupper() and turn == "w") or (beyond_piece.islower() and turn == "b"):
                beyond_val = abs(self.get_piece_value(beyond_piece, turn))
                target_val = abs(self.get_piece_value(target_piece, turn))
                if beyond_val > target_val:
                    return True
            break
        return False

    def is_skewer(self, game, attacker_pos, target_pos, turn):
        attacker_piece = game.board.get_piece(attacker_pos)
        target_piece = game.board.get_piece(target_pos)
        if attacker_piece.lower() not in ["b", "r", "q"]:
            return False

        direction = self.get_direction(attacker_pos, target_pos)
        if direction is None:
            return False

        next_pos = target_pos + direction
        while 0 <= next_pos < 64:
            beyond_piece = game.board.get_piece(next_pos)
            if beyond_piece == " ":
                next_pos += direction
                continue
            if (beyond_piece.isupper() and turn == "w") or (beyond_piece.islower() and turn == "b"):
                # If beyond_piece is a higher-value piece, it's a skewer
                if self.PIECE_VALUES.get(beyond_piece.upper(), 0) > self.PIECE_VALUES.get(target_piece.upper(), 0):
                    return True
            break
        return False

    def is_discovered_attack(self, game, attacker_pos, target_pos, turn):
        current_attacks = self.get_all_attacks(game, turn)

        new_game = Game(game.get_fen())
        new_game.apply_move(move=target_pos)  # Assuming 'move' leads to 'target_pos'

        new_attacks = self.get_all_attacks(new_game, turn)

        discovered_attacks = new_attacks - current_attacks
        return len(discovered_attacks) > 0

    def get_all_attacks(self, game, turn):
        attacks = set()
        for piece_type in ["P", "N", "B", "R", "Q", "K"] if turn == "w" else ["p", "n", "b", "r", "q", "k"]:
            for pos in game.board.positions(piece_type):
                piece = game.board.get_piece(pos)
                attacks.update(game.get_piece_attacks(piece, pos))
        return attacks

    def is_double_threat(self, game, attacker_pos, target_pos, turn):
        attacking_piece = game.board.get_piece(attacker_pos)
        if attacking_piece.lower() not in ["n", "b", "r", "q"]:
            return False

        attacks = game.get_piece_attacks(attacking_piece, attacker_pos)
        high_value_targets = 0
        for pos in attacks:
            piece = game.board.get_piece(pos)
            if piece != " " and ((piece.islower() if turn == "w" else piece.isupper())):
                if abs(self.get_piece_value(piece, turn)) >= 300:
                    high_value_targets += 1
        return high_value_targets >= 2

    def evaluate_mobility(self, game, turn):
        mobility_score = 0
        for piece_type in ["N", "B", "R", "Q", "K"] if turn == "w" else ["n", "b", "r", "q", "k"]:
            for pos in game.board.positions(piece_type):
                piece = game.board.get_piece(pos)
                moves = self.get_piece_moves(piece, pos)
                mobility_score += len(moves) * 10  # Weight mobility
        return mobility_score

    def evaluate_threats(self, game, turn):
        threats_score = 0
        opponent = "b" if turn == "w" else "w"
        for piece_type in ["N", "B", "R", "Q", "P"] if turn == "w" else ["n", "b", "r", "q", "p"]:
            for pos in game.board.positions(piece_type):
                threats = game.get_piece_attacks(piece_type, pos)
                for target in threats:
                    target_piece = game.board.get_piece(target)
                    if target_piece != " " and (target_piece.islower() if turn == "w" else target_piece.isupper()):
                        threats_score += 50  # Bonus for threatening a piece
                    if game.is_attacked(target, opponent):
                        threats_score += 100  # Bonus for threatening the king
        return threats_score

    def evaluate_king_safety(self, game, turn):
        safety_score = 0
        king_pos = game.find_king(turn)
        if not king_pos:
            return safety_score  # Should not happen, but safe guard

        shield_pawns = self.get_pawn_shield(game, king_pos, turn)
        safety_score += len(shield_pawns) * 20  # Reward for each pawn in shield

        if game.is_attacked(king_pos, "b" if turn == "w" else "w"):
            safety_score -= 300  # High penalty if king is in check

        surrounding_squares = self.get_surrounding_squares(game, king_pos)
        open_squares = [sq for sq in surrounding_squares if game.board.get_piece(sq) == " "]
        safety_score -= len(open_squares) * 10  # Penalize for each open square around king

        if in_endgame:
            own_king_pos = test_game.board.get_king_position(turn)
            opp_king_pos = test_game.board.get_king_position("b" if turn == "w" else "w")

            king_proximity_bonus = (
                max(
                    0,
                    14 - (abs(own_king_pos[0] - opp_king_pos[0]) + abs(own_king_pos[1] - opp_king_pos[1])),
                )
                * 200
            )

            safety_score += king_proximity_bonus

        return safety_score

    def get_pawn_shield(self, game, king_pos, turn):
        x, y = Game.i2xy(king_pos)
        directions = [(-1, 1), (0, 1), (1, 1)] if turn == "w" else [(-1, -1), (0, -1), (1, -1)]
        shield_pawns = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                pos = Game.xy2i(f"{chr(nx + ord('a'))}{ny + 1}")
                piece = game.board.get_piece(pos)
                expected_pawn = "P" if turn == "w" else "p"
                if piece == expected_pawn:
                    shield_pawns.append(pos)
        return shield_pawns

    def get_surrounding_squares(self, game, pos):
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
        pawn_score = 0
        pawns = [pos for pos in game.board.positions("P" if turn == "w" else "p")]

        for pawn in pawns:
            if self.is_passed_pawn(game, pawn):
                pawn_score += 100  # Significant bonus for passed pawn
                if in_endgame:
                    pawn_score += 50  # Additional bonus in endgame

            if self.is_doubled_pawn(game, pawn, turn):
                pawn_score -= 50  # Penalty for doubled pawn
            if self.is_isolated_pawn(game, pawn):
                pawn_score -= 50  # Penalty for isolated pawn
            if self.is_backward_pawn(game, pawn):
                pawn_score -= 50  # Penalty for backward pawn

        pawn_majority_score = self.evaluate_pawn_majority(game, turn)
        pawn_score += pawn_majority_score

        return pawn_score

    def is_passed_pawn(self, game, pawn_pos):
        piece = game.board.get_piece(pawn_pos)
        if piece == " ":
            return False

        is_white = piece.isupper()
        x, y = Game.i2xy(pawn_pos)

        if is_white:
            forward_ranks = range(y + 1, 8)  # Check ranks above
            enemy_pawn = "p"
        else:
            forward_ranks = range(y - 1, -1, -1)  # Check ranks below for black
            enemy_pawn = "P"

        for rank in forward_ranks:
            for dx in [x - 1, x, x + 1]:
                if 0 <= dx < 8:
                    pos = Game.xy2i(f"{chr(dx + ord('a'))}{rank + 1}")
                    if game.board.get_piece(pos) == enemy_pawn:
                        return False
        return True

    def is_doubled_pawn(self, game, pawn_pos, turn):
        x, y = Game.i2xy(pawn_pos)
        file = x
        pawns = [pos for pos in game.board.positions("P" if turn == "w" else "p") if Game.i2xy(pos)[0] == file]
        return len(pawns) > 1

    def is_isolated_pawn(self, game, pawn_pos):
        piece = game.board.get_piece(pawn_pos)
        if piece == " ":
            return False

        is_white = piece.isupper()
        x, y = Game.i2xy(pawn_pos)

        adjacent_files = [x - 1, x + 1]

        rank_range = range(y, 8) if is_white else range(y, -1, -1)
        friendly_pawn = "P" if is_white else "p"

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
        piece = game.board.get_piece(pawn_pos)
        if piece == " ":
            return False

        is_white = piece.isupper()
        x, y = Game.i2xy(pawn_pos)
        # Determine the square in front of the pawn
        target_y = y + 1 if is_white else y - 1

        if not (0 <= target_y < 8):
            return False

        forward_pos = Game.xy2i(f"{chr(x + ord('a'))}{target_y + 1}")
        if game.board.get_piece(forward_pos) != " ":
            return True
        return False

    def evaluate_pawn_majority(self, game, turn):
        majority_score = 0
        zones = {
            "left": range(0, 3),  # Files a, b, c
            "center": range(3, 6),  # Files d, e, f
            "right": range(6, 8),  # Files g, h
        }
        for zone_name, files in zones.items():
            own_pawns = len([pos for pos in game.board.positions("P" if turn == "w" else "p") if Game.i2xy(pos)[0] in files])
            opponent_pawns = len([pos for pos in game.board.positions("p" if turn == "w" else "P") if Game.i2xy(pos)[0] in files])
            if own_pawns > opponent_pawns:
                majority_score += (own_pawns - opponent_pawns) * 10  # Reward for pawn majority
            elif own_pawns < opponent_pawns:
                majority_score -= (opponent_pawns - own_pawns) * 10  # Penalize for pawn deficit
        return majority_score

    def evaluate_positional_factors(self, game, turn, in_endgame):
        positional_score = 0

        # Control of the Center
        center_squares = ["d4", "d5", "e4", "e5"]
        for square in center_squares:
            pos = Game.xy2i(square)
            piece = game.board.get_piece(pos)
            if (turn == "w" and piece.isupper()) or (turn == "b" and piece.islower()):
                positional_score += 100  # Bonus for controlling center squares

        open_files = self.get_open_files(game)
        positional_score += len(open_files) * 50  # Bonus for each open file

        open_diagonals = self.get_open_diagonals(game)
        positional_score += len(open_diagonals) * 50  # Bonus for each open diagonal

        controlled_squares = self.count_controlled_squares(game, turn)
        positional_score += controlled_squares * (1 if not in_endgame else 2)  # Slightly higher in endgame

        key_squares = self.get_key_squares(game, turn)
        positional_score += len(key_squares) * 100  # Significant bonus for control of key squares

        zugzwang_score = self.evaluate_zugzwang_potential(game, turn)
        positional_score += zugzwang_score * 100  # High bonus for potential zugzwang

        return positional_score

    def get_open_files(self, game):
        open_files = 0
        for file in range(8):
            has_pawn = False
            for rank in range(8):
                pos = Game.xy2i(f"{chr(file + ord('a'))}{rank + 1}")
                piece = game.board.get_piece(pos)
                if piece.lower() == "p":
                    has_pawn = True
                    break
            if not has_pawn:
                open_files += 1
        return open_files

    def get_open_diagonals(self, game):
        open_diagonals = 0
        for d in range(-7, 8):
            has_pawn = False
            for x in range(8):
                y = x + d
                if 0 <= y < 8:
                    pos = Game.xy2i(f"{chr(x + ord('a'))}{y + 1}")
                    piece = game.board.get_piece(pos)
                    if piece.lower() == "p":
                        has_pawn = True
                        break
            if not has_pawn:
                open_diagonals += 1
        return open_diagonals

    def count_controlled_squares(self, game, turn):
        controlled = set()
        for piece_type in ["P", "N", "B", "R", "Q", "K"] if turn == "w" else ["p", "n", "b", "r", "q", "k"]:
            for pos in game.board.positions(piece_type):
                piece = game.board.get_piece(pos)
                controlled.update(game.get_piece_attacks(piece, pos))
        return len(controlled)

    def get_key_squares(self, game, turn):
        key_squares = set(
            [
                "d4",
                "d5",
                "e4",
                "e5",
                "c3",
                "c6",
                "f3",
                "f6",
                "c4",
                "f4",
                "c5",
                "f5",
            ]
        )  # Expanded for aggressive bottlenecking
        controlled_key_squares = set()
        for square in key_squares:
            pos = Game.xy2i(square)
            piece = game.board.get_piece(pos)
            if (turn == "w" and piece.isupper()) or (turn == "b" and piece.islower()):
                controlled_key_squares.add(pos)
        return controlled_key_squares

    def evaluate_zugzwang_potential(self, game, turn):
        opponent = "b" if turn == "w" else "w"
        opponent_moves = list(game.get_moves(opponent))

        if len(opponent_moves) == 0:
            return 300  # Checkmate already handled elsewhere
        elif len(opponent_moves) <= 2:
            return 200  # High potential
        elif len(opponent_moves) <= 4:
            return 100  # Moderate potential
        return 0

    def evaluate_bottleneck(self, game, turn):
        bottleneck_score = 0
        opponent = "b" if turn == "w" else "w"

        bottleneck_squares = [
            "c4",
            "f4",
            "c5",
            "f5",
            "d4",
            "d5",
            "e4",
            "e5",
            "c3",
            "c6",
            "f3",
            "f6",
        ]  # Include central squares

        for square in bottleneck_squares:
            pos = Game.xy2i(square)
            piece = game.board.get_piece(pos)
            if (turn == "w" and piece.isupper()) or (turn == "b" and piece.islower()):
                bottleneck_score += 100

            opponent_attacks = self.get_controlled_squares(game, opponent)
            if pos in opponent_attacks:
                bottleneck_score += 50

        return bottleneck_score

    def evaluate_aggressive_defense(self, game, turn, in_endgame):
        defense_score = 0

        vulnerable_pieces = self.get_vulnerable_pieces(game, turn)

        for piece_pos in vulnerable_pieces:
            if self.is_defended(game, piece_pos, turn):
                defense_score += 100  # Significant reward for defending a vulnerable piece
            else:
                defense_score -= 100  # Penalty for leaving a piece undefended

        blocking_moves = self.get_blocking_moves(game, turn)
        defense_score += len(blocking_moves) * 50  # Reward for each blocking move

        exposed_pieces = self.get_exposed_pieces(game, turn)
        defense_score -= len(exposed_pieces) * 75  # Penalize for each exposed piece

        return defense_score

    def get_vulnerable_pieces(self, game, turn):
        vulnerable_pieces = []
        for piece_type in ["P", "N", "B", "R", "Q", "K"] if turn == "w" else ["p", "n", "b", "r", "q", "k"]:
            for pos in game.board.positions(piece_type):
                if game.is_attacked(pos, "b" if turn == "w" else "w"):
                    if not self.is_defended(game, pos, turn):
                        vulnerable_pieces.append(pos)
        return vulnerable_pieces

    def is_defended(self, game, pos, turn):
        opponent = "b" if turn == "w" else "w"
        attackers = game.get_attackers(pos, opponent)
        for attacker_pos in attackers:
            if game.is_attacked(attacker_pos, turn):
                return True
        return False

    def get_blocking_moves(self, game, turn):
        blocking_moves = []
        opponent = "b" if turn == "w" else "w"
        for pos in game.board.positions("K" if turn == "w" else "k"):
            attackers = game.get_attackers(pos, opponent)
            for attacker_pos in attackers:
                direction = self.get_direction(attacker_pos, pos)
                if direction is None:
                    continue
                blocker_pos = attacker_pos + direction
                if 0 <= blocker_pos < 64:
                    blocker_piece = game.board.get_piece(blocker_pos)
                    if blocker_piece == " " or blocker_piece.isupper() != (turn == "w"):
                        blocking_moves.append(blocker_pos)
        return blocking_moves

    def get_exposed_pieces(self, game, turn):
        exposed_pieces = []
        for piece_type in ["P", "N", "B", "R", "Q", "K"] if turn == "w" else ["p", "n", "b", "r", "q", "k"]:
            for pos in game.board.positions(piece_type):
                if game.is_attacked(pos, "b" if turn == "w" else "w"):
                    exposed_pieces.append(pos)
        return exposed_pieces

    def evaluate_endgame(self, game, turn, in_endgame):
        endgame_score = 0
        if in_endgame:
            score += self.evaluate_piece_on_key_squares(game, turn) * 2.0

            score += self.evaluate_king_activity(game, turn) * 1.5
            score += self.evaluate_rook_king_coordination(game, turn) * 1.5

            endgame_score += self.evaluate_pawn_promotion(game, turn) * 2  # Higher weight in endgame
            endgame_score += self.evaluate_king_activity(game, turn) * 1.5
            endgame_score += self.evaluate_rook_king_coordination(game, turn) * 1.5
            endgame_score += self.evaluate_control_of_key_squares(game, turn) * 2
            endgame_score += self.evaluate_zugzwang_potential(game, turn) * 1.5
        else:
            pass
        return endgame_score

    def evaluate_pawn_promotion(self, game, turn):
        score = 0
        is_white = turn == "w"
        pawns = game.board.positions("P" if is_white else "p")

        for pawn in pawns:
            x, y = Game.i2xy(pawn)
            if is_white and y == 6:
                score += 250
            elif (not is_white) and y == 1:
                score += 250

        return score

    def evaluate_king_activity(self, game, turn):
        king_pos = game.find_king(turn)
        if not king_pos:
            return 0
        x, y = Game.i2xy(king_pos)
        # Centralize the king
        distance_to_center = abs(x - 3.5) + abs(y - 3.5)
        activity_score = (7 - distance_to_center) * 20  # Higher when closer to center
        return activity_score

    def evaluate_rook_king_coordination(self, game, turn):
        coordination_score = 0
        rooks = [pos for pos in game.board.positions("R" if turn == "w" else "r")]
        king_pos = game.find_king(turn)
        if not king_pos:
            return coordination_score
        for rook in rooks:
            distance = self.distance(rook, king_pos)
            if distance <= 4:
                coordination_score += 50  # Reward rooks near the king
        return coordination_score

    def evaluate_control_of_key_squares(self, game, turn):
        key_squares = set(
            [
                "d4",
                "d5",
                "e4",
                "e5",
                "c3",
                "c6",
                "f3",
                "f6",
                "c4",
                "f4",
                "c5",
                "f5",
            ]
        )  # Expanded for aggressive bottlenecking
        control_score = 0
        for square in key_squares:
            pos = Game.xy2i(square)
            piece = game.board.get_piece(pos)
            if (turn == "w" and piece.isupper()) or (turn == "b" and piece.islower()):
                control_score += 75  # Significant bonus for controlling key squares
        return control_score

    def distance(self, pos1, pos2):
        x1, y1 = Game.i2xy(pos1)
        x2, y2 = Game.i2xy(pos2)
        return abs(x1 - x2) + abs(y1 - y2)

    def calculate_material(self, game, turn):
        material = {"minor_pieces": 0, "rooks": 0, "queens": 0}
        for piece in game.board.pieces:
            if piece.isupper() if turn == "w" else piece.islower():
                if piece.lower() in ["n", "b"]:
                    material["minor_pieces"] += 1
                elif piece.lower() == "r":
                    material["rooks"] += 1
                elif piece.lower() == "q":
                    material["queens"] += 1
        return material

    def evaluate_king_and_pawns_endgame(self, game, turn):
        endgame_score = 0
        king_pos = game.find_king(turn)
        if king_pos:
            x, y = Game.i2xy(king_pos)
            distance_to_center = abs(x - 3.5) + abs(y - 3.5)
            endgame_score += (7 - distance_to_center) * 10  # Closer to center is better
        pawns = [pos for pos in game.board.positions("P" if turn == "w" else "p")]
        for pawn in pawns:
            x, y = Game.i2xy(pawn)
            advancement = y if turn == "w" else 7 - y
            endgame_score += advancement * 10  # Encourage advancing pawns
        return endgame_score

    def evaluate_rook_endgame(self, game, turn):
        endgame_score = 0
        rooks = [pos for pos in game.board.positions("R" if turn == "w" else "r")]
        for rook in rooks:
            x, y = Game.i2xy(rook)
            file = chr(x + ord("a"))
            is_open = all(game.board.get_piece(Game.xy2i(f"{file}{rank}")) == " " for rank in range(1, 9))
            if is_open:
                endgame_score += 50  # Bonus for rook on open file
            distance_to_center = abs(x - 3.5) + abs(y - 3.5)
            endgame_score += (7 - distance_to_center) * 5  # Closer to center is better
        endgame_score += self.evaluate_pawn_structure(game, turn, in_endgame=True)
        return endgame_score

    def evaluate_game_state(self, game, maximizing_player):
        turn = self.get_game_turn(game)
        turn = self.get_game_turn(game)
        in_endgame = self.is_endgame(game, turn)
        score = 0

        for piece in game.board.pieces:
            score += self.get_piece_value(piece, turn)

        king_safety = self.evaluate_king_safety(game, turn)
        score += king_safety * 2.5

        pawn_structure = self.evaluate_pawn_structure(game, turn, in_endgame)
        score += pawn_structure * (1.5 if in_endgame else 1.0)

        positional = self.evaluate_positional_factors(game, turn, in_endgame)
        score += positional * (2.0 if in_endgame else 1.5)

        endgame = self.evaluate_endgame(game, turn, in_endgame)
        score += endgame * (3.0 if in_endgame else 1.0)

        bottleneck = self.evaluate_bottleneck(game, turn)
        score += bottleneck * (4.0 if in_endgame else 2.0)

        aggressive_defense = self.evaluate_aggressive_defense(game, turn, in_endgame)
        score += aggressive_defense * (2.0 if in_endgame else 1.5)

        strategic_planning = self.evaluate_strategic_planning(game, turn)
        score += strategic_planning * (4.0 if in_endgame else 2.0)

        king_activity_score = self.evaluate_king_activity(game, turn)
        score += king_activity_score * (3.0 if in_endgame else 1.0)

        rook_king_coord_score = self.evaluate_rook_king_coordination(game, turn)
        score += rook_king_coord_score * (3.0 if in_endgame else 1.0)

        king_pawn_coord_score = self.evaluate_king_pawn_coordination(game, turn)
        score += king_pawn_coord_score * (3.0 if in_endgame else 1.0)

        # Add multi-piece coordination to static eval
        multi_piece_coord = self.evaluate_multi_piece_coordination(game, turn)
        score += multi_piece_coord * (3.0 if in_endgame else 1.0)

        if in_endgame:
            king_pawns_endgame_score = self.evaluate_king_and_pawns_endgame(game, turn)
            score += king_pawns_endgame_score * 3.0

            rook_endgame_score = self.evaluate_rook_endgame(game, turn)
            score += rook_endgame_score * 2.0

        checkmate_coordination = self.evaluate_checkmate_coordination(game, turn)
        score += checkmate_coordination * (35.0 if in_endgame else 30.0)

        checkmate_patterns_score = self.evaluate_checkmate_patterns(game, turn)
        score += checkmate_patterns_score * (5.0 if in_endgame else 2.0)

        endgame_checkmate_potential_score = self.evaluate_endgame_checkmate_potential(game, turn)
        score += endgame_checkmate_potential_score * (6.0 if in_endgame else 1.5)

        if in_endgame:
            score += self.calculate_checkmate_acceleration_bonus(game, turn) * 15.0

        return score

    def select_strategic_move(self, board_fen, look_ahead_depth=24):
        test_game = Game(board_fen)
        turn = self.get_game_turn(test_game)  # 'w' or 'b'

        if board_fen in self.opening_moves:
            return random.choice(self.opening_moves[board_fen])

        moves = list(test_game.get_moves())

        if not moves:
            if turn == "w":
                return "e2e4"
            else:
                return "e7e5"

        try:
            predicted_moves = self.predict_opponent_moves(test_game)
            if predicted_moves:
                move_evaluations = [
                    (
                        self.evaluate_move(test_game, move_info["initial_move"]),
                        move_info["initial_move"],
                        move_info["probability"],
                        move_info["strategic_score"],
                    )
                    for move_info in predicted_moves[:10]
                ]
                move_evaluations.sort(key=lambda x: (x[0], x[2], x[3]), reverse=True)
                return move_evaluations[0][1]
        except Exception:
            pass

        evaluated_moves = [(self.evaluate_move(test_game, move), move) for move in moves]
        evaluated_moves.sort(reverse=True, key=lambda x: x[0])
        if evaluated_moves[0][1] is None:
            print("Agent is returning an invalid move!")
        return evaluated_moves[0][1]

    def evaluate_piece_coordination(self, game, turn):
        score = 0
        pieces = (game.board.positions("N") + game.board.positions("B") + game.board.positions("R") + game.board.positions("Q") + game.board.positions("K")) if turn == "w" else (game.board.positions("n") + game.board.positions("b") + game.board.positions("r") + game.board.positions("q") + game.board.positions("k"))

        for piece in pieces:
            attacks = game.get_piece_attacks(game.board.get_piece(piece), piece)
            for target in attacks:
                target_piece = game.board.get_piece(target)
                if target_piece != " " and (target_piece.islower() if turn == "w" else target_piece.isupper()):
                    # Reward attacks and mutual support
                    score += 20  # Base reward
                    if self.is_attacked_by_ally(game, target, turn):
                        score += 90  # Extra for coordinated attacks
                    if self.is_defended(game, target, turn):
                        score += 50  # Reward supported squares
        return score

    def is_attacked_by_ally(self, game, pos, turn):
        for piece_type in ["P", "N", "B", "R", "Q", "K"] if turn == "w" else ["p", "n", "b", "r", "q", "k"]:
            for piece in game.board.positions(piece_type):
                if pos in game.get_piece_attacks(game.board.get_piece(piece), piece):
                    return True
        return False

    def predict_opponent_moves(self, game):
        cache_key = (game.get_fen(), base_depth)
        if cache_key in self.move_cache:
            return self.move_cache[cache_key]

        probable_move_sequences = []

        opponent_moves = game.get_moves()
        base_depth = 8 + len(opponent_moves) // 10

        for move in opponent_moves:
            hypothetical_game = Game(game.get_fen())
            hypothetical_game.apply_move(move)

            move_sequence = self.explore_move_sequences(
                hypothetical_game,
                depth=base_depth - 1,
                is_maximizing=False,
            )

            move_probability = self.calculate_move_probability(move, hypothetical_game)
            move_sequence_score = self.evaluate_move_sequence(move_sequence)

            probable_move_sequences.append(
                {
                    "initial_move": move,
                    "sequence": move_sequence,
                    "probability": move_probability,
                    "strategic_score": move_sequence_score,
                }
            )

        probable_move_sequences.sort(
            key=lambda x: (x["strategic_score"] * 0.7 + x["probability"] * 0.3),
            reverse=True,
        )

        self.move_cache[cache_key] = probable_move_sequences

        return probable_move_sequences

    def calculate_move_probability(self, move, game):
        game_stage = self.determine_game_stage(game)

        # Dynamic weighting of factors based on game stage
        if game_stage == "opening":
            weights = {
                "tactical": 1.2,
                "threats": 1.0,
                "positional": 1.5,
            }
        elif game_stage == "midgame":
            weights = {
                "tactical": 1.5,
                "threats": 1.5,
                "positional": 1.2,
            }
        elif game_stage == "endgame":
            weights = {
                "tactical": 1.0,
                "threats": 1.3,
                "positional": 1.8,
            }
        else:
            weights = {
                "tactical": 1.0,
                "threats": 1.0,
                "positional": 1.0,
            }

        # Evaluate contributing factors for move probability
        tactical_score = weights["tactical"] * self.evaluate_tactical_patterns(game, game.turn)
        threat_score = weights["threats"] * self.evaluate_threats(game, game.turn)
        positional_score = weights["positional"] * self.evaluate_positional_factors(game, game.turn, False)

        # Combine scores with normalized weighting
        total_score = tactical_score + threat_score + positional_score
        normalized_probability = total_score / sum(weights.values())

        return normalized_probability

    def evaluate_move_sequence(self, move_sequence):
        if not move_sequence:
            return 0

        strategic_score = sum(move["evaluation"] * (0.9**idx) for idx, move in enumerate(move_sequence))

        return strategic_score

    def explore_move_sequences(self, game, depth, is_maximizing, alpha=-math.inf, beta=math.inf):
        cache_key = (game.get_fen(), depth, is_maximizing)
        if cache_key in self.move_sequence_cache:
            return self.move_sequence_cache[cache_key]

        if depth == 0 or game.status in [game.CHECKMATE, game.DRAW]:
            return []

        moves = game.get_moves()

        moves.sort(
            key=lambda m: self.evaluate_move(game, m),
            reverse=is_maximizing,
        )

        move_sequences = []
        for move in moves:
            hypothetical_game = Game(game.get_fen())
            hypothetical_game.apply_move(move)

            sub_sequences = self.explore_move_sequences(
                hypothetical_game,
                depth - 1,
                not is_maximizing,
                alpha,
                beta,
            )

            move_evaluation = self.evaluate_move(game, move)

            move_sequence_entry = {
                "move": move,
                "evaluation": move_evaluation,
                "sub_sequences": sub_sequences,
            }
            move_sequences.append(move_sequence_entry)

            if is_maximizing:
                alpha = max(alpha, move_evaluation)
            else:
                beta = min(beta, move_evaluation)
            if beta <= alpha:
                break  # Prune remaining branches

        self.move_sequence_cache[cache_key] = move_sequences
        return move_sequences

    def evaluate_multi_piece_coordination(self, game, turn):
        score = 0
        opponent_king = game.find_king("b" if turn == "w" else "w")
        if not opponent_king:
            return score  # No king found, skip evaluation

        attacking_pieces = []
        for piece_type in ["N", "B", "R", "Q", "K"] if turn == "w" else ["n", "b", "r", "q", "k"]:
            for pos in game.board.positions(piece_type):
                if opponent_king in game.get_piece_attacks(game.board.get_piece(pos), pos):
                    attacking_pieces.append(pos)

        num_attackers = len(attacking_pieces)
        if num_attackers >= 2:
            score += num_attackers * 300  # Reward for each attacking piece
            score += num_attackers * (num_attackers - 1) * 100  # Extra synergy reward for multiple attackers

        restricted_squares = len(self.get_surrounding_squares(game, opponent_king)) - len([sq for sq in self.get_surrounding_squares(game, opponent_king) if game.board.get_piece(sq) == " "])
        score += restricted_squares * 50  # Reward for limiting king movement

        pawns_supporting = self.get_pawn_support(game, opponent_king, turn)
        score += len(pawns_supporting) * 100  # Reward pawns controlling key squares around the king

        return score

    def get_pawn_support(self, game, king_pos, turn):
        pawns = game.board.positions("P" if turn == "w" else "p")
        supporting_pawns = []
        for pawn in pawns:
            if king_pos in game.get_piece_attacks(game.board.get_piece(pawn), pawn):
                supporting_pawns.append(pawn)
        return supporting_pawns

    def evaluate_checkmate_patterns(self, game, turn):
        score = 0
        opponent_king = game.find_king("b" if turn == "w" else "w")
        if not opponent_king:
            return score

        if self.is_back_rank_mate_possible(game, opponent_king, turn):
            score += 500

        if self.is_smothered_mate_possible(game, opponent_king, turn):
            score += 500

        score += self.evaluate_multi_piece_coordination(game, turn)

        return score

    def is_back_rank_mate_possible(self, game, king_pos, turn):
        x, y = Game.i2xy(king_pos)
        if turn == "w" and y == 0:  # Opponent (black) king on black's back rank
            return True
        elif turn == "b" and y == 7:  # Opponent (white) king on white's back rank
            return True
        return False

    def is_smothered_mate_possible(self, game, king_pos, turn):
        opponent = "b" if turn == "w" else "w"
        if not game.is_in_check(opponent):
            return False

        attackers = game.get_attackers(king_pos, turn)
        if len(attackers) != 1:
            return False

        attacker_pos = attackers[0]
        attacker_piece = game.board.get_piece(attacker_pos)
        if attacker_piece.lower() != "n":
            return False

        surrounding = self.get_surrounding_squares(game, king_pos)
        for sq in surrounding:
            piece = game.board.get_piece(sq)
            if piece == " " or (piece.isupper() if opponent == "w" else piece.islower()):
                return False

        return True

    def evaluate_key_square_control(self, game, piece_pos, turn):
        score = 0
        key_squares = self.get_key_squares(game, turn)
        piece_attacks = game.get_piece_attacks(game.board.get_piece(piece_pos), piece_pos)

        for square in key_squares:
            if square in piece_attacks:
                score += 50  # Reward for controlling a key square

                if self.is_attacked_by_ally(game, square, turn):
                    score += 30

                if not self.is_defended(game, square, turn):
                    score -= 20

        return score

    def evaluate_piece_on_key_squares(self, game, turn):
        high_rank_pieces = ["Q", "R", "B", "N"] if turn == "w" else ["q", "r", "b", "n"]
        score = 0

        for piece_type in high_rank_pieces:
            for pos in game.board.positions(piece_type):
                score += self.evaluate_key_square_control(game, pos, turn)

        return score

    def get_key_squares(self, game, turn):
        if self.is_endgame(game, turn):
            ranks = [6, 7] if turn == "w" else [0, 1]
            key_positions = [Game.xy2i(f"{chr(x + ord('a'))}{rank + 1}") for x in range(8) for rank in ranks]
        else:
            central_squares = [
                "d4",
                "d5",
                "e4",
                "e5",
                "c3",
                "c6",
                "f3",
                "f6",
            ]
            key_positions = [Game.xy2i(sq) for sq in central_squares]

        return key_positions

    def evaluate_endgame_checkmate_potential(self, game, turn):
        score = 0
        opponent_king = game.find_king("b" if turn == "w" else "w")
        if not opponent_king:
            return score

        restricted_squares = len(self.get_surrounding_squares(game, opponent_king)) - len([sq for sq in self.get_surrounding_squares(game, opponent_king) if game.board.get_piece(sq) == " "])
        score += restricted_squares * 50

        if self.is_ladder_mate_possible(game, opponent_king, turn):
            score += 500

        if self.is_rook_king_endgame(game, opponent_king, turn):
            score += 300

        return score

    def evaluate_king_pawn_coordination(self, game, turn):
        score = 0
        pawns = game.board.positions("P" if turn == "w" else "p")
        king_pos = game.find_king(turn)
        for pawn in pawns:
            distance = self.distance(king_pos, pawn)
            if distance <= 2:
                score += 50  # Reward proximity to pawn
        return score

    def evaluate_checkmate_coordination(self, game, turn):
        score = 0
        opponent = "b" if turn == "w" else "w"
        opponent_king = game.find_king(opponent)
        if not opponent_king:
            return score  # No king found, skip evaluation

        king_attacks = 0
        key_supporting_pieces = 0
        key_squares_near_king = self.get_surrounding_squares(game, opponent_king)

        for square in key_squares_near_king:
            if self.is_attacked_by_ally(game, square, turn):
                king_attacks += 1
            if self.is_defended(game, square, turn):
                key_supporting_pieces += 1

        score += king_attacks * 200  # Attacks around king
        score += key_supporting_pieces * 100  # Support near king

        if game.is_in_check(opponent):
            score += 500

        return score

    def get_piece_moves(self, game, position):
        board = game.board
        piece = board[position]
        if piece == ".":  # No piece at the position
            return []

        deltas = self.get_deltas_for_piece(piece)
        valid_moves = []
        x, y = Game.i2xy(position)

        for dx, dy in deltas:
            nx, ny = x + dx, y + dy

            if not (0 <= nx < 8 and 0 <= ny < 8):  # Ensure move is on the board
                continue

            target_pos = Game.xy2i(nx, ny)
            target_piece = board[target_pos]

            if piece.upper() == "P":  # Special handling for pawns
                if dx == 0 and target_piece == ".":
                    valid_moves.append(target_pos)
                elif dx != 0 and target_piece != "." and target_piece.islower() != piece.islower():
                    valid_moves.append(target_pos)
            elif piece.upper() == "K":  # Exclude moves into check
                if target_piece == "." or target_piece.islower() != piece.islower():
                    valid_moves.append(target_pos)
            else:  # General rule for other pieces
                if target_piece == ".":
                    valid_moves.append(target_pos)
                elif target_piece.islower() != piece.islower():
                    valid_moves.append(target_pos)
                    break
                else:
                    break

        return valid_moves

    def get_deltas_for_piece(self, piece):
        is_white = piece.isupper()
        piece = piece.upper()

        if piece == "P":  # Pawn
            return [(0, 1), (-1, 1), (1, 1)] if is_white else [(0, -1), (-1, -1), (1, -1)]
        elif piece == "R":  # Rook
            return [(dx, 0) for dx in range(-7, 8) if dx != 0] + [(0, dy) for dy in range(-7, 8) if dy != 0]
        elif piece == "N":  # Knight
            return [
                (2, 1),
                (1, 2),
                (-1, 2),
                (-2, 1),
                (-2, -1),
                (-1, -2),
                (1, -2),
                (2, -1),
            ]
        elif piece == "B":  # Bishop
            return [(i, i) for i in range(-7, 8) if i != 0] + [(i, -i) for i in range(-7, 8) if i != 0]
        elif piece == "Q":  # Queen
            return self.get_deltas_for_piece("R") + self.get_deltas_for_piece("B")
        elif piece == "K":  # King
            return [
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
                (1, 1),
                (-1, 1),
                (1, -1),
                (-1, -1),
            ]
        else:
            raise ValueError(f"Unknown piece type: {piece}")

    def determine_game_stage(self, game):
        from collections import Counter

        def parse_fen(fen):
            board_part = fen.split()[0]
            counts = Counter(board_part.replace("/", ""))
            return counts

        def calculate_material_value(material_counts):
            piece_values = {
                "P": 1,
                "N": 3,
                "B": 3,
                "R": 5,
                "Q": 9,
                "p": 1,
                "n": 3,
                "b": 3,
                "r": 5,
                "q": 9,
            }
            total = sum(piece_values.get(piece, 0) * count for piece, count in material_counts.items())
            return total

        def calculate_mobility(game, color):
            moves = game.get_moves()
            if color == "white":
                return len([move for move in moves if move[0].isupper()])
            elif color == "black":
                return len([move for move in moves if move[0].islower()])
            return 0

        fen = game.get_fen()
        material_counts = parse_fen(fen)

        total_material = calculate_material_value(material_counts)

        white_mobility = calculate_mobility(game, "white")
        black_mobility = calculate_mobility(game, "black")

        if total_material >= 30 and (white_mobility > 20 or black_mobility > 20):
            return "opening"
        elif 15 < total_material < 30:
            return "midgame"
        elif total_material <= 15:
            return "endgame"
        else:
            return "unknown"

    def is_ladder_mate_possible(self, game, king_pos, turn):
        opponent = "b" if turn == "w" else "w"
        x, y = Game.i2xy(king_pos)

        if turn == "w" and y != 0:
            return False
        if turn == "b" and y != 7:
            return False

        friendly_heavy = (game.board.positions("R") + game.board.positions("Q")) if turn == "w" else (game.board.positions("r") + game.board.positions("q"))
        if len(friendly_heavy) < 2:
            return False

        surrounding = self.get_surrounding_squares(game, king_pos)
        for sq in surrounding:
            piece = game.board.get_piece(sq)
            if piece == " " or (piece.isupper() if opponent == "w" else piece.islower()):
                return False

        return True

    def is_rook_king_endgame(self, game, king_pos, turn):
        opponent = "b" if turn == "w" else "w"

        opp_pieces = [p for p in game.board.pieces if (p.isupper() if opponent == "w" else p.islower()) and p.lower() != "k"]
        if len(opp_pieces) > 0:
            return False

        if turn == "w":
            rooks = game.board.positions("R")
        else:
            rooks = game.board.positions("r")
        if len(rooks) == 0:
            return False

        x, y = Game.i2xy(king_pos)
        if x not in [0, 7] and y not in [0, 7]:
            return False

        return True

    def evaluate_material_advantage(self, game, move):
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0,  # The king's value is infinite in the game but zero for evaluation.
        }

        # Apply the move to a temporary game state.
        test_game = Game(game.get_fen())  # Clone the game state.
        test_game.apply_move(move)

        material_score_white = 0.0
        material_score_black = 0.0

        for square in chess.SQUARES:
            piece = test_game.board.piece_at(square)
            if piece:
                piece_value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    material_score_white += piece_value
                else:
                    material_score_black += piece_value

        turn = game.turn  # Get the player to move (True for white, False for black).
        if turn == chess.WHITE:
            return material_score_white - material_score_black
        else:
            return material_score_black - material_score_white

    def evaluate_positional_advantage(self, game, move):
        center_control_bonus = 0.1  # Reward for controlling central squares (e4, d4, e5, d5).
        mobility_factor = 0.05  # Reward per legal move available.
        pawn_structure_bonus = 0.2  # Bonus for advanced pawns.
        doubled_pawn_penalty = -0.2  # Penalty for doubled pawns.
        isolated_pawn_penalty = -0.3  # Penalty for isolated pawns.
        king_safety_penalty = -0.5  # Penalty for exposing the king.

        test_game = Game(game.get_fen())
        test_game.apply_move(move)

        board = test_game.board
        turn = test_game.turn  # True for white, False for black.

        central_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        positional_score_white = 0.0
        positional_score_black = 0.0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue

            piece_value = 0.0
            color = piece.color

            if square in central_squares:
                piece_value += center_control_bonus

            piece_moves = board.legal_moves
            mobility_score = mobility_factor * len([m for m in piece_moves if m.from_square == square])
            piece_value += mobility_score

            if piece.piece_type == chess.PAWN:
                if (color == chess.WHITE and chess.square_rank(square) > 4) or (color == chess.BLACK and chess.square_rank(square) < 3):
                    piece_value += pawn_structure_bonus

                file_pawns = [sq for sq in chess.SQUARES if chess.square_file(sq) == chess.square_file(square)]
                if len([sq for sq in file_pawns if board.piece_at(sq) and board.piece_at(sq).color == color]) > 1:
                    piece_value += doubled_pawn_penalty

                adjacent_files = [
                    chess.square_file(square) - 1,
                    chess.square_file(square) + 1,
                ]
                has_support = False
                for adj_file in adjacent_files:
                    if 0 <= adj_file < 8:
                        support_pawns = [sq for sq in chess.SQUARES if chess.square_file(sq) == adj_file and board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN and board.piece_at(sq).color == color]
                        if support_pawns:
                            has_support = True
                if not has_support:
                    piece_value += isolated_pawn_penalty

            if piece.piece_type == chess.KING:
                king_rank = chess.square_rank(square)
                king_file = chess.square_file(square)
                exposed = True
                for dr in [-1, 0, 1]:
                    for df in [-1, 0, 1]:
                        if dr == 0 and df == 0:
                            continue
                        adj_square = chess.square(king_file + df, king_rank + dr) if 0 <= king_file + df < 8 and 0 <= king_rank + dr < 8 else None
                        if adj_square and board.piece_at(adj_square) and board.piece_at(adj_square).color == color and board.piece_at(adj_square).piece_type == chess.PAWN:
                            exposed = False
                if exposed:
                    piece_value += king_safety_penalty

            if color == chess.WHITE:
                positional_score_white += piece_value
            else:
                positional_score_black += piece_value

        if turn == chess.WHITE:
            return positional_score_white - positional_score_black
        else:
            return positional_score_black - positional_score_white


def chess_bot(obs):
    agent = StrategicChessAgent(max_search_depth=32)
    return agent.select_strategic_move(obs.board)
