import time
import math
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum, auto
import random


class OpeningBook:
    def __init__(self, max_positions: int = 10000):
        self.max_positions = max_positions
        self.book = {
            # Starting position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": [
                ((4, 1), (4, 3)),  # 1.e4
                ((3, 1), (3, 3)),  # 1.d4
                ((6, 0), (5, 2)),  # 1.Nf3
                ((2, 1), (2, 3)),  # 1.c4
            ],
            # 1.e4 responses
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3": [
                ((4, 6), (4, 4)),  # 1...e5
                ((2, 7), (2, 5)),  # 1...c5 (Sicilian)
                ((4, 6), (4, 5)),  # 1...e6 (French)
                ((2, 7), (2, 6)),  # 1...c6 (Caro-Kann)
            ],
            # Ruy Lopez line
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6": [
                ((6, 0), (5, 2)),  # 2.Nf3
            ],
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -": [
                ((1, 7), (2, 5)),  # 2...Nc6
            ],
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -": [
                ((5, 0), (1, 4)),  # 3.Bb5 (Ruy Lopez)
            ],
            # Italian Game line
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -": [
                ((5, 0), (2, 3)),  # 3.Bc4 (Italian Game)
            ],
            # Sicilian Defense lines
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6": [
                ((6, 0), (5, 2)),  # 2.Nf3 (Open Sicilian)
                ((1, 1), (1, 3)),  # 2.b4 (Wing Gambit)
                ((2, 1), (2, 3)),  # 2.c3 (Alapin)
            ],
            # Sicilian Najdorf
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -": [
                ((3, 6), (3, 5)),  # 2...d6
            ],
            "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -": [
                ((3, 1), (3, 3)),  # 3.d4
            ],
            # French Defense lines
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": [
                ((3, 1), (3, 3)),  # 2.d4
                ((1, 1), (1, 3)),  # 2.b3 (King's Indian Attack)
            ],
            # Caro-Kann lines
            "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": [
                ((3, 1), (3, 3)),  # 2.d4
                ((1, 0), (2, 2)),  # 2.Nc3
            ],
            # 1.d4 responses
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3": [
                ((6, 7), (5, 5)),  # 1...Nf6 (Indian Defenses)
                ((3, 6), (3, 5)),  # 1...d5 (Queen's Pawn Game)
                ((5, 6), (5, 5)),  # 1...f5 (Dutch Defense)
            ],
            # Queen's Gambit lines
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6": [
                ((2, 1), (2, 3)),  # 2.c4 (Queen's Gambit)
            ],
            # Queen's Gambit Accepted
            "rnbqkbnr/ppp1pppp/8/8/2pP4/8/PPP1PPPP/RNBQKBNR w KQkq -": [
                ((4, 1), (4, 3)),  # 3.e4
                ((6, 0), (5, 2)),  # 3.Nf3
            ],
            # Queen's Gambit Declined
            "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq -": [
                ((1, 0), (2, 2)),  # 3.Nc3
                ((6, 0), (5, 2)),  # 3.Nf3
            ],
            # King's Indian Defense
            "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -": [
                ((2, 1), (2, 3)),  # 2.c4
            ],
            "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq -": [
                ((1, 0), (2, 2)),  # 3.Nc3
            ],
            # Nimzo-Indian Defense
            "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq -": [
                ((4, 1), (4, 3)),  # 4.e4
                ((5, 2), (4, 3)),  # 4.Nf3
                ((2, 1), (2, 2)),  # 4.Qc2
            ],
            # English Opening lines
            "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3": [
                ((4, 6), (4, 5)),  # 1...e6
                ((6, 7), (5, 5)),  # 1...Nf6
                ((2, 6), (2, 5)),  # 1...c5
            ],
            # Reti Opening lines
            "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq -": [
                ((3, 6), (3, 5)),  # 1...d5
                ((6, 7), (5, 5)),  # 1...Nf6
            ],
            # Modern Defense
            "rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": [
                ((3, 1), (3, 3)),  # 2.d4
                ((3, 1), (3, 4)),  # 2.d4
            ],
            # Pirc Defense
            "rnbqkbnr/ppp1pppp/3p4/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": [
                ((3, 1), (3, 3)),  # 2.d4
                ((1, 0), (2, 2)),  # 2.Nc3
            ],
            # Scandinavian Defense
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6": [
                ((4, 3), (3, 4)),  # 2.exd5
            ],
            # Alekhine's Defense
            "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": [
                ((4, 3), (4, 4)),  # 2.e5
            ],
            # Dragon Sicilian
            "rnbqkbnr/pp2pp1p/3p2p1/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq -": [
                ((1, 0), (2, 2)),  # 4.Nc3
            ],
            # Semi-Slav Defense
            "rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq -": [
                ((4, 1), (4, 3)),  # 4.e3
                ((2, 3), (3, 4)),  # 4.cxd5
            ],
        }

    def get_book_move(self, position_fen: str) -> Optional[Tuple[int, int]]:
        """Get a move from the opening book, with some randomization for variety"""
        if len(self.book) >= self.max_positions:
            return None

        moves = self.book.get(position_fen, [])
        if moves:
            # Add some randomization - don't always pick the first move

            return random.choice(moves)
        return None

    def add_position(self, fen: str, move: Tuple[int, int]) -> None:
        """Add a new position to the opening book"""
        if len(self.book) < self.max_positions:
            if fen in self.book:
                if move not in self.book[fen]:
                    self.book[fen].append(move)
            else:
                self.book[fen] = [move]

    def get_variation_name(self, fen: str, move: Tuple[int, int]) -> str:
        """Get the name of the opening variation if known"""
        opening_names = {
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR": {
                ((4, 6), (4, 4)): "King's Pawn (1...e5)",
                ((2, 7), (2, 5)): "Sicilian Defense",
                ((4, 6), (4, 5)): "French Defense",
                ((2, 7), (2, 6)): "Caro-Kann Defense",
            },
            # Add more opening names as needed...
        }

        if fen in opening_names and move in opening_names[fen]:
            return opening_names[fen][move]
        return "Unknown Opening"


class StrategicGoal(Enum):
    ATTACK = auto()
    DEFENSE = auto()
    PIECE_DEVELOPMENT = auto()
    POSITIONAL_PLAY = auto()
    TACTICAL_OPPORTUNITY = auto()


class MaterialBalance(Enum):
    EQUAL = auto()
    SLIGHT_ADVANTAGE = auto()
    SIGNIFICANT_ADVANTAGE = auto()
    DISADVANTAGE = auto()


class PositionalEvaluation:
    def __init__(
        self,
        king_safety: float = 0.0,
        piece_activity: float = 0.0,
        pawn_structure: float = 0.0,
        center_control: float = 0.0,
    ):
        self.king_safety = king_safety
        self.piece_activity = piece_activity
        self.pawn_structure = pawn_structure
        self.center_control = center_control

    def get_overall_score(self) -> float:
        """Calculate an overall positional score"""
        return (
            sum(
                [
                    self.king_safety * 1.5,
                    self.piece_activity * 1.2,
                    self.pawn_structure,
                    self.center_control * 1.3,
                ]
            )
            / 4
        )


class MiddlegameBook:
    def __init__(self, max_positions: int = 10000):
        self.max_positions = max_positions
        self.book = self._initialize_comprehensive_strategies()

    def _initialize_comprehensive_strategies(self) -> Dict:
        return {
            # Comprehensive strategic patterns
            "Strategic Patterns": {
                # Key configurations with detailed strategic recommendations
                "Closed Center": {
                    "typical_fens": [
                        "r1bq1rk1/pp2ppbp/2np1np1/3p4/2PP4/2N2NP1/PP2PP1P/R1BQ1RK1 w - -",
                        "r1bq1rk1/ppp1ppbp/2np1np1/3p2B1/2PP4/2N2NP1/PP2PP1P/R2Q1RK1 w - -",
                    ],
                    "strategic_goals": [
                        StrategicGoal.POSITIONAL_PLAY,
                        StrategicGoal.PIECE_DEVELOPMENT,
                    ],
                    "recommended_moves": [
                        {
                            "move": ((5, 0), (7, 2)),
                            "score": 0.8,
                            "description": "Reposition knight to pressure kingside",
                        },
                        {
                            "move": ((4, 0), (6, 2)),
                            "score": 0.7,
                            "description": "Prepare queen for potential kingside attack",
                        },
                        {
                            "move": ((2, 1), (2, 3)),
                            "score": 0.6,
                            "description": "Create pawn tension, control central squares",
                        },
                    ],
                },
                "Open Center": {
                    "typical_fens": [
                        "r1bq1rk1/pp2ppbp/2n2np1/3p4/2PP4/2N2NP1/PP2PP1P/R1BQ1RK1 w - -",
                        "r1bq1rk1/pp2ppbp/2np1n2/3p4/2PP4/2N2NP1/PP2PP1P/R1BQ1RK1 w - -",
                    ],
                    "strategic_goals": [
                        StrategicGoal.TACTICAL_OPPORTUNITY,
                        StrategicGoal.ATTACK,
                    ],
                    "recommended_moves": [
                        {
                            "move": ((4, 1), (4, 3)),
                            "score": 0.9,
                            "description": "Create central tension, open lines",
                        },
                        {
                            "move": ((5, 0), (3, 2)),
                            "score": 0.8,
                            "description": "Develop bishop to active square",
                        },
                        {
                            "move": ((1, 0), (2, 2)),
                            "score": 0.7,
                            "description": "Improve knight placement",
                        },
                    ],
                },
            },
            # Advanced tactical themes
            "Tactical Themes": {
                "Pin Exploitation": {
                    "typical_fens": [
                        "r1bq1rk1/pp2ppbp/2np1np1/3p4/2PP4/2N2NP1/PP2PP1P/R1BQ1RK1 w - -"
                    ],
                    "strategic_goals": [StrategicGoal.TACTICAL_OPPORTUNITY],
                    "recommended_moves": [
                        {
                            "move": ((5, 0), (7, 5)),
                            "score": 0.9,
                            "description": "Create pin on opponent's knight",
                        },
                        {
                            "move": ((4, 0), (4, 1)),
                            "score": 0.8,
                            "description": "Prepare discovered attack",
                        },
                    ],
                },
                "Piece Sacrifice": {
                    "typical_fens": [
                        "r1bq1rk1/pp2ppbp/2np1np1/3p4/2PP4/2N2NP1/PP2PP1P/R1BQ1RK1 w - -"
                    ],
                    "strategic_goals": [
                        StrategicGoal.ATTACK,
                        StrategicGoal.TACTICAL_OPPORTUNITY,
                    ],
                    "recommended_moves": [
                        {
                            "move": ((5, 2), (3, 3)),
                            "score": 0.9,
                            "description": "Knight sacrifice to expose king",
                        },
                        {
                            "move": ((4, 1), (4, 2)),
                            "score": 0.8,
                            "description": "Prepare positional sacrifice",
                        },
                    ],
                },
            },
        }

    def evaluate_position(self, fen: str) -> PositionalEvaluation:
        """
        Advanced position evaluation based on multiple strategic factors

        Args:
            fen (str): Board position in FEN notation

        Returns:
            PositionalEvaluation: Detailed positional assessment
        """
        # Simplified evaluation (could be expanded with more complex logic)
        return PositionalEvaluation(
            king_safety=0.7,  # Example values
            piece_activity=0.6,
            pawn_structure=0.5,
            center_control=0.8,
        )

    def get_strategic_recommendation(
        self,
        fen: str,
        goals: List[StrategicGoal] = None,
        material_balance: MaterialBalance = MaterialBalance.EQUAL,
    ) -> Dict:
        """
        Get sophisticated strategic recommendations

        Args:
            fen (str): Current board position
            goals (List[StrategicGoal]): Specific strategic objectives
            material_balance (MaterialBalance): Current material situation

        Returns:
            Dict: Comprehensive strategic recommendation
        """
        # Evaluate current position
        position_eval = self.evaluate_position(fen)

        # Select matching strategy
        matching_strategies = []
        for category in self.book.values():
            for strategy_name, strategy_details in category.items():
                # Check FEN pattern match
                if any(
                    fen.startswith(pattern)
                    for pattern in strategy_details.get("typical_fens", [])
                ):
                    # Goal matching
                    if goals:
                        if any(
                            goal in strategy_details.get("strategic_goals", [])
                            for goal in goals
                        ):
                            matching_strategies.append(
                                (strategy_name, strategy_details)
                            )
                    else:
                        matching_strategies.append((strategy_name, strategy_details))

        if not matching_strategies:
            return None

        # Select best strategy
        best_strategy = max(
            matching_strategies,
            key=lambda x: max(
                move["score"] for move in x[1].get("recommended_moves", [])
            ),
        )

        # Select move considering various factors
        recommended_moves = best_strategy[1]["recommended_moves"]
        recommended_move = max(
            recommended_moves,
            key=lambda move: (
                move["score"] * 0.6
                + position_eval.get_overall_score()  # Move inherent score
                * 0.4  # Position compatibility
            ),
        )

        return {
            "strategy_name": best_strategy[0],
            "recommended_move": recommended_move["move"],
            "move_description": recommended_move["description"],
            "position_evaluation": position_eval,
        }

    def suggest_middlegame_plan(
        self, fen: str, primary_goal: StrategicGoal = StrategicGoal.POSITIONAL_PLAY
    ) -> Dict:
        """
        Generate a comprehensive middlegame strategic plan

        Args:
            fen (str): Current board position
            primary_goal (StrategicGoal): Primary strategic objective

        Returns:
            Dict: Detailed strategic plan
        """
        recommendation = self.get_strategic_recommendation(fen, goals=[primary_goal])

        if not recommendation:
            return {
                "overall_strategy": "Flexible play, focus on piece coordination",
                "key_objectives": [
                    "Improve piece positions",
                    "Maintain king safety",
                    "Control central squares",
                ],
            }

        strategic_descriptions = {
            StrategicGoal.ATTACK: "Aggressive play targeting opponent's weaknesses",
            StrategicGoal.DEFENSE: "Prioritize king safety and piece protection",
            StrategicGoal.PIECE_DEVELOPMENT: "Complete piece development and coordination",
            StrategicGoal.POSITIONAL_PLAY: "Gradually improve piece positions and control key squares",
            StrategicGoal.TACTICAL_OPPORTUNITY: "Look for concrete tactical opportunities and combinations",
        }

        return {
            "strategy_name": recommendation["strategy_name"],
            "overall_strategy": strategic_descriptions.get(
                primary_goal, "Balanced approach"
            ),
            "recommended_move": recommendation["recommended_move"],
            "move_description": recommendation["move_description"],
            "key_objectives": [
                "Control key central squares",
                "Maintain flexible piece placement",
                "Create potential tactical opportunities",
            ],
            "positional_score": recommendation[
                "position_evaluation"
            ].get_overall_score(),
        }


# Optional: Advanced usage example
def demonstrate_middlegame_strategy():
    middlegame_book = MiddlegameBook()

    # Example starting position
    sample_fen = "r1bq1rk1/pp2ppbp/2np1np1/3p4/2PP4/2N2NP1/PP2PP1P/R1BQ1RK1 w - -"

    # Get strategic recommendation
    strategy = middlegame_book.suggest_middlegame_plan(
        sample_fen, primary_goal=StrategicGoal.ATTACK
    )

    print("Middlegame Strategy:")
    for key, value in strategy.items():
        print(f"{key.replace('_', ' ').title()}: {value}")


class EndgameObjective(Enum):
    PAWN_PROMOTION = auto()
    KING_ACTIVATION = auto()
    PIECE_ELIMINATION = auto()
    FORTRESS_DEFENSE = auto()
    ZUGZWANG_CREATION = auto()
    DRAW_PREVENTION = auto()
    CHECKMATE_PURSUIT = auto()


class EndgameMaterialState(Enum):
    BALANCED = auto()
    MATERIAL_ADVANTAGE = auto()
    MATERIAL_DISADVANTAGE = auto()
    CRITICAL_MATERIAL_IMBALANCE = auto()


class EndgameEvaluation:
    def __init__(
        self,
        king_activity: float = 0.0,
        pawn_potential: float = 0.0,
        piece_coordination: float = 0.0,
        theoretical_draw_probability: float = 0.0,
    ):
        self.king_activity = king_activity
        self.pawn_potential = pawn_potential
        self.piece_coordination = piece_coordination
        self.theoretical_draw_probability = theoretical_draw_probability

    def get_winning_potential(self) -> float:
        """Calculate overall endgame winning potential"""
        return (
            sum(
                [
                    self.king_activity * 1.5,
                    self.pawn_potential * 2.0,
                    self.piece_coordination * 1.2,
                    (1 - self.theoretical_draw_probability) * 1.3,
                ]
            )
            / 4
        )


class EndgameBook:
    def __init__(self, max_positions: int = 15000):
        self.max_positions = max_positions
        self.book = self._initialize_comprehensive_endgame_strategies()

    def _initialize_comprehensive_endgame_strategies(self) -> Dict:
        return {
            # Pawn Endgame Strategies
            "Pawn Endgames": {
                "King and Pawn vs King": {
                    "typical_fens": [
                        "8/8/8/8/3k4/8/4P3/4K3 w - -",  # Basic pawn promotion scenario
                        "8/3p4/8/8/3K4/8/8/3k4 w - -",  # Pawn blocking king
                    ],
                    "objectives": [EndgameObjective.PAWN_PROMOTION],
                    "strategies": [
                        {
                            "description": "Opposition and key squares control",
                            "recommended_moves": [
                                {
                                    "move": ((1, 4), (1, 3)),
                                    "score": 0.9,
                                    "tactics": "Gain key square",
                                },
                                {
                                    "move": ((1, 4), (2, 4)),
                                    "score": 0.8,
                                    "tactics": "Create opposition",
                                },
                            ],
                        }
                    ],
                },
                "Passed Pawn Dynamics": {
                    "typical_fens": [
                        "8/3p4/8/2P5/3K4/8/8/3k4 w - -",  # Passed pawn with king support
                        "8/8/3p4/2P5/3K4/8/8/3k4 w - -",  # Pawn advancement scenario
                    ],
                    "objectives": [
                        EndgameObjective.PAWN_PROMOTION,
                        EndgameObjective.KING_ACTIVATION,
                    ],
                    "strategies": [
                        {
                            "description": "Pawn breakthrough and king support",
                            "recommended_moves": [
                                {
                                    "move": ((1, 4), (2, 5)),
                                    "score": 0.9,
                                    "tactics": "Support pawn advance",
                                },
                                {
                                    "move": ((2, 5), (3, 4)),
                                    "score": 0.8,
                                    "tactics": "Create pawn threat",
                                },
                            ],
                        }
                    ],
                },
            },
            # Piece Endgame Strategies
            "Piece Endgames": {
                "Rook Endgames": {
                    "typical_fens": [
                        "8/8/8/3R4/3K4/8/8/3k4 w - -",  # Rook and king coordination
                        "8/8/8/3R4/8/3K4/8/3k4 w - -",  # Rook cutting off enemy king
                    ],
                    "objectives": [
                        EndgameObjective.PIECE_ELIMINATION,
                        EndgameObjective.CHECKMATE_PURSUIT,
                    ],
                    "strategies": [
                        {
                            "description": "Rook cutting off king's movement",
                            "recommended_moves": [
                                {
                                    "move": ((3, 4), (3, 5)),
                                    "score": 0.9,
                                    "tactics": "Restrict king movement",
                                },
                                {
                                    "move": ((3, 4), (2, 4)),
                                    "score": 0.8,
                                    "tactics": "Create mating net",
                                },
                            ],
                        }
                    ],
                },
                "Minor Piece Endgames": {
                    "typical_fens": [
                        "8/8/3N4/8/3K4/8/8/3k4 w - -",  # Knight and king coordination
                        "8/8/3B4/8/3K4/8/8/3k4 w - -",  # Bishop and king coordination
                    ],
                    "objectives": [
                        EndgameObjective.PIECE_ELIMINATION,
                        EndgameObjective.KING_ACTIVATION,
                    ],
                    "strategies": [
                        {
                            "description": "Piece coordination for zugzwang",
                            "recommended_moves": [
                                {
                                    "move": ((3, 5), (2, 4)),
                                    "score": 0.9,
                                    "tactics": "Limit king squares",
                                },
                                {
                                    "move": ((3, 5), (4, 6)),
                                    "score": 0.8,
                                    "tactics": "Create tactical threat",
                                },
                            ],
                        }
                    ],
                },
            },
            # Complex Endgame Scenarios
            "Complex Endgames": {
                "Fortress Defense": {
                    "typical_fens": [
                        "8/8/3p4/2pP4/2PK4/8/8/2k5 w - -",  # Defensive pawn structure
                        "8/8/3p4/2pP4/2PK4/8/8/2k5 b - -",
                    ],
                    "objectives": [
                        EndgameObjective.FORTRESS_DEFENSE,
                        EndgameObjective.DRAW_PREVENTION,
                    ],
                    "strategies": [
                        {
                            "description": "Preventing pawn breakthrough",
                            "recommended_moves": [
                                {
                                    "move": ((1, 3), (2, 4)),
                                    "score": 0.9,
                                    "tactics": "Block pawn advance",
                                },
                                {
                                    "move": ((1, 3), (1, 2)),
                                    "score": 0.8,
                                    "tactics": "Maintain defensive position",
                                },
                            ],
                        }
                    ],
                },
                "Zugzwang Creation": {
                    "typical_fens": [
                        "8/8/3p4/8/3K4/8/8/3k4 w - -",  # Position ripe for zugzwang
                        "8/8/3p4/8/3K4/8/8/3k4 b - -",
                    ],
                    "objectives": [
                        EndgameObjective.ZUGZWANG_CREATION,
                        EndgameObjective.CHECKMATE_PURSUIT,
                    ],
                    "strategies": [
                        {
                            "description": "Forcing unfavorable moves",
                            "recommended_moves": [
                                {
                                    "move": ((1, 3), (2, 3)),
                                    "score": 0.9,
                                    "tactics": "Limit king movement",
                                },
                                {
                                    "move": ((1, 3), (1, 2)),
                                    "score": 0.8,
                                    "tactics": "Create decision pressure",
                                },
                            ],
                        }
                    ],
                },
            },
        }

    def evaluate_endgame_position(self, fen: str) -> EndgameEvaluation:
        """
        Advanced endgame position evaluation

        Args:
            fen (str): Board position in FEN notation

        Returns:
            EndgameEvaluation: Detailed endgame assessment
        """
        # Simplified evaluation (could be expanded with more complex logic)
        return EndgameEvaluation(
            king_activity=0.7,
            pawn_potential=0.6,
            piece_coordination=0.5,
            theoretical_draw_probability=0.3,
        )

    def get_endgame_recommendation(
        self,
        fen: str,
        objectives: List[EndgameObjective] = None,
        material_state: EndgameMaterialState = EndgameMaterialState.BALANCED,
    ) -> Dict:
        """
        Generate sophisticated endgame recommendations

        Args:
            fen (str): Current board position
            objectives (List[EndgameObjective]): Specific endgame goals
            material_state (EndgameMaterialState): Current material situation

        Returns:
            Dict: Comprehensive endgame recommendation
        """
        # Evaluate current position
        position_eval = self.evaluate_endgame_position(fen)

        # Select matching strategies
        matching_strategies = []
        for category in self.book.values():
            for strategy_name, strategy_details in category.items():
                # Check FEN pattern match
                if any(
                    fen.startswith(pattern)
                    for pattern in strategy_details.get("typical_fens", [])
                ):
                    # Objective matching
                    if objectives:
                        if any(
                            obj in strategy_details.get("objectives", [])
                            for obj in objectives
                        ):
                            matching_strategies.append(
                                (strategy_name, strategy_details)
                            )
                    else:
                        matching_strategies.append((strategy_name, strategy_details))

        if not matching_strategies:
            return {
                "strategy": "Flexible endgame play",
                "key_objectives": [
                    "Maintain piece activity",
                    "Control key squares",
                    "Minimize opponent's counterplay",
                ],
            }

        # Select best strategy
        best_strategy = max(
            matching_strategies,
            key=lambda x: max(
                strategy["score"]
                for strategy_group in x[1].get("strategies", [])
                for strategy in strategy_group.get("recommended_moves", [])
            ),
        )

        # Select recommended move
        recommended_strategy = best_strategy[1]["strategies"][0]
        recommended_moves = recommended_strategy["recommended_moves"]
        recommended_move = max(
            recommended_moves,
            key=lambda move: (
                move["score"] * 0.6
                + position_eval.get_winning_potential()  # Move inherent score
                * 0.4  # Position compatibility
            ),
        )

        return {
            "strategy_name": best_strategy[0],
            "strategy_description": recommended_strategy["description"],
            "recommended_move": recommended_move["move"],
            "move_tactics": recommended_move["tactics"],
            "winning_potential": position_eval.get_winning_potential(),
        }

    def suggest_endgame_plan(
        self,
        fen: str,
        primary_objective: EndgameObjective = EndgameObjective.PAWN_PROMOTION,
    ) -> Dict:
        """
        Generate a comprehensive endgame strategic plan

        Args:
            fen (str): Current board position
            primary_objective (EndgameObjective): Primary endgame objective

        Returns:
            Dict: Detailed strategic endgame plan
        """
        recommendation = self.get_endgame_recommendation(
            fen, objectives=[primary_objective]
        )

        objective_descriptions = {
            EndgameObjective.PAWN_PROMOTION: "Advance and promote a pawn to create decisive advantage",
            EndgameObjective.KING_ACTIVATION: "Activate the king as a powerful piece",
            EndgameObjective.PIECE_ELIMINATION: "Strategically exchange pieces to simplify position",
            EndgameObjective.FORTRESS_DEFENSE: "Create impenetrable defensive formation",
            EndgameObjective.ZUGZWANG_CREATION: "Force opponent into unfavorable move",
            EndgameObjective.DRAW_PREVENTION: "Maintain winning chances, avoid draw",
            EndgameObjective.CHECKMATE_PURSUIT: "Systematically restrict and checkmate opponent's king",
        }

        return {
            "strategy_name": recommendation.get("strategy_name", "Flexible Endgame"),
            "overall_objective": objective_descriptions.get(
                primary_objective, "Strategic endgame play"
            ),
            "strategy_description": recommendation.get(
                "strategy_description", "Adaptable approach"
            ),
            "recommended_move": recommendation.get("recommended_move"),
            "move_tactics": recommendation.get("move_tactics"),
            "key_objectives": [
                "Maximize piece activity",
                "Control critical squares",
                "Minimize counterplay opportunities",
            ],
            "winning_potential": recommendation.get("winning_potential", 0.5),
        }


def demonstrate_endgame_strategy():
    endgame_book = EndgameBook()

    # Example endgame position
    sample_fen = "8/8/3p4/2pP4/2PK4/8/8/2k5 w - -"

    # Get endgame strategy
    strategy = endgame_book.suggest_endgame_plan(
        sample_fen, primary_objective=EndgameObjective.FORTRESS_DEFENSE
    )

    print("Endgame Strategy:")
    for key, value in strategy.items():
        print(f"{key.replace('_', ' ').title()}: {value}")


class FENTranslator:
    """
    Efficient FEN translation between internal bitboard representation and FEN string
    """

    PIECE_TO_FEN = {
        "R": "R",
        "N": "N",
        "B": "B",
        "Q": "Q",
        "K": "K",
        "P": "P",
        "r": "r",
        "n": "n",
        "b": "b",
        "q": "q",
        "k": "k",
        "p": "p",
    }

    FEN_TO_PIECE = {v: k for k, v in PIECE_TO_FEN.items()}

    @staticmethod
    def bitboard_to_fen(position: "ChessPosition") -> str:
        """
        Convert bitboard representation to FEN string
        Efficiently translate piece positions using bitwise operations
        """
        # Initialize empty board representation
        board = [" "] * 64

        # White pieces placement
        for piece, (white_bb, _) in position.piece_positions.items():
            while white_bb:
                square = position._get_lsb_index(white_bb)
                board[square] = FENTranslator.PIECE_TO_FEN[piece.upper()]
                white_bb &= white_bb - 1

        # Black pieces placement
        for piece, (_, black_bb) in position.piece_positions.items():
            while black_bb:
                square = position._get_lsb_index(black_bb)
                board[square] = FENTranslator.PIECE_TO_FEN[piece.lower()]
                black_bb &= black_bb - 1

        # Convert board to FEN ranks
        ranks = []
        for rank in range(7, -1, -1):
            rank_squares = board[rank * 8 : (rank + 1) * 8]
            rank_fen = FENTranslator._compress_rank(rank_squares)
            ranks.append(rank_fen)

        # Construct FEN components
        fen_components = [
            "/".join(ranks),
            "w" if position.white_to_move else "b",
            FENTranslator._encode_castling(position.castling_rights),
            FENTranslator._encode_en_passant(position.en_passant) or "-",
        ]

        return " ".join(fen_components)

    @staticmethod
    def fen_to_bitboard(fen: str) -> "ChessPosition":
        """
        Convert FEN string to bitboard representation
        Handles complex FEN parsing with bitwise operations
        """
        position = ChessPosition()

        # Split FEN components
        parts = fen.split()
        board_str, turn, castling, en_passant = parts[0], parts[1], parts[2], parts[3]

        # Reset bitboards
        for piece in position.piece_positions:
            position.piece_positions[piece] = [0, 0]

        # Parse board ranks
        ranks = board_str.split("/")
        for rank_idx, rank in enumerate(reversed(ranks)):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    square = rank_idx * 8 + file_idx
                    piece = char.upper()

                    # Determine color and set appropriate bitboard
                    if piece in position.piece_positions:
                        bb_index = 0 if char.isupper() else 1
                        position.piece_positions[piece][bb_index] |= 1 << square

                    file_idx += 1

        # Set turn
        position.white_to_move = turn == "w"

        # Parse castling rights
        position.castling_rights = FENTranslator._decode_castling(castling)

        # Parse en passant
        position.en_passant = FENTranslator._decode_en_passant(en_passant)

        return position

    @staticmethod
    def _compress_rank(rank_squares: List[str]) -> str:
        """Compress rank representation efficiently"""
        compressed = []
        empty_count = 0

        for square in rank_squares:
            if square == " ":
                empty_count += 1
            else:
                if empty_count > 0:
                    compressed.append(str(empty_count))
                    empty_count = 0
                compressed.append(square)

        if empty_count > 0:
            compressed.append(str(empty_count))

        return "".join(compressed)

    @staticmethod
    def _encode_castling(castling_rights: int) -> str:
        """Encode castling rights based on bitwise representation"""
        rights = []
        if castling_rights & 0x8:  # White Kingside
            rights.append("K")
        if castling_rights & 0x4:  # White Queenside
            rights.append("Q")
        if castling_rights & 0x2:  # Black Kingside
            rights.append("k")
        if castling_rights & 0x1:  # Black Queenside
            rights.append("q")

        return "".join(rights) if rights else "-"

    @staticmethod
    def _decode_castling(castling_str: str) -> int:
        """Decode castling rights to bitwise representation"""
        rights = 0
        if "K" in castling_str:  # White Kingside
            rights |= 0x8
        if "Q" in castling_str:  # White Queenside
            rights |= 0x4
        if "k" in castling_str:  # Black Kingside
            rights |= 0x2
        if "q" in castling_str:  # Black Queenside
            rights |= 0x1
        return rights

    @staticmethod
    def _encode_en_passant(en_passant: Optional[int]) -> str:
        """Encode en passant square"""
        if en_passant is None:
            return "-"

        file_char = chr(ord("a") + (en_passant % 8))
        rank_num = 1 + (en_passant // 8)
        return f"{file_char}{rank_num}"

    @staticmethod
    def _decode_en_passant(en_passant_str: str) -> Optional[int]:
        """Decode en passant square"""
        if en_passant_str == "-":
            return None

        file_idx = ord(en_passant_str[0]) - ord("a")
        rank_idx = int(en_passant_str[1]) - 1
        return rank_idx * 8 + file_idx


class ChessPosition:
    def __init__(self):
        # Bitboard representation with more flexible configuration
        self.piece_positions = {
            "P": [0xFF00, 0],  # White pawns
            "R": [0x81, 0x8100000000000000],  # Rooks
            "N": [0x42, 0x4200000000000000],  # Knights
            "B": [0x24, 0x2400000000000000],  # Bishops
            "Q": [0x8, 0x800000000000000],  # Queens
            "K": [0x10, 0x1000000000000000],  # Kings
        }
        self.white_pieces = 0xFFFF
        self.black_pieces = 0xFFFF000000000000
        self.white_to_move = True
        self.castling_rights = 0xF
        self.en_passant = None

    def _get_lsb_index(self, bitboard: int) -> int:
        """Get least significant bit index"""
        return (bitboard & -bitboard).bit_length() - 1

    def to_fen(self) -> str:
        """Convert position to FEN using FENTranslator"""
        return FENTranslator.bitboard_to_fen(self)


class ChessAgent:
    def __init__(self, max_depth=12, max_time=5.0):
        # Integrate multiple components
        self.opening_book = OpeningBook(max_positions=5000)
        self.middlegame_book = MiddlegameBook(max_positions=10000)
        self.endgame_book = EndgameBook(max_positions=15000)

        # Optimization parameters
        self.max_depth = max_depth
        self.max_time = max_time
        self.start_time = None
        self.position_history = []

        # Transposition table for memoization
        self.transposition_table = {}

    def _coordinates_to_square(self, coord: Tuple[int, int]) -> int:
        file, rank = coord
        return rank * 8 + file


    def get_best_move(self, position: ChessPosition) -> Optional[str]:
        """
        Integrated move selection strategy with FEN-style move output
        1. Opening book lookup
        2. Strategic planning
        3. Depth-limited search with pruning
        4. Fallback move generation
        """
        self.current_position = position
        self.start_time = time.time()
        self.position_history.append(position)

        game_phase = self._determine_game_phase(position)
        fen = position.to_fen()

        # Opening book lookup
        if game_phase == "opening" and len(self.position_history) < 15:
            book_move = self.opening_book.get_book_move(fen)
            if book_move:
                return self._convert_move_to_fen(position, book_move)

        # Middlegame strategic planning
        if game_phase == "middlegame":
            strategy = self.middlegame_book.suggest_middlegame_plan(
                fen, primary_goal=StrategicGoal.POSITIONAL_PLAY
            )
            recommended_move = strategy.get("recommended_move")
            if recommended_move:
                return self._convert_move_to_fen(position, recommended_move)

        # Endgame strategic planning
        if game_phase == "endgame":
            strategy = self.endgame_book.suggest_endgame_plan(
                fen, primary_objective=EndgameObjective.PAWN_PROMOTION
            )
            recommended_move = strategy.get("recommended_move")
            if recommended_move:
                return self._convert_move_to_fen(position, recommended_move)

        # Generate moves with fallback
        moves = self._generate_moves(position)
        
        # Fallback if no moves generated
        if not moves:
            # Last resort: generate all legal moves including pawn moves
            moves = self._generate_pawn_moves(
                position, 
                position.piece_positions['P'][0] if position.white_to_move else position.piece_positions['P'][1],
                position.white_to_move,
                position.piece_positions
            )
        
        # If still no moves, return the first valid move or raise an exception
        if not moves:
            raise ValueError(f"No legal moves found in position: {fen}")

        # Strategic move selection
        strategy = self._identify_strategy(position)
        ordered_moves = self._order_moves(position, moves, strategy)
        best_move_coords = self._search_best_move(position, ordered_moves)

        # Ensure a move is always returned
        if best_move_coords is None and moves:
            best_move_coords = moves[0]  # Fallback to first available move

        return (
            self._convert_move_to_fen(position, best_move_coords)
            if best_move_coords
            else None
        )

    def _convert_move_to_fen(
        self, position: ChessPosition, move: Tuple[Any, Any]
    ) -> str:
        """
        Convert move to resulting FEN string
        """
        # Check if move uses coordinate tuples and convert them
        if isinstance(move[0], tuple):
            from_square = self._coordinates_to_square(move[0])
            to_square = self._coordinates_to_square(move[1])
            move = (from_square, to_square)
        new_position = self._simulate_move(position, move)
        return new_position.to_fen()


    # => <= #
    def _generate_moves(self, position: ChessPosition) -> List[Tuple[int, int]]:
        """
        Generate legal moves using bitboard techniques
        """
        moves = []

        # Precompute occupancy bitboards
        white_occupancy = sum(
            position.piece_positions[piece][0] for piece in position.piece_positions
        )
        black_occupancy = sum(
            position.piece_positions[piece][1] for piece in position.piece_positions
        )
        total_occupancy = white_occupancy | black_occupancy
        empty_squares = ~total_occupancy & 0xFFFFFFFFFFFFFFFF

        # Determine current side's pieces and opponent's occupancy
        if position.white_to_move:
            friendly_occupancy = white_occupancy
            enemy_occupancy = black_occupancy
            friendly_pieces = {
                piece: position.piece_positions[piece][0]
                for piece in position.piece_positions
            }
            enemy_pieces = {
                piece: position.piece_positions[piece][1]
                for piece in position.piece_positions
            }
        else:
            friendly_occupancy = black_occupancy
            enemy_occupancy = white_occupancy
            friendly_pieces = {
                piece: position.piece_positions[piece][1]
                for piece in position.piece_positions
            }
            enemy_pieces = {
                piece: position.piece_positions[piece][0]
                for piece in position.piece_positions
            }

        # Generate moves for each piece type
        # Pawn moves
        moves.extend(
            self._generate_pawn_moves(
                position,
                friendly_pieces["P"],
                friendly_occupancy,
                enemy_occupancy,
                empty_squares,
            )
        )

        # Knight moves
        moves.extend(
            self._generate_knight_moves(
                friendly_pieces["N"], friendly_occupancy, enemy_occupancy
            )
        )

        # Bishop moves
        moves.extend(
            self._generate_bishop_moves(
                friendly_pieces["B"],
                friendly_occupancy,
                enemy_occupancy,
                total_occupancy,
            )
        )

        # Rook moves
        moves.extend(
            self._generate_rook_moves(
                friendly_pieces["R"],
                friendly_occupancy,
                enemy_occupancy,
                total_occupancy,
            )
        )

        # Queen moves
        moves.extend(
            self._generate_queen_moves(
                friendly_pieces["Q"],
                friendly_occupancy,
                enemy_occupancy,
                total_occupancy,
            )
        )

        # King moves
        moves.extend(
            self._generate_king_moves(
                position,
                friendly_pieces["K"],
                friendly_occupancy,
                enemy_occupancy,
                total_occupancy,
            )
        )

        return moves
    def _bitboard_to_squares(self, bitboard: int) -> List[int]:
        """
        Convert a bitboard to a list of square indices.
        """
        squares = []
        temp_bb = bitboard
        while temp_bb:
            lsb = temp_bb & -temp_bb
            square = lsb.bit_length() - 1
            squares.append(square)
            temp_bb &= temp_bb - 1
        return squares

    def _generate_moves_from_bitboards(self, from_bb: int, to_bb: int, shift: int) -> List[Tuple[int, int]]:
        """
        Generate moves from bitboards, given a shift direction.
        """
        moves = []
        from_squares = self._bitboard_to_squares(from_bb)
        to_squares = [sq + shift for sq in from_squares if 0 <= sq + shift < 64]
        moves.extend(zip(from_squares, to_squares))
        return moves

    def _generate_pawn_promotions(self, promotion_bb: int, shift: int) -> List[Tuple[int, int, str]]:
        """
        Generate pawn promotion moves.
        """
        moves = []
        from_squares = self._bitboard_to_squares(promotion_bb)
        to_squares = [sq + shift for sq in from_squares if 0 <= sq + shift < 64]
        for from_sq, to_sq in zip(from_squares, to_squares):
            for promotion_piece in ['Q', 'R', 'B', 'N']:
                moves.append((from_sq, to_sq, promotion_piece))
        return moves

    def _bitboard_moves(self, to_bb: int, shift: int) -> List[Tuple[int, int]]:
        """
        Generate moves from bitboard by shifting back to find from squares.
        """
        moves = []
        temp_to_bb = to_bb
        while temp_to_bb:
            lsb = temp_to_bb & -temp_to_bb
            to_square = lsb.bit_length() - 1
            from_square = to_square - shift
            if 0 <= from_square < 64 and 0 <= to_square < 64:
                moves.append((from_square, to_square))
            temp_to_bb &= temp_to_bb - 1
        return moves

    def _generate_promotions(self, to_bb: int, shift: int) -> List[Tuple[int, int, str]]:
        """
        Generate pawn promotion moves.
        """
        moves = []
        temp_to_bb = to_bb
        while temp_to_bb:
            lsb = temp_to_bb & -temp_to_bb
            to_square = lsb.bit_length() - 1
            from_square = to_square - shift
            if 0 <= from_square < 64 and 0 <= to_square < 64:
                for promotion_piece in ['Q', 'R', 'B', 'N']:
                    moves.append((from_square, to_square, promotion_piece))
            temp_to_bb &= temp_to_bb - 1
        return moves

    def _generate_pawn_moves(
        self, position, pawns_bb, friendly_occupancy, enemy_occupancy, empty_squares
    ) -> List[Tuple[int, int]]:
        """
        Generate pawn moves, including promotions and en passant, ensuring valid square indices.
        """
        moves = []
        FILE_A = 0x0101010101010101
        FILE_H = 0x8080808080808080
        RANK_2 = 0x000000000000FF00
        RANK_7 = 0x00FF000000000000
        RANK_3 = 0x0000000000FF0000
        RANK_6 = 0x0000FF0000000000
        RANK_4 = 0x00000000FF000000
        RANK_5 = 0x000000FF00000000
        RANK_8 = 0xFF00000000000000
        RANK_1 = 0x00000000000000FF

        if position.white_to_move:
            # White pawn moves
            # Single pawn push
            single_pushes = (pawns_bb << 8) & empty_squares
            # Promotions from single pushes
            promotions = single_pushes & RANK_8
            single_pushes = single_pushes & ~RANK_8
            # Double pawn push
            double_pushes = ((single_pushes & RANK_3) << 8) & empty_squares
            # Captures
            left_captures = ((pawns_bb << 7) & enemy_occupancy) & ~FILE_A
            right_captures = ((pawns_bb << 9) & enemy_occupancy) & ~FILE_H
            # Promotions from captures
            promotions_left = left_captures & RANK_8
            left_captures = left_captures & ~RANK_8
            promotions_right = right_captures & RANK_8
            right_captures = right_captures & ~RANK_8
            # En passant captures
            en_passant_moves = []
            if position.en_passant is not None:
                en_passant_square = position.en_passant
                en_passant_bb = 1 << en_passant_square
                ep_left = ((pawns_bb << 7) & en_passant_bb) & ~FILE_A
                ep_right = ((pawns_bb << 9) & en_passant_bb) & ~FILE_H
                ep_captures = ep_left | ep_right
                if ep_captures:
                    from_squares = self._bitboard_to_squares((ep_captures >> 7) & pawns_bb) + self._bitboard_to_squares((ep_captures >> 9) & pawns_bb)
                    to_squares = [en_passant_square] * len(from_squares)
                    en_passant_moves.extend(zip(from_squares, to_squares))

            # Generate moves
            moves.extend(self._bitboard_moves(single_pushes, -8))
            moves.extend(self._bitboard_moves(double_pushes, -16))
            moves.extend(self._bitboard_moves(left_captures, -7))
            moves.extend(self._bitboard_moves(right_captures, -9))
            # Generate promotions
            moves.extend(self._generate_promotions(promotions, -8))
            moves.extend(self._generate_promotions(promotions_left, -7))
            moves.extend(self._generate_promotions(promotions_right, -9))
            # Add en passant moves
            moves.extend(en_passant_moves)

        else:
            # Black pawn moves
            # Single pawn push
            single_pushes = (pawns_bb >> 8) & empty_squares
            # Promotions from single pushes
            promotions = single_pushes & RANK_1
            single_pushes = single_pushes & ~RANK_1
            # Double pawn push
            double_pushes = ((single_pushes & RANK_6) >> 8) & empty_squares
            # Captures
            left_captures = ((pawns_bb >> 9) & enemy_occupancy) & ~FILE_H
            right_captures = ((pawns_bb >> 7) & enemy_occupancy) & ~FILE_A
            # Promotions from captures
            promotions_left = left_captures & RANK_1
            left_captures = left_captures & ~RANK_1
            promotions_right = right_captures & RANK_1
            right_captures = right_captures & ~RANK_1
            # En passant captures
            en_passant_moves = []
            if position.en_passant is not None:
                en_passant_square = position.en_passant
                en_passant_bb = 1 << en_passant_square
                ep_left = ((pawns_bb >> 9) & en_passant_bb) & ~FILE_H
                ep_right = ((pawns_bb >> 7) & en_passant_bb) & ~FILE_A
                ep_captures = ep_left | ep_right
                if ep_captures:
                    from_squares = self._bitboard_to_squares((ep_captures << 9) & pawns_bb) + self._bitboard_to_squares((ep_captures << 7) & pawns_bb)
                    to_squares = [en_passant_square] * len(from_squares)
                    en_passant_moves.extend(zip(from_squares, to_squares))

            # Generate moves
            moves.extend(self._bitboard_moves(single_pushes, 8))
            moves.extend(self._bitboard_moves(double_pushes, 16))
            moves.extend(self._bitboard_moves(left_captures, 9))
            moves.extend(self._bitboard_moves(right_captures, 7))
            # Generate promotions
            moves.extend(self._generate_promotions(promotions, 8))
            moves.extend(self._generate_promotions(promotions_left, 9))
            moves.extend(self._generate_promotions(promotions_right, 7))
            # Add en passant moves
            moves.extend(en_passant_moves)

        return moves


    def _bitboard_to_moves(self, from_bb, to_bb) -> List[Tuple[int, int]]:
        """
        Convert bitboards to move list by matching bits in from_bb and to_bb.
        """
        moves = []
        temp_from_bb = from_bb
        temp_to_bb = to_bb
        while temp_from_bb and temp_to_bb:
            from_square = (temp_from_bb & -temp_from_bb).bit_length() - 1
            to_square = (temp_to_bb & -temp_to_bb).bit_length() - 1
            moves.append((from_square, to_square))
            temp_from_bb &= temp_from_bb - 1
            temp_to_bb &= temp_to_bb - 1
        return moves

    def _generate_pawn_promotions(self, from_bb, to_bb) -> List[Tuple[int, int, str]]:
        """
        Generate pawn promotions.
        """
        moves = []
        temp_from_bb = from_bb
        temp_to_bb = to_bb
        while temp_from_bb and temp_to_bb:
            from_square = (temp_from_bb & -temp_from_bb).bit_length() - 1
            to_square = (temp_to_bb & -temp_to_bb).bit_length() - 1
            for promotion_piece in ['Q', 'R', 'B', 'N']:
                moves.append((from_square, to_square, promotion_piece))
            temp_from_bb &= temp_from_bb - 1
            temp_to_bb &= temp_to_bb - 1
        return moves


    def _generate_knight_moves(
        self, knights_bb, friendly_occupancy, enemy_occupancy
    ) -> List[Tuple[int, int]]:
        """
        Generate knight moves
        """
        moves = []
        temp_knights = knights_bb
        while temp_knights:
            from_square = (temp_knights & -temp_knights).bit_length() - 1
            attacks = self._knight_attacks(from_square)
            possible_moves = attacks & ~friendly_occupancy
            temp_to_bb = possible_moves
            while temp_to_bb:
                to_square = (temp_to_bb & -temp_to_bb).bit_length() - 1
                moves.append((from_square, to_square))
                temp_to_bb &= temp_to_bb - 1
            temp_knights &= temp_knights - 1
        return moves

    def _knight_attacks(self, square):
        """
        Returns the bitboard of knight attacks from the given square
        """
        KNIGHT_MOVE_OFFSETS = [17, 15, 10, 6, -6, -10, -15, -17]
        attacks = 0
        for offset in KNIGHT_MOVE_OFFSETS:
            to_square = square + offset
            if 0 <= to_square < 64:
                from_file = square % 8
                to_file = to_square % 8
                file_diff = abs(from_file - to_file)
                if file_diff in [1, 2]:
                    attacks |= 1 << to_square
        return attacks

    def _generate_bishop_moves(
        self, bishops_bb, friendly_occupancy, enemy_occupancy, total_occupancy
    ) -> List[Tuple[int, int]]:
        """
        Generate bishop moves
        """
        moves = []
        temp_bishops = bishops_bb
        while temp_bishops:
            from_square = (temp_bishops & -temp_bishops).bit_length() - 1
            attacks = self._bishop_attacks(from_square, total_occupancy)
            possible_moves = attacks & ~friendly_occupancy
            temp_to_bb = possible_moves
            while temp_to_bb:
                to_square = (temp_to_bb & -temp_to_bb).bit_length() - 1
                moves.append((from_square, to_square))
                temp_to_bb &= temp_to_bb - 1
            temp_bishops &= temp_bishops - 1
        return moves

    def _bishop_attacks(self, square: int, occupancy: int) -> int:
        attacks = 0
        bishop_directions = [7, -7, 9, -9]
        for direction in bishop_directions:
            attacks |= self._ray_attacks(square, occupancy, direction)
        return attacks


    def _generate_rook_moves(
        self, rooks_bb, friendly_occupancy, enemy_occupancy, total_occupancy
    ) -> List[Tuple[int, int]]:
        """
        Generate rook moves
        """
        moves = []
        temp_rooks = rooks_bb
        while temp_rooks:
            from_square = (temp_rooks & -temp_rooks).bit_length() - 1
            attacks = self._rook_attacks(from_square, total_occupancy)
            possible_moves = attacks & ~friendly_occupancy
            temp_to_bb = possible_moves
            while temp_to_bb:
                to_square = (temp_to_bb & -temp_to_bb).bit_length() - 1
                moves.append((from_square, to_square))
                temp_to_bb &= temp_to_bb - 1
            temp_rooks &= temp_rooks - 1
        return moves

    def _rook_attacks(self, square: int, occupancy: int) -> int:
        attacks = 0
        rook_directions = [8, -8, 1, -1]
        for direction in rook_directions:
            attacks |= self._ray_attacks(square, occupancy, direction)
        return attacks


    def _generate_queen_moves(
        self, queens_bb, friendly_occupancy, enemy_occupancy, total_occupancy
    ) -> List[Tuple[int, int]]:
        """
        Generate queen moves
        """
        moves = []
        temp_queens = queens_bb
        while temp_queens:
            from_square = (temp_queens & -temp_queens).bit_length() - 1
            attacks = self._queen_attacks(from_square, total_occupancy)
            possible_moves = attacks & ~friendly_occupancy
            temp_to_bb = possible_moves
            while temp_to_bb:
                to_square = (temp_to_bb & -temp_to_bb).bit_length() - 1
                moves.append((from_square, to_square))
                temp_to_bb &= temp_to_bb - 1
            temp_queens &= temp_queens - 1
        return moves

    def _queen_attacks(self, square: int, occupancy: int) -> int:
        return self._rook_attacks(square, occupancy) | self._bishop_attacks(square, occupancy)

    def _ray_attacks(self, square: int, occupancy: int, direction: int) -> int:
        """
        Generate ray attacks for a sliding piece in a specific direction.
        Uses bitboard techniques to efficiently generate attacks.
        """
        attacks = 0
        directions = {
            1: 1,    # Right
            -1: -1,  # Left
            7: -9,   # Diagonal up-left
            9: -7,   # Diagonal up-right
            -7: 9,   # Diagonal down-right
            -9: 7,   # Diagonal down-left
            8: -8,   # Up
            -8: 8    # Down
        }

        step = directions.get(direction, 0)
        if step == 0:
            return 0

        current_square = square + step
        while 0 <= current_square < 64:
            # Add current square to attacks
            attacks |= 1 << current_square

            # Check if an obstacle blocks further attacks
            if occupancy & (1 << current_square):
                break

            current_square += step
            
            # Break to prevent going off the board
            if not (0 <= current_square < 64):
                break

        return attacks



    def _are_squares_aligned(self, start_square: int, end_square: int, direction: int) -> bool:
        """
        Check if two squares are aligned in the given direction.
        Implement using an iterative approach to avoid recursion.
        """
        directions = {
            1: 1,    # Right
            -1: -1,  # Left
            7: -9,   # Diagonal up-left
            9: -7,   # Diagonal up-right
            -7: 9,   # Diagonal down-right
            -9: 7,   # Diagonal down-left
            8: -8,   # Up
            -8: 8    # Down
        }

        diff = abs(end_square - start_square)
        step = directions.get(direction, 0)

        if step == 0:
            return False

        # Check if the squares are aligned
        current_square = start_square + step
        while 0 <= current_square < 64:
            if current_square == end_square:
                return True
            if abs(current_square - start_square) > diff:
                break
            current_square += step

        return False


    def _generate_king_moves(
        self, position, king_bb, friendly_occupancy, enemy_occupancy, total_occupancy
    ) -> List[Tuple[int, int]]:
        """
        Generate king moves, including castling
        """
        moves = []
        from_square = (king_bb & -king_bb).bit_length() - 1
        attacks = self._king_attacks(from_square)
        possible_moves = attacks & ~friendly_occupancy
        temp_to_bb = possible_moves
        while temp_to_bb:
            to_square = (temp_to_bb & -temp_to_bb).bit_length() - 1
            moves.append((from_square, to_square))
            temp_to_bb &= temp_to_bb - 1

        # Castling (simplified, without checking for checks)
        if position.white_to_move:
            if position.castling_rights & 0x8:  # White kingside castling
                if not total_occupancy & 0x60:  # Squares f1 and g1 are empty
                    moves.append((from_square, from_square + 2))
            if position.castling_rights & 0x4:  # White queenside castling
                if not total_occupancy & 0xE:  # Squares b1, c1, d1 are empty
                    moves.append((from_square, from_square - 2))
        else:
            if position.castling_rights & 0x2:  # Black kingside castling
                if (
                    not total_occupancy & 0x6000000000000000
                ):  # Squares f8 and g8 are empty
                    moves.append((from_square, from_square + 2))
            if position.castling_rights & 0x1:  # Black queenside castling
                if (
                    not total_occupancy & 0x0E00000000000000
                ):  # Squares b8, c8, d8 are empty
                    moves.append((from_square, from_square - 2))

        return moves

    def _king_attacks(self, square):
        """
        Returns the bitboard of king attacks from the given square
        """
        KING_MOVE_OFFSETS = [8, 9, 1, -7, -8, -9, -1, 7]
        attacks = 0
        from_file = square % 8
        for offset in KING_MOVE_OFFSETS:
            to_square = square + offset
            if 0 <= to_square < 64:
                to_file = to_square % 8
                if abs(to_file - from_file) <= 1:
                    attacks |= 1 << to_square
        return attacks

    def _identify_strategy(self, position: ChessPosition) -> dict:
        """
        Identify current strategic context
        """
        phase = self._determine_game_phase(position)
        center_control = self._evaluate_center_control(position)
        return {"phase": phase, "center_control": center_control}

    def _order_moves(
        self, position: ChessPosition, moves: List[Tuple[int, int]], strategy: dict
    ) -> List[Tuple[int, int]]:
        """
        Move ordering with strategic and tactical considerations
        """
        scored_moves = []
        for move in moves:
            score = 0
            # Use heuristics like MVV-LVA, checks, captures, promotions
            if self._is_capture(position, move):
                score += 100
            if self._is_promotion(position, move):
                score += 80
            if self._is_check(position, move):
                score += 70
            if self._aligns_with_strategy(move, strategy):
                score += 50
            scored_moves.append((move, score))

        # Sort moves by score descending
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored_moves]

    def _search_best_move(
        self, position: ChessPosition, moves: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """
        Iterative deepening search with alpha-beta pruning.
        Adjust the depth dynamically based on time constraints.
        """
        best_move = None
        alpha = float("-inf")
        beta = float("inf")
        depth = 1
        while depth <= self.max_depth:
            if time.time() - self.start_time > self.max_time:
                break
            score, move = self._alpha_beta(position, depth, alpha, beta, True)
            if move:
                best_move = move
            depth += 1
        return best_move


    def _alpha_beta(
        self,
        position: ChessPosition,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Alpha-beta pruning algorithm with memoization and quiescence search
        """
        if time.time() - self.start_time > self.max_time:
            return 0, None

        position_key = position.to_fen()
        if position_key in self.transposition_table:
            tt_entry = self.transposition_table[position_key]
            if tt_entry["depth"] >= depth:
                return tt_entry["value"], None

        if depth == 0 or self._is_game_over(position):
            return self._quiescence_search(position, alpha, beta), None

        best_move = None
        if maximizing_player:
            max_eval = float("-inf")
            moves = self._generate_moves(position)
            moves = self._order_moves(position, moves, {})
            for move in moves:
                new_position = self._simulate_move(position, move)
                eval, _ = self._alpha_beta(new_position, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cut-off
            self.transposition_table[position_key] = {"value": max_eval, "depth": depth}
            return max_eval, best_move
        else:
            min_eval = float("inf")
            moves = self._generate_moves(position)
            moves = self._order_moves(position, moves, {})
            for move in moves:
                new_position = self._simulate_move(position, move)
                eval, _ = self._alpha_beta(new_position, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cut-off
            self.transposition_table[position_key] = {"value": min_eval, "depth": depth}
            return min_eval, best_move

    def _quiescence_search(self, position: ChessPosition, alpha: float, beta: float, depth: int = 3) -> float:
        """
        Quiescence search with a maximum depth to prevent infinite recursion.
        """
        if depth <= 0:
            return self._evaluate_position(position)

        if time.time() - self.start_time > self.max_time:
            return self._evaluate_position(position)

        stand_pat = self._evaluate_position(position)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        moves = self._generate_moves(position)
        moves = [move for move in moves if self._is_capture(position, move) and self._is_valid_move(move)]

        for move in moves:
            if time.time() - self.start_time > self.max_time:
                break
            new_position = self._simulate_move(position, move)
            score = -self._quiescence_search(new_position, -beta, -alpha, depth - 1)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _is_valid_move(self, move: Tuple[int, int]) -> bool:
        from_square, to_square = move[:2]
        return 0 <= from_square < 64 and 0 <= to_square < 64

    

    def _evaluate_position(self, position: ChessPosition) -> float:
        """
        Comprehensive evaluation function using bitboards
        """
        # Material Advantage
        material = self._evaluate_material(position)

        # Piece Activity and Mobility
        mobility = self._evaluate_mobility(position)

        # King Safety
        king_safety = self._evaluate_king_safety(position)

        # Pawn Structure
        pawn_structure = self._evaluate_pawn_structure(position)

        # Positional Factors
        positional = self._evaluate_positional_factors(position)

        # Tactical Bonus
        tactical = self._evaluate_tactics(position)

        # Immediate Threat Penalty
        immediate_threat = self._evaluate_immediate_threats(position)

        # Long-Term Risk Penalty
        long_term_risk = self._evaluate_long_term_risks(position)

        # Memoization Bonus
        memoization_bonus = self._evaluate_memoization_bonus(position)

        # Combine all components with weights
        score = (
            material * 1.0
            + mobility * 0.1
            + king_safety * 0.2
            + pawn_structure * 0.1
            + positional * 0.1
            + tactical * 0.3
            - immediate_threat * 0.5
            - long_term_risk * 0.2
            + memoization_bonus * 0.1
        )

        return score if position.white_to_move else -score

    def _get_piece_value(self, piece: str) -> float:
        piece_values = {
            "P": 1.0,
            "N": 3.0,
            "B": 3.0,
            "R": 5.0,
            "Q": 9.0,
            "K": 0.0,  # King is invaluable
        }
        return piece_values.get(piece.upper(), 0.0)

    # Implement evaluation components using bitboards
    def _evaluate_material(self, position: ChessPosition) -> float:
        """
        Evaluate material balance using bitboards
        """
        piece_values = {
            "P": 1.0,
            "N": 3.0,
            "B": 3.0,
            "R": 5.0,
            "Q": 9.0,
            "K": 0.0,  # King value is handled separately
        }
        white_material = sum(
            bin(position.piece_positions[piece][0]).count("1") * value
            for piece, value in piece_values.items()
        )
        black_material = sum(
            bin(position.piece_positions[piece][1]).count("1") * value
            for piece, value in piece_values.items()
        )
        return white_material - black_material

    def _evaluate_mobility(self, position: ChessPosition) -> float:
        """
        Evaluate piece mobility using bitboards.
        Returns a score based on the number of legal moves available.
        """
        moves = self._generate_moves(position)
        mobility = len(moves)
        # Normalize mobility score
        mobility_score = mobility / 100.0  # Adjust the divisor as needed
        return mobility_score

    def _generate_all_attacks(self, position: ChessPosition, side: int) -> int:
        """
        Generate all attack squares for the given side.
        Returns a bitboard of all squares attacked by the side.
        """
        attacks = 0
        friendly_occupancy = sum(
            position.piece_positions[piece][side] for piece in position.piece_positions
        )
        enemy_occupancy = sum(
            position.piece_positions[piece][1 - side]
            for piece in position.piece_positions
        )
        total_occupancy = friendly_occupancy | enemy_occupancy

        # Pawns
        pawns_bb = position.piece_positions["P"][side]
        if side == 0:  # White
            left_attacks = (
                pawns_bb << 7
            ) & 0xFEFEFEFEFEFEFEFE  # Mask to prevent wrapping
            right_attacks = (pawns_bb << 9) & 0x7F7F7F7F7F7F7F7F
        else:  # Black
            left_attacks = (pawns_bb >> 9) & 0xFEFEFEFEFEFEFEFE
            right_attacks = (pawns_bb >> 7) & 0x7F7F7F7F7F7F7F7F
        attacks |= left_attacks | right_attacks

        # Knights
        knights_bb = position.piece_positions["N"][side]
        temp_knights = knights_bb
        while temp_knights:
            from_square = (temp_knights & -temp_knights).bit_length() - 1
            attacks |= self._knight_attacks(from_square)
            temp_knights &= temp_knights - 1

        # Bishops
        bishops_bb = position.piece_positions["B"][side]
        temp_bishops = bishops_bb
        while temp_bishops:
            from_square = (temp_bishops & -temp_bishops).bit_length() - 1
            attacks |= self._bishop_attacks(from_square, total_occupancy)
            temp_bishops &= temp_bishops - 1

        # Rooks
        rooks_bb = position.piece_positions["R"][side]
        temp_rooks = rooks_bb
        while temp_rooks:
            from_square = (temp_rooks & -temp_rooks).bit_length() - 1
            attacks |= self._rook_attacks(from_square, total_occupancy)
            temp_rooks &= temp_rooks - 1

        # Queens
        queens_bb = position.piece_positions["Q"][side]
        temp_queens = queens_bb
        while temp_queens:
            from_square = (temp_queens & -temp_queens).bit_length() - 1
            attacks |= self._queen_attacks(from_square, total_occupancy)
            temp_queens &= temp_queens - 1

        # King (for completeness)
        king_bb = position.piece_positions["K"][side]
        if king_bb:
            from_square = (king_bb & -king_bb).bit_length() - 1
            attacks |= self._king_attacks(from_square)

        return attacks

    def _get_pawn_shield(self, position: ChessPosition, side: int) -> int:
        """
        Evaluate the pawn shield in front of the king.
        Returns the number of pawns protecting the king.
        """
        king_bb = position.piece_positions["K"][side]
        if king_bb == 0:
            return 0

        king_square = (king_bb & -king_bb).bit_length() - 1
        king_file = king_square % 8
        king_rank = king_square // 8

        pawn_bb = position.piece_positions["P"][side]
        shield = 0

        # Check squares in front of the king
        if side == 0:  # White
            forward_rank = king_rank + 1
        else:
            forward_rank = king_rank - 1

        if 0 <= forward_rank <= 7:
            for file_offset in [-1, 0, 1]:
                file = king_file + file_offset
                if 0 <= file <= 7:
                    square = forward_rank * 8 + file
                    if pawn_bb & (1 << square):
                        shield += 1

        return shield

    def _evaluate_king_safety(self, position: ChessPosition) -> float:
        """
        Evaluate king safety using bitboards.
        Penalizes exposed kings and rewards safe king positions.
        """
        safety_score = 0.0
        side = 0 if position.white_to_move else 1
        opponent_side = 1 - side

        # Get king's position
        king_bb = position.piece_positions["K"][side]
        if king_bb == 0:
            # King is missing, game over
            return float("-inf")

        king_square = (king_bb & -king_bb).bit_length() - 1

        # Determine if king is in danger
        opponent_moves = self._generate_all_attacks(position, opponent_side)

        if opponent_moves & king_bb:
            # King is under attack
            safety_score -= 5.0  # Adjust penalty as needed

        # Check for pawn shield
        pawn_shield = self._get_pawn_shield(position, side)
        safety_score += pawn_shield * 0.5  # Adjust weight as needed

        return safety_score

    def _evaluate_pawn_structure(self, position: ChessPosition) -> float:
        """
        Evaluate pawn structure using bitboards.
        Penalize weaknesses like isolated, doubled, or backward pawns.
        Reward strengths like passed pawns.
        """
        score = 0.0
        side = 0 if position.white_to_move else 1

        for current_side in [0, 1]:
            pawns_bb = position.piece_positions["P"][current_side]
            score += self._evaluate_passed_pawns(pawns_bb, current_side)
            score -= self._evaluate_isolated_pawns(pawns_bb, current_side)
            score -= self._evaluate_doubled_pawns(pawns_bb, current_side)

        return score if side == 0 else -score

    def _evaluate_passed_pawns(self, pawns_bb: int, side: int) -> float:
        """
        Evaluate passed pawns.
        """
        # Simplified passed pawn evaluation
        passed_pawns = 0
        temp_pawns = pawns_bb
        while temp_pawns:
            square = (temp_pawns & -temp_pawns).bit_length() - 1
            # In a full implementation, check if there are no opposing pawns blocking
            passed_pawns += 1  # Adjust weight as needed
            temp_pawns &= temp_pawns - 1
        return passed_pawns * 0.5  # Adjust weight as needed

    def _evaluate_isolated_pawns(self, pawns_bb: int, side: int) -> float:
        """
        Evaluate isolated pawns.
        """
        isolated_pawns = 0
        files = [0x0101010101010101 << i for i in range(8)]
        for i in range(8):
            file_bb = pawns_bb & files[i]
            if file_bb:
                # Check adjacent files for pawns
                adjacent_files = []
                if i > 0:
                    adjacent_files.append(files[i - 1])
                if i < 7:
                    adjacent_files.append(files[i + 1])
                adjacent_pawns = pawns_bb & sum(adjacent_files)
                if not adjacent_pawns:
                    # Pawn is isolated
                    isolated_pawns += bin(file_bb).count("1")
        return isolated_pawns * 0.5  # Adjust weight as needed

    def _evaluate_doubled_pawns(self, pawns_bb: int, side: int) -> float:
        """
        Evaluate doubled pawns.
        """
        doubled_pawns = 0
        files = [0x0101010101010101 << i for i in range(8)]
        for file_bb in files:
            pawns_in_file = pawns_bb & file_bb
            count = bin(pawns_in_file).count("1")
            if count > 1:
                doubled_pawns += count - 1
        return doubled_pawns * 0.5  # Adjust weight as needed

    def _evaluate_positional_factors(self, position: ChessPosition) -> float:
        """
        Evaluate positional factors like control of center, piece positioning.
        """
        score = 0.0
        side = 0 if position.white_to_move else 1

        # Control of center squares
        center_squares = [27, 28, 35, 36]  # d4, e4, d5, e5
        for current_side in [0, 1]:
            occupancy = sum(
                position.piece_positions[piece][current_side]
                for piece in position.piece_positions
            )
            for square in center_squares:
                if occupancy & (1 << square):
                    score += 0.5 if current_side == side else -0.5

        return score

    def _evaluate_tactics(self, position: ChessPosition) -> float:
        """
        Evaluate tactical opportunities using bitboards.
        """
        score = 0.0
        side = 0 if position.white_to_move else 1

        # Check for hanging pieces (undefended)
        opponent_side = 1 - side
        opponent_pieces = sum(
            position.piece_positions[piece][opponent_side]
            for piece in position.piece_positions
        )

        attacks = self._generate_all_attacks(position, side)

        for piece, bitboards in position.piece_positions.items():
            piece_bb = bitboards[opponent_side]
            temp_bb = piece_bb
            while temp_bb:
                square = (temp_bb & -temp_bb).bit_length() - 1
                if attacks & (1 << square):
                    # Piece is attacked
                    defenders = self._generate_all_attacks(position, opponent_side) & (
                        1 << square
                    )
                    if not defenders:
                        # Hanging piece
                        piece_value = self._get_piece_value(piece)
                        score += piece_value * 0.5  # Adjust weight as needed
                temp_bb &= temp_bb - 1

        return score

    def _evaluate_immediate_threats(self, position: ChessPosition) -> float:
        """
        Evaluate immediate threats from the opponent.
        Penalize if own pieces are under attack.
        """
        score = 0.0
        side = 0 if position.white_to_move else 1
        opponent_side = 1 - side

        attacks = self._generate_all_attacks(position, opponent_side)

        for piece, bitboards in position.piece_positions.items():
            piece_bb = bitboards[side]
            temp_bb = piece_bb
            while temp_bb:
                square = (temp_bb & -temp_bb).bit_length() - 1
                if attacks & (1 << square):
                    # Own piece is under attack
                    defenders = self._generate_all_attacks(position, side) & (
                        1 << square
                    )
                    piece_value = self._get_piece_value(piece)
                    if not defenders:
                        # Undefended piece
                        score += piece_value * 0.5  # Adjust penalty as needed
                    else:
                        # Defended piece
                        score += piece_value * 0.2  # Adjust penalty as needed
                temp_bb &= temp_bb - 1

        return score

    def _evaluate_long_term_risks(self, position: ChessPosition) -> float:
        """
        Evaluate long-term risks such as weak pawn structures, exposed king, etc.
        """
        score = 0.0
        # For simplicity, we can combine some previous evaluations
        score += self._evaluate_pawn_structure(position) * 0.5  # Adjust weight
        score += self._evaluate_king_safety(position) * 0.5  # Adjust weight
        return score

    def _evaluate_memoization_bonus(self, position: ChessPosition) -> float:
        """
        Evaluate bonus from memoization using bitboards.
        Provide a bonus if the position has been evaluated before with a good score.
        """
        position_key = position.to_fen()
        if position_key in self.transposition_table:
            tt_entry = self.transposition_table[position_key]
            return tt_entry["value"] * 0.1  # Adjust weight as needed
        return 0.0

    def _simulate_move(
        self, position: ChessPosition, move: Tuple[int, int]
    ) -> ChessPosition:
        """
        Simulate a move on the given position using bitboards.
        Returns a new ChessPosition representing the state after the move.
        """
        from_square, to_square = move[0], move[1]
        promotion_piece = move[2] if len(move) == 3 else None

        # Validate square indices
        if not (0 <= from_square < 64) or not (0 <= to_square < 64):
            # Invalid square index
            return position  # Return the original position or handle the error appropriately

        # Create a deep copy of the position
        new_position = ChessPosition()
        new_position.piece_positions = {
            piece: [white_bb, black_bb]
            for piece, (white_bb, black_bb) in position.piece_positions.items()
        }
        new_position.white_to_move = not position.white_to_move
        new_position.castling_rights = position.castling_rights
        new_position.en_passant = None  # Reset en passant square unless updated below

        # Determine the moving side
        if position.white_to_move:
            side = 0  # White
            opponent_side = 1  # Black
        else:
            side = 1  # Black
            opponent_side = 0  # White

        # Identify the piece being moved
        moving_piece = None
        for piece, bitboards in new_position.piece_positions.items():
            if bitboards[side] & (1 << from_square):
                moving_piece = piece
                break

        if not moving_piece:
            # No piece found at from_square, invalid move
            return new_position  # Could raise an exception or handle error

        # Remove the piece from its original square
        new_position.piece_positions[moving_piece][side] &= ~(1 << from_square)

        # Handle captures
        for piece, bitboards in new_position.piece_positions.items():
            if bitboards[opponent_side] & (1 << to_square):
                # Remove captured piece
                new_position.piece_positions[piece][opponent_side] &= ~(1 << to_square)
                break  # Only one piece can be captured

        # Handle special moves
        if moving_piece == "P":
            # Pawn moves
            if promotion_piece and (
                (position.white_to_move and to_square >= 56)
                or (not position.white_to_move and to_square <= 7)
            ):
                # Handle promotion
                new_position.piece_positions[promotion_piece][side] |= 1 << to_square
            else:
                # Regular pawn move
                new_position.piece_positions["P"][side] |= 1 << to_square

                # Handle en passant capture
                if position.en_passant == to_square:
                    # Remove the pawn that is captured en passant
                    capture_square = to_square + (8 if position.white_to_move else -8)
                    new_position.piece_positions["P"][opponent_side] &= ~(
                        1 << capture_square
                    )

                # Set en passant square if applicable
                if abs(to_square - from_square) == 16:
                    # Pawn moved two squares forward
                    en_passant_square = (from_square + to_square) // 2
                    new_position.en_passant = en_passant_square
        elif moving_piece == "K":
            # King moves
            new_position.piece_positions["K"][side] |= 1 << to_square

            # Update castling rights
            if position.white_to_move:
                new_position.castling_rights &= (
                    0x3  # Clear white castling rights (bits 3 and 2)
                )
            else:
                new_position.castling_rights &= (
                    0xC  # Clear black castling rights (bits 1 and 0)
                )

            # Handle castling
            if abs(to_square - from_square) == 2:
                # Kingside or queenside castling
                if to_square > from_square:
                    # Kingside castling
                    rook_from = to_square + 1
                    rook_to = to_square - 1
                else:
                    # Queenside castling
                    rook_from = to_square - 2
                    rook_to = to_square + 1

                # Move the rook
                new_position.piece_positions["R"][side] &= ~(1 << rook_from)
                new_position.piece_positions["R"][side] |= 1 << rook_to
        elif moving_piece == "R":
            # Rook moves
            new_position.piece_positions["R"][side] |= 1 << to_square

            # Update castling rights if rooks are moved
            if position.white_to_move:
                if from_square == 0:
                    # Moved white queenside rook
                    new_position.castling_rights &= (
                        ~0x4
                    )  # Clear white queenside castling right
                elif from_square == 7:
                    # Moved white kingside rook
                    new_position.castling_rights &= (
                        ~0x8
                    )  # Clear white kingside castling right
            else:
                if from_square == 56:
                    # Moved black queenside rook
                    new_position.castling_rights &= (
                        ~0x1
                    )  # Clear black queenside castling right
                elif from_square == 63:
                    # Moved black kingside rook
                    new_position.castling_rights &= (
                        ~0x2
                    )  # Clear black kingside castling right
        else:
            # Other pieces (Queen, Bishop, Knight)
            new_position.piece_positions[moving_piece][side] |= 1 << to_square

        # Update occupancy bitboards
        new_position.white_pieces = sum(
            bb[0] for bb in new_position.piece_positions.values()
        )
        new_position.black_pieces = sum(
            bb[1] for bb in new_position.piece_positions.values()
        )

        return new_position

    def _evaluate_total_material(self, position: ChessPosition) -> float:
        """
        Evaluate the total material on the board for both sides.
        """
        piece_values = {
            "P": 1.0,
            "N": 3.0,
            "B": 3.0,
            "R": 5.0,
            "Q": 9.0,
            "K": 0.0,  # King is invaluable
        }
        white_material = sum(
            bin(position.piece_positions[piece][0]).count("1") * value
            for piece, value in piece_values.items()
        )
        black_material = sum(
            bin(position.piece_positions[piece][1]).count("1") * value
            for piece, value in piece_values.items()
        )
        return white_material + black_material

    def _determine_game_phase(self, position: ChessPosition) -> str:
        """
        Determine the game phase based on total material.
        """
        total_material = self._evaluate_total_material(position)
        if total_material > 40:
            return "opening"
        elif total_material > 20:
            return "middlegame"
        else:
            return "endgame"


    def _is_capture(self, position: ChessPosition, move: Tuple[int, int]) -> bool:
        from_square, to_square = move[:2]
        if not (0 <= to_square < 64):
            # Invalid to_square index
            return False
        opponent_bb = (
            sum(
                position.piece_positions[piece][1]
                for piece in position.piece_positions
            )
            if position.white_to_move
            else sum(
                position.piece_positions[piece][0]
                for piece in position.piece_positions
            )
        )
        return (1 << to_square) & opponent_bb != 0


    def _is_promotion(self, position: ChessPosition, move: Tuple[int, int]) -> bool:
        """
        Check if the move is a pawn promotion
        """
        from_square, to_square = move[0], move[1]
        piece = self._get_piece_at(position, from_square)
        if piece and piece.upper() == "P":
            promotion_rank = 7 if position.white_to_move else 0
            return to_square // 8 == promotion_rank
        return False


    def _is_check(self, position: ChessPosition, move: Tuple[int, int]) -> bool:
        """
        Check if the move results in a check to the opponent's king.
        """
        # Simulate the move
        new_position = self._simulate_move(position, move)
        opponent_side = 1 if position.white_to_move else 0

        # Get opponent's king position
        king_bb = new_position.piece_positions["K"][opponent_side]
        if king_bb == 0:
            # Opponent's king is missing, so it's a checkmate
            return True

        king_square = (king_bb & -king_bb).bit_length() - 1

        # Generate all attacks from the moving side
        attacks = self._generate_all_attacks(new_position, 1 - opponent_side)

        # Check if the king is under attack
        if attacks & king_bb:
            return True
        return False

    def _get_piece_at(self, position: ChessPosition, square: int) -> Optional[str]:
        """
        Get the piece at a given square.
        """
        if not (0 <= square < 64):
            return None  # Invalid square index

        for piece, (white_bb, black_bb) in position.piece_positions.items():
            if (white_bb & (1 << square)):
                return piece
            if (black_bb & (1 << square)):
                return piece.lower()  # Lowercase for black pieces
        return None



    def _aligns_with_strategy(self, move: Tuple[int, int], strategy: dict) -> bool:
        """
        Check if a move aligns with the current strategic context.
        """
        from_square, to_square = move[0], move[1]
        moving_piece = self._get_piece_at(self.current_position, from_square)

        # Strategy considerations
        game_phase = strategy.get("phase", "middlegame")
        center_control = strategy.get("center_control", 0.0)

        # Center squares
        center_squares = [27, 28, 35, 36]  # d4, e4, d5, e5

        # Piece-specific strategies
        if game_phase == "opening":
            # Prefer developing minor pieces and controlling the center
            if moving_piece in ["N", "B"]:
                if to_square in center_squares or self._is_near_center(to_square):
                    return True
            # Encourage pawn moves to control the center
            if moving_piece == "P":
                if to_square in center_squares:
                    return True
        elif game_phase == "middlegame":
            # Focus on tactical opportunities and improving piece activity
            if self._is_capture(self.current_position, move) or self._is_check(
                self.current_position, move
            ):
                return True
            if self._evaluate_mobility(self.current_position) > 10:
                return True
        elif game_phase == "endgame":
            # Activate king and promote pawns
            if moving_piece == "K":
                return True
            if moving_piece == "P":
                if self._is_passed_pawn(self.current_position, to_square):
                    return True

        # Default to true if none of the above
        return False


    def _is_near_center(self, square: int) -> bool:
        """
        Check if a square is near the center of the board.
        """
        near_center_squares = [
            18,
            19,
            20,
            21,
            26,
            27,
            28,
            29,
            34,
            35,
            36,
            37,
            42,
            43,
            44,
            45,
        ]
        return square in near_center_squares

    def _is_passed_pawn(self, position: ChessPosition, square: int) -> bool:
        """
        Determine if a pawn is a passed pawn.
        """
        side = 0 if position.white_to_move else 1
        opponent_side = 1 - side
        pawn_bb = position.piece_positions["P"][side]
        opponent_pawn_bb = position.piece_positions["P"][opponent_side]

        file = square % 8
        rank = square // 8

        # Create mask for the pawn's file and adjacent files
        files = []
        if file > 0:
            files.append(0x0101010101010101 << (file - 1))
        files.append(0x0101010101010101 << file)
        if file < 7:
            files.append(0x0101010101010101 << (file + 1))
        mask = sum(files)

        # Mask for ranks in front of the pawn
        if side == 0:  # White pawn
            ranks = sum(1 << (r * 8 + f) for r in range(rank + 1, 8) for f in range(8))
        else:  # Black pawn
            ranks = sum(1 << (r * 8 + f) for r in range(0, rank) for f in range(8))

        # Opponent pawns in front of the pawn in the same or adjacent files
        blocking_pawns = opponent_pawn_bb & mask & ranks
        return blocking_pawns == 0

    def _is_game_over(self, position: ChessPosition) -> bool:
        """
        Check for game over conditions: checkmate or stalemate.
        """
        side = 0 if position.white_to_move else 1

        # Generate all legal moves
        moves = self._generate_legal_moves(position)

        if moves:
            return False  # Game is not over if there are legal moves

        # No legal moves, check for checkmate or stalemate
        if self._is_in_check(position, side):
            # Checkmate
            return True
        else:
            # Stalemate
            return True

    def _generate_legal_moves(self, position: ChessPosition) -> List[Tuple[int, int]]:
        """
        Generate all legal moves, filtering out moves that leave the king in check.
        """
        moves = self._generate_moves(position)
        legal_moves = []
        for move in moves:
            if not self._move_leaves_king_in_check(position, move):
                legal_moves.append(move)
        return legal_moves

    def _move_leaves_king_in_check(
        self, position: ChessPosition, move: Tuple[int, int]
    ) -> bool:
        """
        Check if making a move would leave the king in check.
        """
        new_position = self._simulate_move(position, move)
        side = 0 if position.white_to_move else 1
        return self._is_in_check(new_position, side)

    def _is_in_check(self, position: ChessPosition, side: int) -> bool:
        """
        Check if the king of the given side is in check.
        """
        king_bb = position.piece_positions["K"][side]
        if king_bb == 0:
            # King is missing, game over
            return True

        king_square = (king_bb & -king_bb).bit_length() - 1

        # Generate all attacks from the opponent
        attacks = self._generate_all_attacks(position, 1 - side)

        # Check if the king is under attack
        if attacks & king_bb:
            return True
        return False

    def _evaluate_center_control(self, position: ChessPosition) -> float:
        """
        Evaluate control over the center using bitboards.
        """
        center_squares = [27, 28, 35, 36]  # d4, e4, d5, e5
        extended_center_squares = center_squares + [
            18,
            19,
            20,
            21,
            26,
            29,
            34,
            37,
            42,
            43,
            44,
            45,
        ]

        side = 0 if position.white_to_move else 1
        opponent_side = 1 - side

        # Calculate control for both sides
        control_score = 0.0

        # Generate attack bitboards for both sides
        own_attacks = self._generate_all_attacks(position, side)
        opponent_attacks = self._generate_all_attacks(position, opponent_side)

        # Evaluate center control
        for square in center_squares:
            square_bb = 1 << square
            if own_attacks & square_bb:
                control_score += 1.0
            if opponent_attacks & square_bb:
                control_score -= 1.0

        # Evaluate extended center control (less weight)
        for square in extended_center_squares:
            square_bb = 1 << square
            if own_attacks & square_bb:
                control_score += 0.5
            if opponent_attacks & square_bb:
                control_score -= 0.5

        return control_score

"""
TESTING
"""

def test_chess_agent():
    """
    Test the ChessAgent implementation by simulating a game and assessing the correctness of its decisions.
    """
    # Initialize the chess agent
    agent = ChessAgent(max_depth=3, max_time=5.0)

    # Define test positions in FEN notation
    test_positions = [
        {
            "description": "Initial position",
            "fen": "rn1qkbnr/pppbpppp/8/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 2 3",
            "expected_best_move": None,  # Since the initial position is complex, we won't specify an expected move
        },
        {
            "description": "Simple tactical opportunity",
            "fen": "rnbqkbnr/ppp2ppp/4p3/3p4/3P4/4PN2/PPP2PPP/RNBQKB1R w KQkq - 0 3",
            "expected_best_move": "e3e4",  # Expecting pawn to e4 to challenge the center
        },
        {
            "description": "Mate in one",
            "fen": "r1bqkbnr/pppppppp/2n5/8/8/4Q3/PPPP1PPP/RNB1KBNR b KQkq - 0 2",
            "expected_best_move": "d8e7",  # Blocking the queen's attack on e7
        },
        {
            "description": "Endgame position",
            "fen": "8/5pk1/6p1/8/8/6P1/5P1K/8 w - - 0 1",
            "expected_best_move": None,  # Agent should activate the king or push pawns
        },
    ]

    for test in test_positions:
        print(f"\nTesting position: {test['description']}")
        # Convert FEN to bitboard position
        position = FENTranslator.fen_to_bitboard(test['fen'])
        agent.current_position = position  # Set the current position for the agent

        # Get the best move from the agent
        best_move_fen = agent.get_best_move(position)

        if best_move_fen:
            print("Agent's Best Move in FEN:", best_move_fen)
            # Optionally, you can convert the FEN back to a human-readable move
            new_position = FENTranslator.fen_to_bitboard(best_move_fen)
            move_made = get_move_made(position, new_position)
            print("Move Made:", move_made)
        else:
            print("Agent did not find a move.")

        # Check if the move matches the expected best move (if provided)
        if test['expected_best_move']:
            if move_made == test['expected_best_move']:
                print("Test Passed: Agent made the expected move.")
            else:
                print("Test Failed: Agent did not make the expected move.")
                print(f"Expected Move: {test['expected_best_move']}")
        else:
            print("No expected move specified for this test case.")

def get_move_made(old_position: ChessPosition, new_position: ChessPosition) -> str:
    """
    Determine the move made between two positions and return it in algebraic notation.
    """
    files = "abcdefgh"
    ranks = "12345678"

    # Compare the piece positions to find the move
    for piece in old_position.piece_positions:
        old_white_bb, old_black_bb = old_position.piece_positions[piece]
        new_white_bb, new_black_bb = new_position.piece_positions[piece]

        # Check for white pieces
        if old_position.white_to_move:
            moved_from = old_white_bb & ~new_white_bb
            moved_to = new_white_bb & ~old_white_bb
        else:
            moved_from = old_black_bb & ~new_black_bb
            moved_to = new_black_bb & ~old_black_bb

        if moved_from and moved_to:
            from_square = (moved_from & -moved_from).bit_length() - 1
            to_square = (moved_to & -moved_to).bit_length() - 1

            from_file = files[from_square % 8]
            from_rank = ranks[from_square // 8]
            to_file = files[to_square % 8]
            to_rank = ranks[to_square // 8]

            move = f"{from_file}{from_rank}{to_file}{to_rank}"
            return move

    return "Unknown Move"

if __name__ == "__main__":
    test_chess_agent()

