import time
import math
import random
from typing import Dict, List, Optional, Tuple



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
            import random

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


import random
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum, auto


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


from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Union
import math
import random


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
    def __init__(self, max_depth=4, max_time=5.0):
        # Integrate multiple components
        self.opening_book = OpeningBook(max_positions=5000)
        self.middlegame_book = MiddlegameBook(max_positions=10000)
        self.endgame_book = EndgameBook(max_positions=15000)

        # Optimization parameters
        self.max_depth = max_depth
        self.max_time = max_time
        self.start_time = None
        self.position_history = []

    def _convert_move_to_algebraic(
        self, position: ChessPosition, move: Tuple[int, int]
    ) -> str:
        """
        Convert move coordinates to algebraic notation

        Args:
            position: Current chess position
            move: Tuple of (from_square, to_square)

        Returns:
            Algebraic notation of the move (e.g., 'e2e4')
        """
        # Define file and rank conversion
        files = "abcdefgh"
        ranks = "12345678"

        # Convert square index to algebraic notation
        def square_to_algebraic(square: int) -> str:
            file = files[square % 8]
            rank = ranks[square // 8]
            return f"{file}{rank}"

        from_square, to_square = move

        return f"{square_to_algebraic(from_square)}{square_to_algebraic(to_square)}"

    # Modify the get_best_move method to use the new notation conversion
    def get_best_move(self, position: ChessPosition) -> Optional[str]:
        """
        Integrated move selection strategy with FEN-style move output
        1. Opening book lookup
        2. Strategic planning
        3. Depth-limited search with pruning
        """
        self.start_time = time.time()
        self.position_history.append(position)

        game_phase = self._determine_game_phase(position)
        fen = position.to_fen()

        if game_phase == "opening" and len(self.position_history) < 15:
            book_move = self.opening_book.get_book_move(fen)
            if book_move:
                return self._convert_move_to_algebraic(position, book_move)

        if game_phase == "middlegame":
            strategy = self.middlegame_book.suggest_middlegame_plan(
                fen, primary_goal=StrategicGoal.POSITIONAL_PLAY
            )
            recommended_move = strategy.get("recommended_move")
            if recommended_move:
                return self._convert_move_to_algebraic(position, recommended_move)

        if game_phase == "endgame":
            strategy = self.endgame_book.suggest_endgame_plan(
                fen, primary_objective=EndgameObjective.PAWN_PROMOTION
            )
            recommended_move = strategy.get("recommended_move")
            if recommended_move:
                return self._convert_move_to_algebraic(position, recommended_move)

        moves = self._generate_moves(position)
        strategy = self._identify_strategy(position)
        ordered_moves = self._order_moves(position, moves, strategy)
        best_move_coords = self._search_best_move(position, ordered_moves, strategy)

        return (
            self._convert_move_to_algebraic(position, best_move_coords)
            if best_move_coords
            else None
        )

    def _generate_moves(self, position: ChessPosition) -> List[Tuple[int, int]]:
        """
        Comprehensive move generation using bitboard techniques
        Generates legal moves for all pieces considering:
        - Piece-specific movement rules
        - Blockers and capture mechanics
        - Board boundaries
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

        # Determine current side's pieces and opponent's occupancy
        if position.white_to_move:
            friendly_occupancy = white_occupancy
            enemy_occupancy = black_occupancy
            friendly_pieces = {
                piece: position.piece_positions[piece][0]
                for piece in position.piece_positions
            }
        else:
            friendly_occupancy = black_occupancy
            enemy_occupancy = white_occupancy
            friendly_pieces = {
                piece: position.piece_positions[piece][1]
                for piece in position.piece_positions
            }

        # Precomputed move direction masks
        NORTH = 8
        SOUTH = -8
        EAST = 1
        WEST = -1

        # Piece movement helpers
        def get_bit(bitboard: int, square: int) -> bool:
            return bool(bitboard & (1 << square))

        def set_bit(bitboard: int, square: int) -> int:
            return bitboard | (1 << square)

        def clear_bit(bitboard: int, square: int) -> int:
            return bitboard & ~(1 << square)

        # Pawn move generation
        def generate_pawn_moves(pawn_bb: int) -> List[Tuple[int, int]]:
            pawn_moves = []
            direction = NORTH if position.white_to_move else SOUTH

            while pawn_bb:
                from_square = position._get_lsb_index(pawn_bb)
                to_square = from_square + direction

                # Single push
                if 0 <= to_square < 64 and not get_bit(total_occupancy, to_square):
                    pawn_moves.append((from_square, to_square))

                    # Double push from starting rank
                    start_rank = 1 if position.white_to_move else 6
                    start_offset = 8 if position.white_to_move else -8
                    if (
                        from_square // 8 == start_rank
                        and not get_bit(total_occupancy, from_square + direction)
                        and not get_bit(total_occupancy, from_square + start_offset)
                    ):
                        pawn_moves.append((from_square, from_square + start_offset))

                # Captures (diagonal)
                capture_directions = [direction + EAST, direction + WEST]
                for capture_dir in capture_directions:
                    to_square = from_square + capture_dir
                    if (
                        0 <= to_square < 64
                        and get_bit(enemy_occupancy, to_square)
                        and abs((to_square % 8) - (from_square % 8)) == 1
                    ):
                        pawn_moves.append((from_square, to_square))

                # Remove current pawn
                pawn_bb = clear_bit(pawn_bb, from_square)

            return pawn_moves

        # Knight move generation
        def generate_knight_moves(knight_bb: int) -> List[Tuple[int, int]]:
            knight_moves = []
            knight_move_offsets = [-17, -15, -10, -6, 6, 10, 15, 17]

            while knight_bb:
                from_square = position._get_lsb_index(knight_bb)

                for offset in knight_move_offsets:
                    to_square = from_square + offset

                    # Check board boundaries and avoid friendly piece capture
                    if (
                        0 <= to_square < 64
                        and abs((to_square % 8) - (from_square % 8)) <= 2
                        and not get_bit(friendly_occupancy, to_square)
                    ):
                        knight_moves.append((from_square, to_square))

                # Remove current knight
                knight_bb = clear_bit(knight_bb, from_square)

            return knight_moves

        # Sliding piece (Bishop, Rook, Queen) move generation
        def generate_sliding_moves(
            slider_bb: int, directions: List[int]
        ) -> List[Tuple[int, int]]:
            slider_moves = []

            while slider_bb:
                from_square = position._get_lsb_index(slider_bb)

                for direction in directions:
                    current_square = from_square

                    while True:
                        current_square += direction

                        # Check board boundaries
                        if (
                            current_square < 0
                            or current_square >= 64
                            or abs(
                                (current_square % 8)
                                - ((current_square - direction) % 8)
                            )
                            > 2
                        ):
                            break

                        # Stop if friendly piece
                        if get_bit(friendly_occupancy, current_square):
                            break

                        # Add move
                        slider_moves.append((from_square, current_square))

                        # Stop if enemy piece (capture)
                        if get_bit(enemy_occupancy, current_square):
                            break

                # Remove current slider
                slider_bb = clear_bit(slider_bb, from_square)

            return slider_moves

        # King move generation
        def generate_king_moves(king_bb: int) -> List[Tuple[int, int]]:
            king_moves = []
            king_move_offsets = [-9, -8, -7, -1, 1, 7, 8, 9]

            while king_bb:
                from_square = position._get_lsb_index(king_bb)

                for offset in king_move_offsets:
                    to_square = from_square + offset

                    # Check board boundaries and avoid friendly piece capture
                    if (
                        0 <= to_square < 64
                        and abs((to_square % 8) - (from_square % 8)) <= 1
                        and not get_bit(friendly_occupancy, to_square)
                    ):
                        king_moves.append((from_square, to_square))

                # Remove current king
                king_bb = clear_bit(king_bb, from_square)

            return king_moves

        # Combine moves for all pieces
        moves.extend(generate_pawn_moves(friendly_pieces["P"]))
        moves.extend(generate_knight_moves(friendly_pieces["N"]))

        # Bishop moves (diagonal directions)
        bishop_directions = [7, 9, -7, -9]
        moves.extend(generate_sliding_moves(friendly_pieces["B"], bishop_directions))

        # Rook moves (orthogonal directions)
        rook_directions = [8, -8, 1, -1]
        moves.extend(generate_sliding_moves(friendly_pieces["R"], rook_directions))

        # Queen moves (combined bishop and rook directions)
        queen_directions = [7, 9, -7, -9, 8, -8, 1, -1]
        moves.extend(generate_sliding_moves(friendly_pieces["Q"], queen_directions))

        # King moves
        moves.extend(generate_king_moves(friendly_pieces["K"]))

        return moves

    def _identify_strategy(self, position: ChessPosition) -> dict:
        phase = self._determine_game_phase(position)
        center_control = self._evaluate_center_control(position)
        return {"phase": phase, "center_control": center_control}

    def _order_moves(
        self, position: ChessPosition, moves: List[Tuple[int, int]], strategy: dict
    ) -> List[Tuple[int, int]]:
        """
        Move ordering with strategic and tactical considerations
        - Captures
        - Checks
        - Strategic alignment
        """
        scored_moves = []
        for move in moves:
            score = 0
            # Tactical scoring
            if self._is_capture(position, move):
                score += 100

            # Strategic alignment
            if self._aligns_with_strategy(move, strategy):
                score += 50

            scored_moves.append((move, score))

        return [
            move for move, _ in sorted(scored_moves, key=lambda x: x[1], reverse=True)
        ]

    def _search_best_move(
        self, position: ChessPosition, moves: List[Tuple[int, int]], strategy: dict
    ) -> Optional[Tuple[int, int]]:
        """
        Depth-limited search with alpha-beta pruning
        Incorporates iterative deepening for time management
        """
        best_move = None
        best_score = float("-inf")

        for depth in range(1, self.max_depth + 1):
            if time.time() - self.start_time > self.max_time:
                break

            for move in moves:
                # Create move simulation
                new_position = self._simulate_move(position, move)

                # Negamax search with alpha-beta pruning
                score = -self._negamax(
                    new_position,
                    depth - 1,
                    float("-inf"),
                    float("inf"),
                    not position.white_to_move,
                )

                if score > best_score:
                    best_score = score
                    best_move = move

        return best_move

    def _negamax(
        self,
        position: ChessPosition,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
    ) -> float:
        """
        Negamax implementation with alpha-beta pruning
        Includes basic evaluation and depth control
        """
        if depth == 0 or self._is_game_over(position):
            return self._evaluate_position(position)

        moves = self._generate_moves(position)
        best_score = float("-inf")

        for move in moves:
            new_position = self._simulate_move(position, move)
            score = -self._negamax(
                new_position, depth - 1, -beta, -alpha, not maximizing_player
            )

            best_score = max(best_score, score)
            alpha = max(alpha, score)

            if alpha >= beta:
                break

        return best_score

    def _determine_game_phase(self, position: ChessPosition) -> str:
        """
        Determine the current game phase based on material and piece positions
        """
        # Count remaining pieces
        total_white_pieces = sum(
            bin(bb).count("1") for bb, _ in position.piece_positions.values()
        )

        # Basic game phase estimation
        if total_white_pieces > 20:
            return "opening"
        elif total_white_pieces > 10:
            return "middlegame"
        else:
            return "endgame"

    def _evaluate_center_control(self, position: ChessPosition) -> float:
        """
        Evaluate center control based on piece presence and influence
        Center squares are d4, d5, e4, e5
        """
        center_squares = [27, 28, 35, 36]
        center_control_score = 0

        for piece, (white_bb, black_bb) in position.piece_positions.items():
            # White pieces center control
            for sq in center_squares:
                if white_bb & (1 << sq):
                    # Different weights for different pieces
                    if piece == "P":
                        center_control_score += 3
                    elif piece in ["N", "B"]:
                        center_control_score += 5
                    elif piece in ["Q", "R"]:
                        center_control_score += 2

            # Black pieces center control (negative score)
            for sq in center_squares:
                if black_bb & (1 << sq):
                    # Different weights for different pieces
                    if piece == "P":
                        center_control_score -= 3
                    elif piece in ["N", "B"]:
                        center_control_score -= 5
                    elif piece in ["Q", "R"]:
                        center_control_score -= 2

        return center_control_score

    def _is_capture(self, position: ChessPosition, move: Tuple[int, int]) -> bool:
        """
        Determine if a move is a capture
        """
        from_square, to_square = move

        # Check if the destination square contains an opponent's piece
        for piece, (white_bb, black_bb) in position.piece_positions.items():
            # Determine opponent's bitboard based on current turn
            opponent_bb = black_bb if position.white_to_move else white_bb

            # Check if the destination square contains an opponent's piece
            if opponent_bb & (1 << to_square):
                return True

        return False

    def _aligns_with_strategy(self, move: Tuple[int, int], strategy: dict) -> bool:
        """
        Check if a move aligns with the current strategic context
        """
        from_square, to_square = move

        # Center control alignment
        center_squares = [27, 28, 35, 36]
        if to_square in center_squares:
            return True

        # Game phase specific strategies
        if strategy["phase"] == "opening":
            # Prefer developing pieces and controlling center
            return abs(to_square - from_square) > 8  # Significant move distance

        elif strategy["phase"] == "middlegame":
            # Prefer moves that create tactical opportunities
            return abs(to_square - from_square) < 16  # More tactical, smaller moves

        return False

    def _simulate_move(
        self, position: ChessPosition, move: Tuple[int, int]
    ) -> ChessPosition:
        """
        Simulate a move on the given position
        Returns a new ChessPosition representing the state after the move
        """
        # Create a deep copy of the position
        new_position = ChessPosition()
        new_position.piece_positions = {
            piece: [white, black]
            for piece, (white, black) in position.piece_positions.items()
        }
        new_position.white_to_move = not position.white_to_move
        new_position.castling_rights = position.castling_rights
        new_position.en_passant = position.en_passant

        from_square, to_square = move

        # Determine which pieces are involved
        for piece, (white_bb, black_bb) in new_position.piece_positions.items():
            # Determine current player's bitboard
            current_bb = white_bb if position.white_to_move else black_bb

            # Check if the piece is on the from_square
            if current_bb & (1 << from_square):
                # Remove piece from original square
                if position.white_to_move:
                    new_position.piece_positions[piece][0] &= ~(1 << from_square)
                    # Add piece to new square
                    new_position.piece_positions[piece][0] |= 1 << to_square
                else:
                    new_position.piece_positions[piece][1] &= ~(1 << from_square)
                    # Add piece to new square
                    new_position.piece_positions[piece][1] |= 1 << to_square

                break  # Piece found and moved

        return new_position

    def _is_game_over(self, position: ChessPosition) -> bool:
        """
        Basic check for game-ending conditions
        Very simplified - would need full chess rules implementation
        """
        # Check if king is captured (simplified)
        white_king_exists = any(
            position.piece_positions["K"][0] & (1 << sq) for sq in range(64)
        )
        black_king_exists = any(
            position.piece_positions["K"][1] & (1 << sq) for sq in range(64)
        )

        return not (white_king_exists and black_king_exists)

    def _evaluate_position(self, position: ChessPosition) -> float:
        """
        Evaluate the chess position
        Provides a basic positional and material evaluation
        """
        # Material values
        piece_values = {
            "P": 1,  # Pawn
            "N": 3,  # Knight
            "B": 3,  # Bishop
            "R": 5,  # Rook
            "Q": 9,  # Queen
            "K": 0,  # King (infinite strategic value)
        }

        # Score calculation
        white_score = 0
        black_score = 0

        # Material evaluation
        for piece, (white_bb, black_bb) in position.piece_positions.items():
            # Count white pieces
            white_piece_count = bin(white_bb).count("1")
            white_score += white_piece_count * piece_values[piece]

            # Count black pieces
            black_piece_count = bin(black_bb).count("1")
            black_score += black_piece_count * piece_values[piece]

        # Strategic bonuses
        strategy = {
            "phase": self._determine_game_phase(position),
            "center_control": self._evaluate_center_control(position),
        }

        # Adjust score based on center control and game phase
        if strategy["phase"] == "opening":
            white_score += strategy["center_control"] * 0.5
            black_score -= strategy["center_control"] * 0.5
        elif strategy["phase"] == "middlegame":
            white_score += strategy["center_control"]
            black_score -= strategy["center_control"]

        # Final score (positive favors white, negative favors black)
        return (
            white_score - black_score
            if position.white_to_move
            else black_score - white_score
        )


# Reuse existing OpeningBook and StrategicPlanner classes from the original implementation

if __name__ == "__main__":
    # Create initial position
    initial_pos = ChessPosition()

    # Convert to FEN
    initial_fen = initial_pos.to_fen()
    print(f"Initial FEN: {initial_fen}")

    # Convert back to bitboard
    reconstructed_pos = FENTranslator.fen_to_bitboard(initial_fen)

    # Create chess agent
    agent = ChessAgent()

    # Get best move in algebraic notation
    best_move = agent.get_best_move(initial_pos)
    print(f"Best initial move: {best_move}")

    # => <= #
    middlegame_book = MiddlegameBook()

    # Get strategic recommendation for an attacking plan
    strategy = middlegame_book.suggest_middlegame_plan(
        "r1bq1rk1/pp2ppbp/2np1np1/3p4/2PP4/2N2NP1/PP2PP1P/R1BQ1RK1 w - -",
        primary_goal=StrategicGoal.ATTACK,
    )
    print("strategy: ", strategy)

    # => <= #
    endgame_book = EndgameBook()

    # Get endgame strategy for fortress defense
    strategy = endgame_book.suggest_endgame_plan(
        "8/8/3p4/2pP4/2PK4/8/8/2k5 w - -",
        primary_objective=EndgameObjective.FORTRESS_DEFENSE,
    )
    print("strategy: ", strategy)
