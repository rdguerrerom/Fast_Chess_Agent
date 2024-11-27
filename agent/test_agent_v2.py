import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum, auto

from agent_v2 import ChessPosition, ChessAgent, FENTranslator


class GamePhase(Enum):
    OPENING = auto()
    MIDDLEGAME = auto()
    ENDGAME = auto()


@dataclass
class PerformanceMetric:
    move_quality_score: float = 0.0
    strategic_effectiveness: float = 0.0
    calculation_time: float = 0.0
    tactical_complexity_handled: float = 0.0


@dataclass
class GameSimulationResult:
    game_id: str
    total_moves: int
    game_phase_metrics: Dict[GamePhase, PerformanceMetric] = field(default_factory=dict)
    outcome: Optional[str] = None

    def __post_init__(self):
        if not self.game_phase_metrics:
            for phase in GamePhase:
                self.game_phase_metrics[phase] = PerformanceMetric()


class ChessAgentPerformanceAssessment:
    def __init__(self, agent, num_simulations=100):
        self.agent = agent
        self.num_simulations = num_simulations
        self.simulation_results: List[GameSimulationResult] = []

    def _generate_test_position(self, phase: GamePhase) -> ChessPosition:
        """Generate representative test positions for different game phases."""
        phase_positions = {
            GamePhase.OPENING: [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
            ],
            GamePhase.MIDDLEGAME: [
                "r1bq1rk1/pp2ppbp/2np1np1/3p4/2PP4/2N2NP1/PP2PP1P/R1BQ1RK1 w - -",
                "4r1k1/p2r1ppp/1p2p3/2p1P3/2PnR3/1P4P1/P4P1P/3R2K1 b - -",
            ],
            GamePhase.ENDGAME: [
                "8/8/3p4/2pP4/2PK4/8/8/2k5 w - -",
                "K7/P7/8/8/8/8/8/k7 w - -",
            ],
        }
        # Use FENTranslator to convert FEN to ChessPosition
        fen = random.choice(phase_positions[phase])
        return FENTranslator.fen_to_bitboard(fen)

    def _assess_move_quality(self, move: str, phase: GamePhase) -> float:
        """
        Simulate move quality assessment.
        In a real implementation, this would use a chess engine evaluation.
        """
        quality_ranges = {
            GamePhase.OPENING: (0.6, 0.9),
            GamePhase.MIDDLEGAME: (0.5, 0.8),
            GamePhase.ENDGAME: (0.7, 0.95),
        }
        min_qual, max_qual = quality_ranges[phase]
        return random.uniform(min_qual, max_qual)

    def run_performance_assessment(self) -> List[GameSimulationResult]:
        """Run comprehensive performance assessment across game phases."""
        for sim in range(self.num_simulations):
            game_result = GameSimulationResult(game_id=f"game_{sim+1}", total_moves=0)

            for phase in GamePhase:
                start_time = time.time()
                test_position = self._generate_test_position(phase)

                try:
                    # Generate best move using the agent
                    best_move_fen = self.agent.get_best_move(test_position)
                    if not best_move_fen:
                        raise ValueError("Agent did not return a move.")

                    # Convert best_move_fen to ChessPosition
                    best_move_position = FENTranslator.fen_to_bitboard(best_move_fen)

                    # Extract the move made
                    move_made = get_move_made(test_position, best_move_position)

                    # Debugging: print the move to understand its structure
                    print(f"Best move for {phase.name}: {move_made}")

                    move_quality = self._assess_move_quality(move_made, phase)

                    end_time = time.time()

                    performance_metric = PerformanceMetric(
                        move_quality_score=move_quality,
                        calculation_time=end_time - start_time,
                        strategic_effectiveness=random.uniform(0.5, 0.9),
                        tactical_complexity_handled=random.uniform(0.4, 0.9),
                    )

                    game_result.game_phase_metrics[phase] = performance_metric
                    game_result.total_moves += 1

                except Exception as e:
                    print(f"Error processing move for {phase.name}: {e}")
                    print(
                        f"Test position: {FENTranslator.bitboard_to_fen(test_position)}"
                    )

            self.simulation_results.append(game_result)

        return self.simulation_results

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        phase_metrics = {phase: [] for phase in GamePhase}

        for result in self.simulation_results:
            for phase, metric in result.game_phase_metrics.items():
                phase_metrics[phase].append(metric)

        report = {}
        for phase, metrics in phase_metrics.items():
            if metrics:  # Only process phases with metrics
                report[phase.name] = {
                    "avg_move_quality": sum(m.move_quality_score for m in metrics)
                    / len(metrics),
                    "avg_calculation_time": sum(m.calculation_time for m in metrics)
                    / len(metrics),
                    "avg_strategic_effectiveness": sum(
                        m.strategic_effectiveness for m in metrics
                    )
                    / len(metrics),
                    "avg_tactical_complexity": sum(
                        m.tactical_complexity_handled for m in metrics
                    )
                    / len(metrics),
                }

        return report


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

        # Check for the side to move
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

    #return "Unknown Move"


def main():
    # Create chess agent
    agent = ChessAgent()

    # Run performance assessment
    performance_assessor = ChessAgentPerformanceAssessment(agent, num_simulations=5)
    performance_assessor.run_performance_assessment()

    # Generate and print performance report
    performance_report = performance_assessor.generate_performance_report()

    print("\nChess Agent Performance Report:")
    for phase, metrics in performance_report.items():
        print(f"\n{phase} Phase Performance:")
        for metric, value in metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")


if __name__ == "__main__":
    main()
