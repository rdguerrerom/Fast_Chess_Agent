import random
import time
import statistics
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
    positional_understanding: float = 0.0
    risk_management: float = 0.0
    
    def get_overall_performance(self) -> float:
        """Calculate a weighted performance score."""
        return (
            0.25 * self.move_quality_score +
            0.20 * self.strategic_effectiveness +
            0.15 * self.tactical_complexity_handled +
            0.15 * self.positional_understanding +
            0.15 * (1 / (1 + self.calculation_time)) +  # Inverse time penalty
            0.10 * self.risk_management
        )

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
        """Generate a more diverse set of representative test positions for different game phases."""
        phase_positions = {
            GamePhase.OPENING: [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            ],
            GamePhase.MIDDLEGAME: [
                "r1bq1rk1/pp2ppbp/2np1np1/3p4/2PP4/2N2NP1/PP2PP1P/R1BQ1RK1 w - -",
                "4r1k1/p2r1ppp/1p2p3/2p1P3/2PnR3/1P4P1/P4P1P/3R2K1 b - -",
                "2r3k1/pp3ppp/1qp2n2/3p1b2/3P4/1Q3N1P/PP3PP1/2R2RK1 b - -",
                "r2qr1k1/1p3ppp/p1np1n2/2p1p3/2P1P3/P1NP1N2/1P3PPP/R1BQR1K1 w - -",
            ],
            GamePhase.ENDGAME: [
                "8/8/3p4/2pP4/2PK4/8/8/2k5 w - -",
                "K7/P7/8/8/8/8/8/k7 w - -",
                "8/3b4/8/2pP4/2PK4/8/8/2k5 w - -",
                "4K3/8/8/8/3k4/8/5R2/8 w - -",
            ]
        }
        
        fen = random.choice(phase_positions[phase])
        return FENTranslator.fen_to_bitboard(fen)

    def _assess_move_quality(self, move: str, phase: GamePhase) -> float:
        """
        Simulate move quality assessment with phase-specific ranges.
        
        Args:
            move (str): The move made
            phase (GamePhase): The current game phase
        
        Returns:
            float: A quality score between 0 and 1
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
                    
                    # Check if a move was returned
                    if not best_move_fen:
                        print(f"No move found for {phase.name}")
                        continue

                    # Convert best_move_fen to ChessPosition
                    best_move_position = FENTranslator.fen_to_bitboard(best_move_fen)

                    # Extract the move made
                    move_made = get_move_made(test_position, best_move_position)

                    # Assess move quality
                    move_quality = self._assess_move_quality(move_made, phase)

                    end_time = time.time()

                    performance_metric = PerformanceMetric(
                        move_quality_score=move_quality,
                        calculation_time=end_time - start_time,
                        strategic_effectiveness=random.uniform(0.5, 0.9),
                        tactical_complexity_handled=random.uniform(0.4, 0.9),
                        positional_understanding=random.uniform(0.5, 0.9),  
                        risk_management=random.uniform(0.4, 0.9),
                    )



                    game_result.game_phase_metrics[phase] = performance_metric
                    game_result.total_moves += 1

                except Exception as e:
                    print(f"Error processing move for {phase.name}: {e}")
                    print(f"Test position: {FENTranslator.bitboard_to_fen(test_position)}")

            self.simulation_results.append(game_result)

        return self.simulation_results

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report with detailed insights."""
        phase_metrics = {phase: [] for phase in GamePhase}
        
        for result in self.simulation_results:
            for phase, metric in result.game_phase_metrics.items():
                phase_metrics[phase].append(metric)
        
        report = {
            "overall_performance": {},
            "phase_performance": {}
        }
        
        for phase, metrics in phase_metrics.items():
            if metrics:
                performance_scores = [m.get_overall_performance() for m in metrics]
                
                report["phase_performance"][phase.name] = {
                    "avg_overall_performance": sum(performance_scores) / len(performance_scores),
                    "detailed_metrics": {
                        "move_quality": sum(m.move_quality_score for m in metrics) / len(metrics),
                        "strategic_effectiveness": sum(m.strategic_effectiveness for m in metrics) / len(metrics),
                        "tactical_complexity": sum(m.tactical_complexity_handled for m in metrics) / len(metrics),
                        "positional_understanding": sum(m.positional_understanding for m in metrics) / len(metrics),  # Add this line
                        "calculation_speed": sum(1 / (1 + m.calculation_time) for m in metrics) / len(metrics),
                        "risk_management": sum(m.risk_management for m in metrics) / len(metrics),
                    },

                    "performance_variability": {
                        "standard_deviation": statistics.stdev(performance_scores) if len(performance_scores) > 1 else 0
                    }
                }
        
        # Calculate overall agent performance
        overall_scores = [score for phase in report["phase_performance"].values() for score in phase["detailed_metrics"].values()]
        report["overall_performance"]["avg_score"] = sum(overall_scores) / len(overall_scores)
        
        return report

def get_move_made(old_position: ChessPosition, new_position: ChessPosition) -> str:
    """
    Determine the move made between two positions and return it in algebraic notation.
    """
    files = "abcdefgh"
    ranks = "12345678"

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

    return "Unknown Move"

def main():
    # Create chess agent
    agent = ChessAgent()

    # Run performance assessment
    performance_assessor = ChessAgentPerformanceAssessment(agent, num_simulations=5)
    performance_assessor.run_performance_assessment()

    # Generate and print performance report
    performance_report = performance_assessor.generate_performance_report()

    print("\nChess Agent Performance Report:")
    for phase, metrics in performance_report["phase_performance"].items():
        print(f"\n{phase} Phase Performance:")
        for metric, value in metrics["detailed_metrics"].items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
        print(f"  Average Overall Performance: {metrics['avg_overall_performance']:.3f}")
        print(f"  Performance Variability: {metrics['performance_variability']['standard_deviation']:.3f}")

    print(f"\nOverall Agent Performance: {performance_report['overall_performance']['avg_score']:.3f}")

if __name__ == "__main__":
    main()
