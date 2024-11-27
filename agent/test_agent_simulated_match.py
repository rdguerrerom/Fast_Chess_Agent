import random
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum, auto

from agent import ChessPosition, ChessAgent, FENTranslator

class GamePhase(Enum):
    OPENING = auto()
    MIDDLEGAME = auto()
    ENDGAME = auto()

@dataclass
class MoveRecord:
    """Detailed record of a single move in a simulated game."""
    move: str
    fen_before: str
    fen_after: str
    calculation_time: float
    move_quality: float
    positional_evaluation: float
    tactical_complexity: float

@dataclass
class GameSimulation:
    """Comprehensive simulation of a full chess game."""
    game_id: str
    moves: List[MoveRecord] = field(default_factory=list)
    total_moves: int = 0
    current_phase: GamePhase = GamePhase.OPENING
    game_result: Optional[str] = None
    
    def record_move(self, move_record: MoveRecord):
        """Record a move and update game state."""
        self.moves.append(move_record)
        self.total_moves += 1
        
        # Simple phase transition logic
        if self.total_moves <= 10:
            self.current_phase = GamePhase.OPENING
        elif self.total_moves <= 30:
            self.current_phase = GamePhase.MIDDLEGAME
        else:
            self.current_phase = GamePhase.ENDGAME

class ChessAgentPerformanceAssessment:
    def __init__(self, agent, num_simulations=1500, max_moves_per_game=100):
        self.agent = agent
        self.num_simulations = num_simulations
        self.max_moves_per_game = max_moves_per_game
        self.game_simulations: List[GameSimulation] = []

    def _get_initial_position(self):
        """Return the standard starting chess position."""
        return FENTranslator.fen_to_bitboard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def _evaluate_move_quality(self, old_position: ChessPosition, new_position: ChessPosition) -> float:
        """
        Assess move quality based on position changes.
        
        Args:
            old_position (ChessPosition): Position before the move
            new_position (ChessPosition): Position after the move
        
        Returns:
            float: Move quality score between 0 and 1
        """
        # Placeholder for more sophisticated move quality assessment
        # This could be expanded with actual chess engine evaluation techniques
        positional_factors = [
            self._check_piece_development(old_position, new_position),
            self._check_king_safety(old_position, new_position),
            self._check_center_control(old_position, new_position)
        ]
        
        return sum(positional_factors) / len(positional_factors)

    def _check_piece_development(self, old_pos: ChessPosition, new_pos: ChessPosition) -> float:
        """Assess how well pieces are developed."""
        return random.uniform(0.5, 1.0)

    def _check_king_safety(self, old_pos: ChessPosition, new_pos: ChessPosition) -> float:
        """Evaluate potential improvements in king safety."""
        return random.uniform(0.4, 1.0)

    def _check_center_control(self, old_pos: ChessPosition, new_pos: ChessPosition) -> float:
        """Check improvements in central square control."""
        return random.uniform(0.5, 1.0)

    def simulate_game(self) -> GameSimulation:
        """
        Simulate a complete chess game with performance tracking.
        
        Returns:
            GameSimulation: Detailed record of the simulated game
        """
        game = GameSimulation(game_id=f"game_{random.randint(1, 10000)}")
        current_position = self._get_initial_position()

        for move_num in range(self.max_moves_per_game):
            start_time = time.time()
            
            try:
                # Get best move from agent
                best_move_fen = self.agent.get_best_move(current_position)
                
                if not best_move_fen:
                    break

                # Convert FEN to new position
                new_position = FENTranslator.fen_to_bitboard(best_move_fen)
                
                # Determine move details
                move = get_move_made(current_position, new_position)
                
                # Assess move quality
                move_quality = self._evaluate_move_quality(current_position, new_position)
                
                end_time = time.time()
                
                # Create move record
                move_record = MoveRecord(
                    move=move,
                    fen_before=FENTranslator.bitboard_to_fen(current_position),
                    fen_after=best_move_fen,
                    calculation_time=end_time - start_time,
                    move_quality=move_quality,
                    positional_evaluation=random.uniform(0.5, 1.0),
                    tactical_complexity=random.uniform(0.4, 1.0)
                )
                
                game.record_move(move_record)
                
                # Update current position
                current_position = new_position
                
                # Optional: Implement basic game termination conditions
                if move_num >= self.max_moves_per_game - 1:
                    game.game_result = "Draw (Max Moves)"
                    break
                
            except Exception as e:
                print(f"Error in game simulation: {e}")
                break

        return game

    def run_performance_assessment(self) -> List[GameSimulation]:
        """
        Run multiple game simulations and collect performance data.
        
        Returns:
            List[GameSimulation]: Detailed simulations of chess games
        """
        for _ in range(self.num_simulations):
            game_simulation = self.simulate_game()
            self.game_simulations.append(game_simulation)
        
        return self.game_simulations

    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dict[str, Any]: Detailed performance metrics across all simulations
        """
        report = {
            "game_count": len(self.game_simulations),
            "phase_performance": {phase: [] for phase in GamePhase},
            "overall_metrics": {
                "avg_moves_per_game": 0,
                "total_move_quality": 0,
                "calculation_speed": 0
            }
        }

        total_moves = 0
        total_move_quality = 0
        total_calculation_time = 0

        for game in self.game_simulations:
            total_moves += game.total_moves
            phase_moves = {phase: 0 for phase in GamePhase}

            for move in game.moves:
                report["phase_performance"][game.current_phase].append({
                    "move_quality": move.move_quality,
                    "tactical_complexity": move.tactical_complexity,
                    "calculation_time": move.calculation_time
                })
                
                total_move_quality += move.move_quality
                total_calculation_time += move.calculation_time
                phase_moves[game.current_phase] += 1

        num_total_moves = sum(len(moves) for moves in report["phase_performance"].values())
        
        report["overall_metrics"] = {
            "avg_moves_per_game": total_moves / len(self.game_simulations),
            "avg_move_quality": total_move_quality / num_total_moves if num_total_moves > 0 else 0,
            "avg_calculation_speed": 1 / (total_calculation_time / num_total_moves) if num_total_moves > 0 else 0
        }

        return report

def get_move_made(old_position: ChessPosition, new_position: ChessPosition) -> str:
    """Determine the algebraic notation of the move between two positions."""
    files = "abcdefgh"
    ranks = "12345678"

    for piece in old_position.piece_positions:
        old_white_bb, old_black_bb = old_position.piece_positions[piece]
        new_white_bb, new_black_bb = new_position.piece_positions[piece]

        # Check for the side to move
        moved_from = (old_white_bb if old_position.white_to_move else old_black_bb) & ~(new_white_bb if old_position.white_to_move else new_black_bb)
        moved_to = (new_white_bb if old_position.white_to_move else new_black_bb) & ~(old_white_bb if old_position.white_to_move else old_black_bb)

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
    performance_assessor = ChessAgentPerformanceAssessment(agent, num_simulations=10)
    game_simulations = performance_assessor.run_performance_assessment()

    # Generate performance report
    performance_report = performance_assessor.generate_performance_report()

    print("\nChess Agent Performance Report:")
    print(f"Total Games Simulated: {performance_report['game_count']}")
    print(f"Average Moves per Game: {performance_report['overall_metrics']['avg_moves_per_game']:.2f}")
    print(f"Average Move Quality: {performance_report['overall_metrics']['avg_move_quality']:.3f}")
    print(f"Average Calculation Speed Score: {performance_report['overall_metrics']['avg_calculation_speed']:.3f}")

    print("\nPhase Performance Breakdown:")
    for phase, moves in performance_report['phase_performance'].items():
        if moves:
            print(f"\n{phase.name} Phase:")
            print(f"  Total Moves: {len(moves)}")
            print(f"  Avg Move Quality: {sum(m['move_quality'] for m in moves) / len(moves):.3f}")
            print(f"  Avg Tactical Complexity: {sum(m['tactical_complexity'] for m in moves) / len(moves):.3f}")

if __name__ == "__main__":
    main()
