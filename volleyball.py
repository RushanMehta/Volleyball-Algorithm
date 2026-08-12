import csv
import io
import math
from typing import Dict, Tuple, Optional


class VolleyballMatchSimulator:
    def __init__(self, level: str, gender: str):
        self.level = level
        self.gender = gender
        self.base_rates = self._get_baseline_rates()

    def _get_baseline_rates(self):
        """
        Establishes real-world benchmark attack success rates based on level and gender.

        'in_system': Opponent kill percentage when they pass perfectly.
        'out_system': Opponent kill percentage when forced into a poor pass.
        """
        transitions = {
            ("Pro", "Boys"): {"in_system": 0.64, "out_system": 0.38},
            ("Pro", "Girls"): {"in_system": 0.52, "out_system": 0.28},
            ("College", "Boys"): {"in_system": 0.58, "out_system": 0.34},
            ("College", "Girls"): {"in_system": 0.46, "out_system": 0.24},
            ("High School", "Boys"): {"in_system": 0.50, "out_system": 0.28},
            ("High School", "Girls"): {"in_system": 0.38, "out_system": 0.16},
        }

        return transitions.get(
            (self.level, self.gender),
            {"in_system": 0.45, "out_system": 0.25},
        )

    def calculate_point_win_probability(self, serve_stats: dict) -> float:
        """
        Calculates the probability that the serving team wins the rally.

        Expected keys:
            ace_rate
            error_rate
            opp_perfect_pass_rate

        All three values must be probabilities from 0 to 1.
        """
        try:
            p_ace = float(serve_stats["ace_rate"])
            p_error = float(serve_stats["error_rate"])
            p_perfect_pass = float(serve_stats["opp_perfect_pass_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "serve_stats must contain numeric ace_rate, error_rate, "
                "and opp_perfect_pass_rate values."
            ) from exc

        if not 0 <= p_ace <= 1:
            raise ValueError("ace_rate must be between 0 and 1.")

        if not 0 <= p_error <= 1:
            raise ValueError("error_rate must be between 0 and 1.")

        if p_ace + p_error > 1:
            raise ValueError("ace_rate + error_rate cannot exceed 1.")

        if not 0 <= p_perfect_pass <= 1:
            raise ValueError("opp_perfect_pass_rate must be between 0 and 1.")

        p_in = 1.0 - (p_ace + p_error)

        opp_kill_in_sys = self.base_rates["in_system"]
        opp_kill_out_sys = self.base_rates["out_system"]

        opp_efficiency_weighted = (
            p_perfect_pass * opp_kill_in_sys
            + (1.0 - p_perfect_pass) * opp_kill_out_sys
        )

        p_we_win_rally = 1.0 - opp_efficiency_weighted

        return p_ace + (p_in * p_we_win_rally)

    def compute_set_win_expectancy(
        self,
        score_us: int,
        score_them: int,
        serving: bool,
        p_win_serve: float,
        p_win_receive: float,
        memo: Optional[Dict[Tuple[int, int, bool], float]] = None,
    ) -> float:
        """
        Uses dynamic programming / recursion to find the probability of
        winning a 25-point rally-scoring set, requiring a 2-point margin.

        serving=True means our team is serving the next rally.
        serving=False means the opponent is serving the next rally.
        """
        if not 0 <= p_win_serve <= 1:
            raise ValueError("p_win_serve must be between 0 and 1.")

        if not 0 <= p_win_receive <= 1:
            raise ValueError("p_win_receive must be between 0 and 1.")

        if score_us < 0 or score_them < 0:
            raise ValueError("Scores cannot be negative.")

        if memo is None:
            memo = {}

        state = (score_us, score_them, serving)

        if state in memo:
            return memo[state]

        # Terminal set conditions.
        if score_us >= 25 and (score_us - score_them) >= 2:
            return 1.0

        if score_them >= 25 and (score_them - score_us) >= 2:
            return 0.0

        # Exact long-deuce shortcut:
        # once both teams are above 30, only the score difference matters.
        # At 31-31, for example, the next two-point race is symmetric.
        if score_us > 30 or score_them > 30:
            if score_us == score_them:
                return 0.5
            return 1.0 if score_us > score_them else 0.0

        p_win_point = p_win_serve if serving else p_win_receive

        # Winning the rally gives us the point and the serve.
        prob_if_we_win_point = self.compute_set_win_expectancy(
            score_us + 1,
            score_them,
            True,
            p_win_serve,
            p_win_receive,
            memo,
        )

        # Losing the rally gives the opponent the point and serve.
        prob_if_we_lose_point = self.compute_set_win_expectancy(
            score_us,
            score_them + 1,
            False,
            p_win_serve,
            p_win_receive,
            memo,
        )

        win_expectancy = (
            p_win_point * prob_if_we_win_point
            + (1.0 - p_win_point) * prob_if_we_lose_point
        )

        memo[state] = win_expectancy
        return win_expectancy

    def get_optimal_strategy(
        self,
        current_score: tuple,
        player_profile: dict,
        serving: bool,
        our_sideout_rate: float,
    ):
        """
        Evaluates all serving strategies and returns the one with the
        highest projected probability of winning the current set.

        current_score: (score_us, score_them)
        serving: True if we are about to serve.
        our_sideout_rate: probability that we win a rally while receiving
                           the opponent's serve.
        """
        if not isinstance(current_score, tuple) or len(current_score) != 2:
            raise ValueError("current_score must be a tuple: (score_us, score_them).")

        score_us, score_them = current_score

        if score_us < 0 or score_them < 0:
            raise ValueError("Scores cannot be negative.")

        if not 0 <= our_sideout_rate <= 1:
            raise ValueError("our_sideout_rate must be between 0 and 1.")

        if not player_profile:
            raise ValueError("player_profile cannot be empty.")

        best_strategy = None
        max_set_win_prob = -1.0
        strategy_analysis = {}

        for strategy_name, stats in player_profile.items():
            p_point = self.calculate_point_win_probability(stats)

            # Use a fresh memo table for each strategy so probabilities
            # cannot accidentally leak between different serve profiles.
            p_set = self.compute_set_win_expectancy(
                score_us,
                score_them,
                serving,
                p_point,
                our_sideout_rate,
            )

            strategy_analysis[strategy_name] = {
                "point_win_prob": round(p_point, 3),
                "set_win_expectancy": round(p_set, 3),
            }

            if p_set > max_set_win_prob:
                max_set_win_prob = p_set
                best_strategy = strategy_name

        return {
            "current_score": current_score,
            "recommended_strategy": best_strategy,
            "projected_set_win_probability": round(max_set_win_prob * 100, 1),
            "full_analysis": strategy_analysis,
        }


# ==========================================
# COACH'S USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Example Scenario: High School Girls Match
    coach_simulator = VolleyballMatchSimulator(
        level="High School",
        gender="Girls",
    )

    player_serve_matrix = {
        "Aggressive Jump Float (Zone 1)": {
            "ace_rate": 0.12,
            "error_rate": 0.06,
            "opp_perfect_pass_rate": 0.25,
        },
        "High-Risk Jump Spin (Zone 6)": {
            "ace_rate": 0.22,
            "error_rate": 0.26,
            "opp_perfect_pass_rate": 0.15,
        },
        "Safe Standing Float (Target Weak Passer)": {
            "ace_rate": 0.04,
            "error_rate": 0.02,
            "opp_perfect_pass_rate": 0.45,
        },
    }

    current_game_state = (22, 23)
    we_are_serving = True
    our_sideout_rate = 0.62

    decision = coach_simulator.get_optimal_strategy(
        current_game_state,
        player_serve_matrix,
        we_are_serving,
        our_sideout_rate,
    )

    print(
        f"--- MATCH DECISION REPORT "
        f"({coach_simulator.level} {coach_simulator.gender}) ---"
    )
    print(
        f"Score: Us {decision['current_score'][0]} | "
        f"Opponent {decision['current_score'][1]}"
    )
    print(f"RECOMMENDED STRATEGY: {decision['recommended_strategy']}\n")

    print("Strategy Breakdown:")
    for strategy, metrics in decision["full_analysis"].items():
        print(f" -> {strategy}:")
        print(
            f"    Rally Win Prob (on our serve): "
            f"{metrics['point_win_prob'] * 100:.1f}% | "
            f"Set Win Expectancy: "
            f"{metrics['set_win_expectancy'] * 100:.1f}%"
        )

    print(
        f"\nExecuting the recommended strategy gives you a "
        f"{decision['projected_set_win_probability']}% chance to win the set."
    )
    # --- Validation scaffold demo ---
    # These two "records" are ILLUSTRATIVE PLACEHOLDERS, not real match data. Replace with
    # your own logged sets (final score, who served each key point, observed winner)
    # before trusting the accuracy number this produces.
    print("\n--- Validation scaffold demo (placeholder data, NOT real matches) ---")
    example_records = [
        {"score_us": 24, "score_them": 22, "serving": True, "p_win_serve": 0.55, "p_win_receive": 0.60, "observed_winner": "us"},
        {"score_us": 20, "score_them": 24, "serving": False, "p_win_serve": 0.55, "p_win_receive": 0.60, "observed_winner": "them"},
    ]
    validation = VolleyballMatchSimulator.validate_against_recorded_sets(coach_simulator, example_records)
    print(f"Accuracy on {validation['n_records']} placeholder records: {validation['accuracy']*100:.1f}% "
          f"(replace with real logged sets for this to mean anything)")
