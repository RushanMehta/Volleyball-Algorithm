from typing import Dict, List, Tuple, Optional


class VolleyballMatchSimulator:
    def __init__(self, level: str, gender: str):
        self.level = level
        self.gender = gender
        self.base_rates = self._get_baseline_rates()

    def _get_baseline_rates(self):
        """
        Benchmark RALLY-WIN rates for the receiving team, by pass quality.

        Three tiers (probability the RECEIVING team wins the rally given the
        pass quality -- i.e. 1 minus the server's sideout rate for that tier):
            perfect : in-system pass    (server's worst case)
            average : routine pass
            poor    : out-of-system pass (server's best case)

        Values are rough, level/gender-tuned starting points; override any of
        them with your own charted rates for a real team.
        """
        transitions = {
            ("Pro", "Boys"):          {"perfect": 0.64, "average": 0.52, "poor": 0.38},
            ("Pro", "Girls"):         {"perfect": 0.52, "average": 0.40, "poor": 0.28},
            ("College", "Boys"):      {"perfect": 0.58, "average": 0.46, "poor": 0.34},
            ("College", "Girls"):     {"perfect": 0.46, "average": 0.35, "poor": 0.24},
            ("High School", "Boys"):  {"perfect": 0.50, "average": 0.39, "poor": 0.28},
            ("High School", "Girls"): {"perfect": 0.38, "average": 0.27, "poor": 0.16},
        }
        return transitions.get(
            (self.level, self.gender),
            {"perfect": 0.45, "average": 0.34, "poor": 0.25},
        )

    def calculate_point_win_probability(self, serve_stats: dict) -> float:
        """
        Probability the SERVING team wins the rally.

        Required keys (probabilities 0..1):
            ace_rate, error_rate
            opp_perfect_pass_rate  -- fraction of IN-PLAY serves that are perfect
        Optional (enables the 3-tier model for more accuracy):
            opp_average_pass_rate  -- fraction of IN-PLAY serves that are average
        If opp_average_pass_rate is omitted, the legacy 2-tier model is used
        (every non-perfect in-play serve is treated as poor), preserving
        backward compatibility with existing scripts.

        p_ace + p_error must be <= 1. Pass-rate fractions are of the in-play
        serves (ace/error excluded); perfect + average must be <= 1, and the
        remainder is treated as poor.
        """
        try:
            p_ace = float(serve_stats["ace_rate"])
            p_error = float(serve_stats["error_rate"])
            p_perfect = float(serve_stats["opp_perfect_pass_rate"])
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
        if not 0 <= p_perfect <= 1:
            raise ValueError("opp_perfect_pass_rate must be between 0 and 1.")

        p_in = 1.0 - (p_ace + p_error)

        if "opp_average_pass_rate" in serve_stats:
            p_average = float(serve_stats["opp_average_pass_rate"])
            if not 0 <= p_average <= 1:
                raise ValueError("opp_average_pass_rate must be between 0 and 1.")
            if p_perfect + p_average > 1.0 + 1e-9:
                raise ValueError("opp_perfect_pass_rate + opp_average_pass_rate cannot exceed 1.")
            p_poor = max(0.0, 1.0 - p_perfect - p_average)
            opp_rally_win = (
                p_perfect * self.base_rates["perfect"]
                + p_average * self.base_rates["average"]
                + p_poor * self.base_rates["poor"]
            )
        else:
            # Legacy 2-tier: non-perfect in-play serves treated as poor.
            p_poor = 1.0 - p_perfect
            opp_rally_win = (
                p_perfect * self.base_rates["perfect"]
                + p_poor * self.base_rates["poor"]
            )

        # Serving team wins on an ace, or by winning the rally after an in-play serve.
        return p_ace + p_in * (1.0 - opp_rally_win)

    def _deuce_win_prob(
        self,
        diff: int,
        serving: bool,
        p_win_serve: float,
        p_win_receive: float,
    ) -> float:
        """
        Exact set-win probability in the win-by-2 region (both teams >= 25).

        In this region only the clamped score difference d in {-1, 0, 1} and the
        next server matter, so the game is a stationary Markov chain with a
        closed-form solution. Let ps = P(we win a point we serve), pr = P(we win
        a point we receive). The set resolves when one team wins two consecutive
        rallies; otherwise it returns to deuce.

            V0   = ps*pr / (ps*pr + (1-ps)*(1-pr))   [from deuce, either server]
            V1T  = ps + (1-ps)*V0                    [up 1, we serve]
            V1F  = pr + (1-pr)*V0                    [up 1, opponent serves]
            Vm1T = ps*V0                             [down 1, we serve]
            Vm1F = pr*V0                             [down 1, opponent serves]

        This replaces the previous ">30 -> 0.5 / leader wins" shortcut, which
        was only correct when ps == pr == 0.5 and otherwise biased the estimate.
        """
        ps, pr = p_win_serve, p_win_receive
        denom = ps * pr + (1.0 - ps) * (1.0 - pr)
        if denom < 1e-12:
            # Pathological: no two-rally cycle ever resolves (e.g. ps=1, pr=0).
            return 0.5
        v0 = ps * pr / denom
        v1t = ps + (1.0 - ps) * v0
        v1f = pr + (1.0 - pr) * v0
        vm1t = ps * v0
        vm1f = pr * v0
        if diff == 0:
            return v0  # proved equal for either server
        if diff == 1:
            return v1t if serving else v1f
        return vm1t if serving else vm1f

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
        Dynamic-programming set-win probability for a 25-point, win-by-2 set.
        serving=True means OUR team serves the next rally.
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

        # Exact closed form for the win-by-2 region (both teams past 25),
        # instead of the old ">30" approximation that only held at ps=pr=0.5.
        if score_us >= 25 and score_them >= 25:
            diff = max(-1, min(1, score_us - score_them))
            value = self._deuce_win_prob(diff, serving, p_win_serve, p_win_receive)
            memo[state] = value
            return value

        p_win_point = p_win_serve if serving else p_win_receive

        prob_if_we_win_point = self.compute_set_win_expectancy(
            score_us + 1, score_them, True, p_win_serve, p_win_receive, memo
        )
        prob_if_we_lose_point = self.compute_set_win_expectancy(
            score_us, score_them + 1, False, p_win_serve, p_win_receive, memo
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
        Evaluates all serving strategies and returns the one with the highest
        projected probability of winning the current set.
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
            # Fresh memo per strategy so probabilities cannot leak between profiles.
            p_set = self.compute_set_win_expectancy(
                score_us, score_them, serving, p_point, our_sideout_rate
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

    @staticmethod
    def validate_against_recorded_sets(simulator, records: list) -> dict:
        """
        Compares model predictions against observed set outcomes.

        Each record: {
            "score_us", "score_them": ints,
            "serving": bool (True if OUR team served the next rally),
            "p_win_serve", "p_win_receive": floats in 0..1,
            "observed_winner": "us" | "them"
        }
        Returns accuracy = fraction of records whose predicted winner
        (model set-win prob >= 0.5 -> "us") matches the observed winner.

        NOTE: the example records in __main__ are ILLUSTRATIVE PLACEHOLDERS.
        Replace them with your own logged sets before trusting this number.
        """
        n = len(records)
        if n == 0:
            return {"accuracy": 0.0, "n_records": 0, "correct": 0, "predictions": []}
        correct = 0
        predictions = []
        for rec in records:
            p_set = simulator.compute_set_win_expectancy(
                rec["score_us"], rec["score_them"], rec["serving"],
                rec["p_win_serve"], rec["p_win_receive"],
            )
            predicted = "us" if p_set >= 0.5 else "them"
            ok = predicted == rec["observed_winner"]
            if ok:
                correct += 1
            predictions.append({
                "score": (rec["score_us"], rec["score_them"]),
                "predicted": predicted,
                "observed": rec["observed_winner"],
                "model_p": round(p_set, 3),
                "correct": ok,
            })
        return {
            "accuracy": correct / n,
            "n_records": n,
            "correct": correct,
            "predictions": predictions,
        }


# ==========================================
# COACH'S USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Example Scenario: High School Girls Match
    coach_simulator = VolleyballMatchSimulator(level="High School", gender="Girls")

    # 3-tier example (more accurate when you chart the average-pass rate too).
    # Pass-rate fractions are of IN-PLAY serves (ace/error excluded).
    player_serve_matrix = {
        "Aggressive Jump Float (Zone 1)": {
            "ace_rate": 0.12, "error_rate": 0.06,
            "opp_perfect_pass_rate": 0.25, "opp_average_pass_rate": 0.40,
        },
        "High-Risk Jump Spin (Zone 6)": {
            "ace_rate": 0.22, "error_rate": 0.26,
            "opp_perfect_pass_rate": 0.15, "opp_average_pass_rate": 0.30,
        },
        "Safe Standing Float (Target Weak Passer)": {
            "ace_rate": 0.04, "error_rate": 0.02,
            "opp_perfect_pass_rate": 0.45, "opp_average_pass_rate": 0.35,
        },
    }

    current_game_state = (22, 23)
    we_are_serving = True
    our_sideout_rate = 0.62

    decision = coach_simulator.get_optimal_strategy(
        current_game_state, player_serve_matrix, we_are_serving, our_sideout_rate
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
    # These two "records" are ILLUSTRATIVE PLACEHOLDERS, not real match data.
    # Replace with your own logged sets before trusting the accuracy number.
    print("\n--- Validation scaffold demo (placeholder data, NOT real matches) ---")
    example_records = [
        {"score_us": 24, "score_them": 22, "serving": True, "p_win_serve": 0.55, "p_win_receive": 0.60, "observed_winner": "us"},
        {"score_us": 20, "score_them": 24, "serving": False, "p_win_serve": 0.55, "p_win_receive": 0.60, "observed_winner": "them"},
    ]
    validation = VolleyballMatchSimulator.validate_against_recorded_sets(coach_simulator, example_records)
    print(f"Accuracy on {validation['n_records']} placeholder records: {validation['accuracy']*100:.1f}% "
          f"(replace with real logged sets for this to mean anything)")
