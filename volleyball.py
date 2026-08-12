from typing import Dict, Optional, Tuple


class VolleyballMatchSimulator:
    """Volleyball rally, set, and serving-strategy decision engine."""

    DEFAULT_RATES = {
        ("Pro", "Boys"): {"perfect": 0.64, "average": 0.52, "poor": 0.38},
        ("Pro", "Girls"): {"perfect": 0.52, "average": 0.40, "poor": 0.28},
        ("College", "Boys"): {"perfect": 0.58, "average": 0.46, "poor": 0.34},
        ("College", "Girls"): {"perfect": 0.46, "average": 0.35, "poor": 0.24},
        ("High School", "Boys"): {"perfect": 0.50, "average": 0.39, "poor": 0.28},
        ("High School", "Girls"): {"perfect": 0.38, "average": 0.27, "poor": 0.16},
        ("Club", "Boys"): {"perfect": 0.56, "average": 0.44, "poor": 0.32},
        ("Club", "Girls"): {"perfect": 0.44, "average": 0.33, "poor": 0.23},
        ("All Levels", "Boys"): {"perfect": 0.53, "average": 0.42, "poor": 0.30},
        ("All Levels", "Girls"): {"perfect": 0.43, "average": 0.33, "poor": 0.22},
        ("All Levels", "Mixed"): {"perfect": 0.48, "average": 0.38, "poor": 0.26},
    }

    def __init__(self, level: str = "All Levels", gender: str = "Mixed"):
        self.level = str(level).strip() or "All Levels"
        self.gender = str(gender).strip() or "Mixed"
        self.base_rates = self._get_baseline_rates()

    def _get_baseline_rates(self) -> Dict[str, float]:
        rates = self.DEFAULT_RATES.get((self.level, self.gender))
        if rates is not None:
            return rates.copy()

        # Use an all-levels gender baseline for custom competition levels.
        rates = self.DEFAULT_RATES.get(("All Levels", self.gender))
        if rates is not None:
            return rates.copy()

        return {"perfect": 0.48, "average": 0.38, "poor": 0.26}

    @staticmethod
    def _probability(value, name: str) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric.") from exc

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
        return value

    def calculate_point_win_probability(self, serve_stats: dict) -> float:
        """
        Calculates the probability that our team wins a rally on serve.

        Pass rates are percentages of in-play serves, excluding aces and errors.
        """
        if not isinstance(serve_stats, dict):
            raise ValueError("serve_stats must be a dictionary.")

        required = ("ace_rate", "error_rate", "opp_perfect_pass_rate")
        missing = [key for key in required if key not in serve_stats]
        if missing:
            raise ValueError(f"Missing serve statistic(s): {', '.join(missing)}.")

        p_ace = self._probability(serve_stats["ace_rate"], "ace_rate")
        p_error = self._probability(serve_stats["error_rate"], "error_rate")
        p_perfect = self._probability(
            serve_stats["opp_perfect_pass_rate"],
            "opp_perfect_pass_rate",
        )

        if p_ace + p_error > 1.0:
            raise ValueError("ace_rate + error_rate cannot exceed 1.")

        if "opp_average_pass_rate" in serve_stats:
            p_average = self._probability(
                serve_stats["opp_average_pass_rate"],
                "opp_average_pass_rate",
            )
            if p_perfect + p_average > 1.0:
                raise ValueError(
                    "opp_perfect_pass_rate + opp_average_pass_rate cannot exceed 1."
                )
            p_poor = 1.0 - p_perfect - p_average
            opponent_rally_win = (
                p_perfect * self.base_rates["perfect"]
                + p_average * self.base_rates["average"]
                + p_poor * self.base_rates["poor"]
            )
        else:
            # Backward-compatible two-tier model.
            opponent_rally_win = (
                p_perfect * self.base_rates["perfect"]
                + (1.0 - p_perfect) * self.base_rates["poor"]
            )

        in_play_rate = 1.0 - p_ace - p_error
        result = p_ace + in_play_rate * (1.0 - opponent_rally_win)

        return max(0.0, min(1.0, result))

    def _deuce_win_prob(
        self,
        diff: int,
        serving: bool,
        p_win_serve: float,
        p_win_receive: float,
    ) -> float:
        """Exact set-win probability once both teams have reached 25 points."""
        ps = p_win_serve
        pr = p_win_receive

        denominator = ps * pr + (1.0 - ps) * (1.0 - pr)
        if denominator < 1e-12:
            return 0.5

        from_deuce = (ps * pr) / denominator
        up_while_serving = ps + (1.0 - ps) * from_deuce
        up_while_receiving = pr + (1.0 - pr) * from_deuce
        down_while_serving = ps * from_deuce
        down_while_receiving = pr * from_deuce

        if diff == 0:
            return from_deuce
        if diff == 1:
            return up_while_serving if serving else up_while_receiving
        return down_while_serving if serving else down_while_receiving

    def compute_set_win_expectancy(
        self,
        score_us: int,
        score_them: int,
        serving: bool,
        p_win_serve: float,
        p_win_receive: float,
        memo: Optional[Dict[Tuple[int, int, bool], float]] = None,
    ) -> float:
        """Returns the probability that our team wins a 25-point set."""
        if not isinstance(score_us, int) or not isinstance(score_them, int):
            raise ValueError("Scores must be integers.")
        if score_us < 0 or score_them < 0:
            raise ValueError("Scores cannot be negative.")

        p_win_serve = self._probability(p_win_serve, "p_win_serve")
        p_win_receive = self._probability(p_win_receive, "p_win_receive")

        if memo is None:
            memo = {}

        state = (score_us, score_them, bool(serving))
        if state in memo:
            return memo[state]

        if score_us >= 25 and score_us - score_them >= 2:
            return 1.0
        if score_them >= 25 and score_them - score_us >= 2:
            return 0.0

        if score_us >= 25 and score_them >= 25:
            difference = max(-1, min(1, score_us - score_them))
            result = self._deuce_win_prob(
                difference,
                bool(serving),
                p_win_serve,
                p_win_receive,
            )
            memo[state] = result
            return result

        point_probability = p_win_serve if serving else p_win_receive

        win_next = self.compute_set_win_expectancy(
            score_us + 1,
            score_them,
            True,
            p_win_serve,
            p_win_receive,
            memo,
        )
        lose_next = self.compute_set_win_expectancy(
            score_us,
            score_them + 1,
            False,
            p_win_serve,
            p_win_receive,
            memo,
        )

        result = point_probability * win_next + (1.0 - point_probability) * lose_next
        memo[state] = result
        return result

    def get_optimal_strategy(
        self,
        current_score: tuple,
        player_profile: dict,
        serving: bool,
        our_sideout_rate: float,
    ) -> dict:
        """Compares serving strategies and returns the strongest recommendation."""
        if (
            not isinstance(current_score, tuple)
            or len(current_score) != 2
            or not all(isinstance(score, int) for score in current_score)
        ):
            raise ValueError("current_score must be a tuple of integers: (us, them).")

        score_us, score_them = current_score
        if score_us < 0 or score_them < 0:
            raise ValueError("Scores cannot be negative.")
        if not isinstance(player_profile, dict) or not player_profile:
            raise ValueError("player_profile must contain at least one strategy.")

        our_sideout_rate = self._probability(
            our_sideout_rate,
            "our_sideout_rate",
        )

        analysis = {}
        best_strategy = None
        best_probability = -1.0
        best_point_probability = -1.0

        for strategy_name, stats in player_profile.items():
            point_probability = self.calculate_point_win_probability(stats)
            set_probability = self.compute_set_win_expectancy(
                score_us,
                score_them,
                bool(serving),
                point_probability,
                our_sideout_rate,
            )

            analysis[strategy_name] = {
                "point_win_prob": round(point_probability, 4),
                "set_win_expectancy": round(set_probability, 4),
                "point_win_percent": round(point_probability * 100, 1),
                "set_win_percent": round(set_probability * 100, 1),
            }

            # If set probabilities tie, prefer the strategy with the higher
            # rally probability instead of relying on dictionary order.
            if (
                set_probability > best_probability
                or (
                    abs(set_probability - best_probability) < 1e-12
                    and point_probability > best_point_probability
                )
            ):
                best_strategy = strategy_name
                best_probability = set_probability
                best_point_probability = point_probability

        return {
            "current_score": current_score,
            "recommended_strategy": best_strategy,
            "projected_set_win_probability": round(best_probability * 100, 1),
            "full_analysis": analysis,
        }

    @staticmethod
    def validate_against_recorded_sets(simulator, records: list) -> dict:
        """Evaluates predictions against recorded set outcomes."""
        if not isinstance(records, list):
            raise ValueError("records must be a list.")

        predictions = []
        correct = 0

        for index, record in enumerate(records, start=1):
            required = (
                "score_us",
                "score_them",
                "serving",
                "p_win_serve",
                "p_win_receive",
                "observed_winner",
            )
            missing = [key for key in required if key not in record]
            if missing:
                raise ValueError(
                    f"Record {index} is missing: {', '.join(missing)}."
                )

            observed = str(record["observed_winner"]).lower()
            if observed not in {"us", "them"}:
                raise ValueError(
                    f"Record {index} observed_winner must be 'us' or 'them'."
                )

            probability = simulator.compute_set_win_expectancy(
                record["score_us"],
                record["score_them"],
                record["serving"],
                record["p_win_serve"],
                record["p_win_receive"],
            )
            predicted = "us" if probability >= 0.5 else "them"
            is_correct = predicted == observed
            correct += int(is_correct)

            predictions.append(
                {
                    "score": (record["score_us"], record["score_them"]),
                    "predicted": predicted,
                    "observed": observed,
                    "model_probability": round(probability, 4),
                    "correct": is_correct,
                }
            )

        count = len(records)
        return {
            "accuracy": round(correct / count, 4) if count else 0.0,
            "n_records": count,
            "correct": correct,
            "predictions": predictions,
        }


if __name__ == "__main__":
    simulator = VolleyballMatchSimulator(
        level="All Levels",
        gender="Mixed",
    )

    serve_strategies = {
        "Aggressive Jump Float": {
            "ace_rate": 0.12,
            "error_rate": 0.06,
            "opp_perfect_pass_rate": 0.25,
            "opp_average_pass_rate": 0.40,
        },
        "High-Risk Jump Spin": {
            "ace_rate": 0.22,
            "error_rate": 0.26,
            "opp_perfect_pass_rate": 0.15,
            "opp_average_pass_rate": 0.30,
        },
        "Safe Standing Float": {
            "ace_rate": 0.04,
            "error_rate": 0.02,
            "opp_perfect_pass_rate": 0.45,
            "opp_average_pass_rate": 0.35,
        },
    }

    decision = simulator.get_optimal_strategy(
        current_score=(22, 23),
        player_profile=serve_strategies,
        serving=True,
        our_sideout_rate=0.62,
    )

    print("--- SMART MATCH PLANNING & DECISION ENGINE ---")
    print("For all levels of volleyball")
    print(
        f"Score: Us {decision['current_score'][0]} | "
        f"Opponent {decision['current_score'][1]}"
    )
    print(f"Recommended strategy: {decision['recommended_strategy']}")
    print(
        f"Projected set-win probability: "
        f"{decision['projected_set_win_probability']}%"
    )

    print("\nStrategy breakdown:")
    for strategy, metrics in decision["full_analysis"].items():
        print(
            f"{strategy}: "
            f"{metrics['point_win_percent']}% rally win | "
            f"{metrics['set_win_percent']}% set win"
        )
