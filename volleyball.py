import csv
import io
import csv
import io
from typing import Dict, Tuple, Optional, List


class VolleyballMatchSimulator:
    """
    Serve-strategy decision engine for rally-scoring volleyball. Applies to High School,
    Club, College, and Pro play -- pass the level/gender that fits, or better, supply
    opponent_stats= directly from what you've actually observed of a specific opponent.

    Improvements in this version, in response to external review:
      1. Opponent-specific attack rates can override the level/gender baseline.
      2. Small-sample serve stats are shrunk toward a prior instead of trusted at face value.
      3. Deuce (both sides >= 24) is solved with an exact closed-form formula instead of a
         score-cap approximation.
      4. Sample-size-aware confidence: recommendations flag when a strategy's stats are
         too thin to trust.
      5. Sensitivity analysis: how much set-win probability moves per +1pt improvement in
         each serve outcome.
      6. Validation scaffolding: a method to check predictions against real recorded sets
         (you must supply real match logs -- there is no substitute for that, and this
         method will happily report a misleadingly good number if fed synthetic data).
      7. CSV import for serve stats exported from stat-keeping apps (e.g. GameChanger).
    """

    # ---- Shrinkage priors -------------------------------------------------------------
    # These are rough, generic starting points for serve/pass outcomes, NOT measured
    # truth for any specific team. They only matter when a strategy has few observations;
    # as sample size grows the shrinkage below converges to the observed rate regardless
    # of what these are set to. Treat them as tunable defaults, adjust if you have a
    # better generic prior (e.g. your program's historical average across all players).
    PRIOR_ACE_RATE: float = 0.10
    PRIOR_ERROR_RATE: float = 0.12
    PRIOR_OPP_PERFECT_PASS_RATE: float = 0.32
    SHRINKAGE_PSEUDO_COUNT: float = 20.0  # "weight" of the prior, in equivalent observations

    def __init__(self, level: str, gender: str, opponent_stats: Optional[Dict[str, float]] = None):
        self.level = level
        self.gender = gender
        # If real, measured opponent stats are supplied (in_system / out_system kill %),
        # use them. Otherwise fall back to the level/gender baseline table.
        self.base_rates = opponent_stats if opponent_stats else self._get_baseline_rates()

    def _get_baseline_rates(self) -> Dict[str, float]:
        """
        Fallback benchmark attack success rates based on level and gender, used only when
        you haven't supplied real opponent-specific stats via opponent_stats=.
        'in_system': opponent kill percentage when they pass perfectly.
        'out_system': opponent kill percentage when forced into a poor pass.
        """
        transitions = {
            ('Pro', 'Boys'): {'in_system': 0.64, 'out_system': 0.38},
            ('Pro', 'Girls'): {'in_system': 0.52, 'out_system': 0.28},
            ('College', 'Boys'): {'in_system': 0.58, 'out_system': 0.34},
            ('College', 'Girls'): {'in_system': 0.46, 'out_system': 0.24},
            # Club/travel ball spans a wide skill range (12s through 18s, various
            # circuits) -- these sit between High School and College as a rough midpoint,
            # not a measured figure. Prefer opponent_stats= if you have real numbers for
            # the specific club team you're facing.
            ('Club', 'Boys'): {'in_system': 0.54, 'out_system': 0.31},
            ('Club', 'Girls'): {'in_system': 0.42, 'out_system': 0.20},
            ('High School', 'Boys'): {'in_system': 0.50, 'out_system': 0.28},
            ('High School', 'Girls'): {'in_system': 0.38, 'out_system': 0.16}
        }
        return transitions.get((self.level, self.gender), {'in_system': 0.45, 'out_system': 0.25})

    # ---- Sample-size-aware rate estimation ---------------------------------------------

    @staticmethod
    def _shrink(successes: float, n: float, prior_p: float, pseudo_count: float) -> float:
        """
        Empirical-Bayes shrinkage: blends an observed rate toward a prior, weighted by how
        much data backs the observation up. A rate from 12 serves gets pulled hard toward
        the prior; one from 600 serves is barely moved. Equivalent to a Beta(prior_p *
        pseudo_count, (1-prior_p) * pseudo_count) prior updated with the observed counts.
        """
        n = max(0.0, n)
        return (successes + prior_p * pseudo_count) / (n + pseudo_count)

    @staticmethod
    def _wilson_bounds(successes: float, n: float, z: float = 1.96) -> Tuple[float, float]:
        """95% Wilson score interval for a binomial rate -- reported alongside shrunk rates
        so you can see how much uncertainty remains even after shrinkage."""
        if n <= 0:
            return (0.0, 1.0)
        p = successes / n
        denom = 1 + (z ** 2) / n
        center = p + (z ** 2) / (2 * n)
        spread = z * ((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2)) ** 0.5
        low = max(0.0, (center - spread) / denom)
        high = min(1.0, (center + spread) / denom)
        return (low, high)

    @classmethod
    def estimate_serve_rates(cls, serve_counts: Dict[str, float]) -> Dict[str, object]:
        """
        Converts raw counts into shrinkage-adjusted rates ready for
        calculate_point_win_probability, plus metadata about how much to trust them.

        serve_counts expects:
            total_serves, ace_count, error_count,
            opp_perfect_pass_count, opp_total_passes_observed (defaults to total_serves,
            but ONLY if opp_perfect_pass_count was actually supplied)

        If opp_perfect_pass_count is omitted entirely (e.g. importing from an app that
        doesn't track pass quality), that's "no data", not "observed zero successes" --
        treating it as the latter would wrongly drag the estimate toward 0 instead of
        toward the prior. So a missing key here means "trust the prior fully."
        """
        n = serve_counts['total_serves']
        ace_count = serve_counts['ace_count']
        error_count = serve_counts['error_count']

        pass_data_available = 'opp_perfect_pass_count' in serve_counts
        pass_successes = serve_counts.get('opp_perfect_pass_count', 0)
        pass_n = serve_counts.get('opp_total_passes_observed', n) if pass_data_available else 0

        ace_rate = cls._shrink(ace_count, n, cls.PRIOR_ACE_RATE, cls.SHRINKAGE_PSEUDO_COUNT)
        error_rate = cls._shrink(error_count, n, cls.PRIOR_ERROR_RATE, cls.SHRINKAGE_PSEUDO_COUNT)
        opp_perfect_pass_rate = cls._shrink(pass_successes, pass_n, cls.PRIOR_OPP_PERFECT_PASS_RATE, cls.SHRINKAGE_PSEUDO_COUNT)

        return {
            'ace_rate': ace_rate,
            'error_rate': error_rate,
            'opp_perfect_pass_rate': opp_perfect_pass_rate,
            'pass_data_available': pass_data_available,
            'sample_size': n,
            'raw_ace_rate': (ace_count / n) if n else 0.0,
            'raw_error_rate': (error_count / n) if n else 0.0,
            'ace_rate_95ci': cls._wilson_bounds(ace_count, n),
            'error_rate_95ci': cls._wilson_bounds(error_count, n),
        }

    # ---- Importing stats from tracking apps (GameChanger, etc.) -----------------------

    # Column-header aliases to try to auto-match against a CSV's header row. Case- and
    # whitespace-insensitive, substring match. NOTE: I don't have a confirmed sample of
    # GameChanger's actual volleyball export column names (their docs describe the stats
    # tracked -- service attempts, aces, errors -- but not the literal CSV headers), so
    # this is a best-effort fuzzy matcher, not a guaranteed-exact GameChanger parser. It's
    # written broadly enough to also catch similarly-named exports from other stat apps
    # (Hudl, VolleyMetrics, MaxPreps, hand-built spreadsheets, etc.). Always check the
    # printed column mapping against your file, and use column_map= to override anything
    # it gets wrong.
    _PLAYER_ALIASES = ["player", "name", "athlete"]
    _ATTEMPTS_ALIASES = ["serve att", "service att", "sa", "attempts", "total serves", "serves"]
    _ACE_ALIASES = ["ace"]
    _ERROR_ALIASES = ["serve err", "service err", "se", "errors", "err"]

    @classmethod
    def _match_column(cls, headers: List[str], aliases: List[str]) -> Optional[str]:
        normalized = {h: h.strip().lower() for h in headers}
        # Prefer exact matches first, then substring matches, so e.g. "SA" doesn't
        # accidentally grab a column like "Sac Flies" -- exact alias match wins.
        for h, norm in normalized.items():
            if norm in aliases:
                return h
        for h, norm in normalized.items():
            if any(alias in norm for alias in aliases):
                return h
        return None

    @classmethod
    def import_gamechanger_csv(cls, filepath_or_text: str, is_text: bool = False,
                                player_filter: Optional[str] = None,
                                column_map: Optional[Dict[str, str]] = None) -> Dict[str, dict]:
        """
        Imports per-player serve counts (attempts, aces, errors) from a CSV export --
        built primarily with GameChanger's volleyball "Export Stats" file in mind, but
        works with any CSV that has a player column plus serve attempts/aces/errors
        columns under roughly those names.

        IMPORTANT LIMITATION: GameChanger's volleyball stat tracking (as documented)
        covers service attempts, aces, and errors -- it does not appear to track *pass
        quality forced on the opponent* (poor/average/perfect pass rates), which this
        model also wants. Imported profiles will therefore have real ace/error/attempt
        counts but NO pass-quality data. That's handled correctly (see the shrinkage fix
        above) by falling back entirely to the model's prior for that piece, rather than
        wrongly assuming "we observed zero good passes." If you separately chart pass
        quality, add 'opp_perfect_pass_count' / 'opp_total_passes_observed' to the
        returned dict for a given player before calling get_optimal_strategy.

        Also note: most stat-tracking apps (including GameChanger, as far as I can tell)
        record serves in aggregate -- they don't tag each serve as "topspin" vs "float"
        the way this tool's own dashboard does. So an import gives you one combined serve
        profile per player, not a pre-split topspin/float pair. If you track serve type
        yourself, you'll still need to split those counts by hand.

        filepath_or_text: path to the CSV file, or the raw CSV text itself if is_text=True.
        player_filter: if given, only rows whose player column contains this substring
            (case-insensitive) are included; otherwise every player row found is returned.
        column_map: optional explicit override, e.g.
            {'player': 'Player Name', 'attempts': 'SA', 'ace': 'Ace', 'error': 'SE'}
            for any column the auto-matcher gets wrong.

        Returns: {player_name: {"total_serves": int, "ace_count": int, "error_count": int}}
        plus a special "_column_mapping_used" key describing what was auto-detected, so
        you can sanity-check it against your actual file.
        """
        text = filepath_or_text if is_text else open(filepath_or_text, newline='', encoding='utf-8-sig').read()
        reader = csv.DictReader(io.StringIO(text))
        headers = reader.fieldnames or []
        if not headers:
            raise ValueError("No header row detected -- is this actually a CSV export?")

        column_map = column_map or {}
        player_col = column_map.get('player') or cls._match_column(headers, cls._PLAYER_ALIASES)
        attempts_col = column_map.get('attempts') or cls._match_column(headers, cls._ATTEMPTS_ALIASES)
        ace_col = column_map.get('ace') or cls._match_column(headers, cls._ACE_ALIASES)
        error_col = column_map.get('error') or cls._match_column(headers, cls._ERROR_ALIASES)

        missing = [name for name, col in
                   [("player", player_col), ("serve attempts", attempts_col),
                    ("aces", ace_col), ("errors", error_col)] if col is None]
        if missing:
            raise ValueError(
                f"Couldn't auto-detect column(s) for: {', '.join(missing)}. "
                f"Headers found in the file were: {headers}. "
                f"Pass column_map={{'player': ..., 'attempts': ..., 'ace': ..., 'error': ...}} "
                f"to specify them manually."
            )

        players: Dict[str, dict] = {}
        for row in reader:
            name = (row.get(player_col) or "").strip()
            if not name:
                continue
            if player_filter and player_filter.lower() not in name.lower():
                continue

            def _to_int(val):
                try:
                    return int(float(str(val).strip() or 0))
                except (ValueError, TypeError):
                    return 0

            attempts = _to_int(row.get(attempts_col))
            aces = _to_int(row.get(ace_col))
            errors = _to_int(row.get(error_col))

            if name not in players:
                players[name] = {"total_serves": 0, "ace_count": 0, "error_count": 0}
            players[name]["total_serves"] += attempts
            players[name]["ace_count"] += aces
            players[name]["error_count"] += errors

        players["_column_mapping_used"] = {
            "player": player_col, "attempts": attempts_col, "ace": ace_col, "error": error_col,
            "note": "Double-check these against your file's actual headers. No pass-quality "
                    "data was imported (see method docstring) -- the model will fall back to "
                    "its prior for opponent pass quality until you supply that separately."
        }
        return players

    def calculate_point_win_probability(self, serve_stats: dict) -> float:
        """
        Calculates the probability that the serving team wins the rally,
        i.e. the probability we win a point WHEN WE ARE SERVING.
        Expects rate-based keys (ace_rate, error_rate, opp_perfect_pass_rate) -- if you
        have raw counts, run them through estimate_serve_rates() first.
        """
        p_ace = serve_stats['ace_rate']
        p_error = serve_stats['error_rate']
        p_in = 1.0 - (p_ace + p_error)

        if p_in <= 0:
            return p_ace

        p_perfect_pass = serve_stats['opp_perfect_pass_rate']
        opp_kill_in_sys = self.base_rates['in_system']
        opp_kill_out_sys = self.base_rates['out_system']

        opp_efficiency_weighted = (p_perfect_pass * opp_kill_in_sys) + ((1 - p_perfect_pass) * opp_kill_out_sys)
        p_we_win_rally = 1.0 - opp_efficiency_weighted

        return p_ace + (p_in * p_we_win_rally)

    # ---- Exact deuce solution ------------------------------------------------------------

    @staticmethod
    def _solve_deuce_analytically(p_s: float, p_r: float) -> Dict[int, Dict[bool, float]]:
        """
        Closed-form solution for the deuce sub-game once both teams have reached 24+.

        Beyond that point nothing in this model depends on the absolute score any more --
        only the score DIFFERENCE d (in {-1, 0, 1}, since |d| >= 2 already ends the set)
        and who is currently serving. That makes it a stationary Markov chain, solvable
        exactly instead of approximated with a score cap:

            V(d, serving) = p * V(d+1, True) + (1-p) * V(d-1, False)
            V(+2, *) = 1,  V(-2, *) = 0
            p = p_s if serving else p_r

        Writing out V(0,T), V(0,F), V(1,T), V(1,F), V(-1,T), V(-1,F) gives six linear
        equations; substituting the +-1 states into the d=0 equations reduces it to a 2x2
        linear system in V(0,T) and V(0,F), solved below. (Verified numerically against a
        brute-force high-cap DP to 6+ decimal places.)
        """
        A1 = 1 - (1 - p_s) * p_r
        A2 = p_s * (1 - p_s)
        A_rhs = p_s ** 2
        B2 = (1 - p_r) * p_r
        B_rhs = p_r * p_s

        denom = A1 ** 2 - A2 * B2
        V0F = (A1 * B_rhs + A_rhs * B2) / denom
        V0T = (A_rhs + A2 * V0F) / A1
        V1T = p_s + (1 - p_s) * V0F
        V1F = p_r + (1 - p_r) * V0F
        Vm1T = p_s * V0T
        Vm1F = p_r * V0T

        return {
            0: {True: V0T, False: V0F},
            1: {True: V1T, False: V1F},
            -1: {True: Vm1T, False: Vm1F},
        }

    def compute_set_win_expectancy(self, score_us: int, score_them: int, serving: bool,
                                    p_win_serve: float, p_win_receive: float, memo=None) -> float:
        """
        Exact probability of winning a 25-point set (win by 2), via DP for score_us/them
        < 24 and the closed-form deuce solution once both sides have reached 24+.

        p_win_serve: our probability of winning a rally when we're serving.
        p_win_receive: our probability of winning a rally when the opponent is serving
            (i.e. our side-out rate). Independent of serve strategy.
        Winning a rally means we serve next; losing it means the opponent does.
        """
        if memo is None:
            memo = {}

        if score_us >= 25 and (score_us - score_them) >= 2:
            return 1.0
        if score_them >= 25 and (score_them - score_us) >= 2:
            return 0.0

        # Exact tail: once both sides are past 24, only the score difference matters.
        if score_us >= 24 and score_them >= 24:
            diff = score_us - score_them
            deuce_solution = self._solve_deuce_analytically(p_win_serve, p_win_receive)
            return deuce_solution[diff][serving]

        state = (score_us, score_them, serving)
        if state in memo:
            return memo[state]

        p_win_point = p_win_serve if serving else p_win_receive

        prob_if_we_win_point = self.compute_set_win_expectancy(
            score_us + 1, score_them, True, p_win_serve, p_win_receive, memo)
        prob_if_we_lose_point = self.compute_set_win_expectancy(
            score_us, score_them + 1, False, p_win_serve, p_win_receive, memo)

        win_expectancy = (p_win_point * prob_if_we_win_point) + ((1 - p_win_point) * prob_if_we_lose_point)
        memo[state] = win_expectancy
        return win_expectancy

    # ---- Sensitivity analysis -----------------------------------------------------------

    def compute_sensitivity(self, score_us: int, score_them: int, serving: bool,
                             serve_stats: dict, our_sideout_rate: float,
                             step: float = 0.01) -> Dict[str, float]:
        """
        How much set-win probability changes for a +1 percentage point improvement in each
        serve outcome (higher ace rate, lower error rate, forcing worse opponent passing),
        holding the score fixed. Mirrors the tornado-chart sensitivity analysis used in the
        web dashboard version of this tool.
        """
        baseline_point = self.calculate_point_win_probability(serve_stats)
        baseline_set = self.compute_set_win_expectancy(score_us, score_them, serving, baseline_point, our_sideout_rate)

        results = {}
        for label, key, direction in [
            ("ace_rate_up", "ace_rate", +1),
            ("error_rate_down", "error_rate", -1),
            ("opp_perfect_pass_rate_down", "opp_perfect_pass_rate", -1),
        ]:
            modified = dict(serve_stats)
            modified[key] = max(0.0, min(1.0, modified[key] + direction * step))
            p_point = self.calculate_point_win_probability(modified)
            p_set = self.compute_set_win_expectancy(score_us, score_them, serving, p_point, our_sideout_rate)
            results[label] = round(p_set - baseline_set, 5)

        return results

    # ---- Strategy selection --------------------------------------------------------------

    def get_optimal_strategy(self, current_score: tuple, player_profile: Dict[str, dict],
                              serving: bool, our_sideout_rate: float,
                              min_reliable_sample: int = 30) -> dict:
        """
        Evaluates all strategies and returns the optimal choice for the coach.

        current_score: (score_us, score_them)
        serving: True if WE are about to serve at this score, False if the opponent is.
        our_sideout_rate: our probability of winning a rally when RECEIVING serve.
        player_profile: each strategy's stats can be given either as pre-computed rates
            (ace_rate, error_rate, opp_perfect_pass_rate) or as raw counts (total_serves,
            ace_count, error_count, opp_perfect_pass_count[, opp_total_passes_observed]).
            Raw counts get shrunk toward a prior via estimate_serve_rates(); pre-computed
            rates are used as-is (no sample-size correction possible without counts).
        min_reliable_sample: strategies with fewer observed serves than this get flagged
            as low_confidence in the output, regardless of how good they look.
        """
        score_us, score_them = current_score
        best_strategy = None
        max_set_win_prob = -1.0
        strategy_analysis = {}

        for strategy_name, stats in player_profile.items():
            sample_size = None
            low_confidence = False

            if 'total_serves' in stats:
                est = self.estimate_serve_rates(stats)
                rate_stats = {
                    'ace_rate': est['ace_rate'],
                    'error_rate': est['error_rate'],
                    'opp_perfect_pass_rate': est['opp_perfect_pass_rate'],
                }
                sample_size = est['sample_size']
                low_confidence = sample_size < min_reliable_sample
            else:
                rate_stats = stats

            p_point = self.calculate_point_win_probability(rate_stats)
            p_set = self.compute_set_win_expectancy(score_us, score_them, serving, p_point, our_sideout_rate)
            sensitivity = self.compute_sensitivity(score_us, score_them, serving, rate_stats, our_sideout_rate)

            strategy_analysis[strategy_name] = {
                "point_win_prob": round(p_point, 3),
                "set_win_expectancy": round(p_set, 3),
                "sample_size": sample_size,
                "low_confidence": low_confidence,
                "sensitivity_per_1pt_improvement": sensitivity,
            }

            if p_set > max_set_win_prob:
                max_set_win_prob = p_set
                best_strategy = strategy_name

        return {
            "current_score": current_score,
            "recommended_strategy": best_strategy,
            "projected_set_win_probability": round(max_set_win_prob * 100, 1),
            "any_low_confidence": any(v["low_confidence"] for v in strategy_analysis.values()),
            "full_analysis": strategy_analysis
        }

    # ---- CSV import from stat-keeping apps -------------------------------------------------

    # Fuzzy header aliases for common volleyball serve-stat exports. Note: as of this
    # writing, GameChanger's own documentation confirms a season-stats CSV export for
    # baseball/softball/basketball specifically, but does not document a fixed CSV export
    # schema for volleyball -- volleyball box scores (attempts/aces/errors) exist in-app,
    # but the exact exported column names aren't something I can verify without a sample
    # file. This matcher works on wording common to volleyball box scores generally
    # (GameChanger included, if you're able to export or copy one), rather than assuming
    # one exact schema. If a file doesn't match, the error message lists the headers it
    # found so you can extend _HEADER_ALIASES or map columns by hand.
    _HEADER_ALIASES = {
        'player': ['player', 'name', 'player name', 'athlete'],
        'total_serves': ['sa', 'serve att', 'serve attempts', 'serves', 'attempts', 'total serves', 's att', 'srv att'],
        'ace_count': ['ace', 'aces', 'sa-ace', 'service aces'],
        'error_count': ['se', 'serve err', 'serve errors', 'errors', 'service errors', 'err'],
    }

    @classmethod
    def import_from_csv(cls, csv_text: str) -> Dict[str, dict]:
        """
        Parses a CSV export (e.g. copied/exported from GameChanger or a similar
        stat-keeping app) into a player_profile-shaped dict of raw serve counts, ready
        for get_optimal_strategy() -- after you fill in opp_perfect_pass_count /
        opp_total_passes_observed, which serve-stat apps generally don't track (that's
        opponent pass-quality charting -- a different data source, like film review or a
        scouting tool such as DataVolley or Hudl).

        Column headers are matched fuzzily against common naming conventions (see
        _HEADER_ALIASES). Unmatched columns are ignored. Raises ValueError listing the
        headers it found if player name / total serves / aces / errors can't be
        identified, so you can see what needs remapping.

        Returns each player's profile with opp_perfect_pass_count set to 0 and
        opp_total_passes_observed set to (total - aces - errors) as an explicit
        placeholder -- NOT a real estimate -- so the counts still sum correctly. Replace
        these with real pass-quality data before trusting the recommendation.
        """
        reader = csv.DictReader(io.StringIO(csv_text))
        if not reader.fieldnames:
            raise ValueError("CSV appears to be empty or unreadable.")

        normalized = {h.strip().lower(): h for h in reader.fieldnames}

        def find_column(aliases):
            for alias in aliases:
                if alias in normalized:
                    return normalized[alias]
            return None

        col_player = find_column(cls._HEADER_ALIASES['player'])
        col_total = find_column(cls._HEADER_ALIASES['total_serves'])
        col_ace = find_column(cls._HEADER_ALIASES['ace_count'])
        col_err = find_column(cls._HEADER_ALIASES['error_count'])

        missing = [name for name, col in [('player', col_player), ('total serves', col_total),
                                           ('aces', col_ace), ('errors', col_err)] if col is None]
        if missing:
            raise ValueError(
                f"Couldn't find columns for: {', '.join(missing)}. "
                f"Detected headers were: {list(reader.fieldnames)}. "
                f"Rename the relevant columns to something like 'Player', 'Serve Att', "
                f"'Aces', 'Errors', or extend VolleyballMatchSimulator._HEADER_ALIASES."
            )

        profiles: Dict[str, dict] = {}
        for row in reader:
            name = (row.get(col_player) or "").strip()
            if not name:
                continue
            try:
                total = int(float(row[col_total]))
                ace = int(float(row[col_ace]))
                err = int(float(row[col_err]))
            except (ValueError, TypeError):
                continue
            profiles[name] = {
                "total_serves": total,
                "ace_count": ace,
                "error_count": err,
                "opp_perfect_pass_count": 0,
                "opp_total_passes_observed": max(0, total - ace - err),
            }

        if not profiles:
            raise ValueError("No valid player rows found in the CSV.")
        return profiles

    # ---- Validation scaffolding -----------------------------------------------------------

    @staticmethod
    def validate_against_recorded_sets(simulator: "VolleyballMatchSimulator",
                                        records: List[dict]) -> dict:
        """
        Checks the model's predictions against REAL recorded match states.

        IMPORTANT: this only means something if `records` comes from actual logged
        matches. There is no shortcut around collecting that data -- feeding this
        synthetic or made-up records will produce a number that looks like validation
        but proves nothing.

        Each record needs:
            score_us, score_them (int), serving (bool),
            p_win_serve, p_win_receive (float, the rates that were actually in effect),
            observed_winner ('us' or 'them')  -- who actually won that set

        Returns overall accuracy plus a per-record breakdown so you can see where the
        model agrees or disagrees with what actually happened.
        """
        if not records:
            raise ValueError("No records supplied -- validate_against_recorded_sets needs real match data.")

        breakdown = []
        correct = 0
        for r in records:
            predicted_prob = simulator.compute_set_win_expectancy(
                r['score_us'], r['score_them'], r['serving'], r['p_win_serve'], r['p_win_receive'])
            predicted_winner = "us" if predicted_prob >= 0.5 else "them"
            is_correct = predicted_winner == r['observed_winner']
            correct += int(is_correct)
            breakdown.append({
                **r,
                "predicted_win_prob": round(predicted_prob, 3),
                "predicted_winner": predicted_winner,
                "correct": is_correct,
            })

        return {
            "n_records": len(records),
            "accuracy": round(correct / len(records), 3),
            "breakdown": breakdown,
        }


# ==========================================
# COACH'S USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Example Scenario: High School Girls Match, generic level/gender baseline
    # (pass opponent_stats={'in_system':..., 'out_system':...} instead if you've
    # actually measured THIS opponent's attack efficiency).
    coach_simulator = VolleyballMatchSimulator(level="High School", gender="Girls")

    # Raw counts, not pre-computed rates -- this is what enables sample-size shrinkage.
    # "opp_perfect_pass_count"/"opp_total_passes_observed" track how often that serve
    # type actually let the opponent pass perfectly, out of how many times you logged it.
    player_serve_matrix = {
        "Aggressive Jump Float (Zone 1)": {
            "total_serves": 40,
            "ace_count": 5,
            "error_count": 3,
            "opp_perfect_pass_count": 10,
            "opp_total_passes_observed": 32,
        },
        "High-Risk Jump Spin (Zone 6)": {
            # Small sample on purpose, to demonstrate the low_confidence flag
            "total_serves": 11,
            "ace_count": 3,
            "error_count": 3,
            "opp_perfect_pass_count": 2,
            "opp_total_passes_observed": 5,
        },
        "Safe Standing Float (Target Weak Passer)": {
            "total_serves": 85,
            "ace_count": 3,
            "error_count": 2,
            "opp_perfect_pass_count": 36,
            "opp_total_passes_observed": 80,
        }
    }

    # Current Score: Us 22, Them 23. We are about to serve.
    current_game_state = (22, 23)
    we_are_serving = True
    our_sideout_rate = 0.62

    decision = coach_simulator.get_optimal_strategy(
        current_game_state, player_serve_matrix, we_are_serving, our_sideout_rate)

    # --- Output Report ---
    print(f"--- MATCH DECISION REPORT ({coach_simulator.level} {coach_simulator.gender}) ---")
    print(f"Score: Us {decision['current_score'][0]} | Opponent {decision['current_score'][1]}")
    print(f"RECOMMENDED STRATEGY: {decision['recommended_strategy']}")
    if decision["any_low_confidence"]:
        print("(Note: at least one strategy above has too few logged serves to fully trust its numbers.)")
    print()
    print("Strategy Breakdown:")
    for strategy, metrics in decision['full_analysis'].items():
        flag = "  [LOW CONFIDENCE - small sample]" if metrics["low_confidence"] else ""
        print(f" -> {strategy}{flag}")
        print(f"    n={metrics['sample_size']} | Rally Win Prob (on our serve): {metrics['point_win_prob']*100:.1f}% "
              f"| Set Win Expectancy: {metrics['set_win_expectancy']*100:.1f}%")
        print(f"    Sensitivity (+1pt each): {metrics['sensitivity_per_1pt_improvement']}")
    print(f"\nExecuting the recommended strategy gives you a {decision['projected_set_win_probability']}% chance to win the set.")

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
