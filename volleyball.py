import numpy as np

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
            ('Pro', 'Boys'): {'in_system': 0.64, 'out_system': 0.38},
            ('Pro', 'Girls'): {'in_system': 0.52, 'out_system': 0.28},
            ('College', 'Boys'): {'in_system': 0.58, 'out_system': 0.34},
            ('College', 'Girls'): {'in_system': 0.46, 'out_system': 0.24},
            ('High School', 'Boys'): {'in_system': 0.50, 'out_system': 0.28},
            ('High School', 'Girls'): {'in_system': 0.38, 'out_system': 0.16}
        }
        return transitions.get((self.level, self.gender), {'in_system': 0.45, 'out_system': 0.25})

    def calculate_point_win_probability(self, serve_stats: dict) -> float:
        """
        Calculates the probability that the serving team wins the rally.
        """
        p_ace = serve_stats['ace_rate']
        p_error = serve_stats['error_rate']
        p_in = 1.0 - (p_ace + p_error)
        
        if p_in <= 0:
            return p_ace

        p_perfect_pass = serve_stats['opp_perfect_pass_rate']
        opp_kill_in_sys = self.base_rates['in_system']
        opp_kill_out_sys = self.base_rates['out_system']

        # Probability opponent fails to score when ball is in play
        opp_efficiency_weighted = (p_perfect_pass * opp_kill_in_sys) + ((1 - p_perfect_pass) * opp_kill_out_sys)
        p_we_win_rally = 1.0 - opp_efficiency_weighted

        # Total probability = Direct Ace + (Ball In Play * Rally Win Rate)
        return p_ace + (p_in * p_we_win_rally)

    def compute_set_win_expectancy(self, score_us: int, score_them: int, p_win_point: float, memo=None) -> float:
        """
        Uses Dynamic Programming / Recursion to find the exact probability 
        of winning a 25-point set (must win by 2) from any current score state.
        """
        if memo is None:
            memo = {}
            
        state = (score_us, score_them)
        if state in memo:
            return memo[state]

        # Base Win/Loss Conditions
        if score_us >= 25 and (score_us - score_them) >= 2:
            return 1.0
        if score_them >= 25 and (score_them - score_us) >= 2:
            return 0.0
        
        # Cap deep deuce simulations to prevent stack overflows while maintaining accuracy
        if score_us > 30 or score_them > 30:
            if score_us == score_them:
                return 0.5
            return 1.0 if score_us > score_them else 0.0

        # Law of Total Probability: Probability of winning from this state
        prob_if_we_win_point = self.compute_set_win_expectancy(score_us + 1, score_them, p_win_point, memo)
        prob_if_we_lose_point = self.compute_set_win_expectancy(score_us, score_them + 1, p_win_point, memo)
        
        win_expectancy = (p_win_point * prob_if_we_win_point) + ((1 - p_win_point) * prob_if_we_lose_point)
        memo[state] = win_expectancy
        return win_expectancy

    def get_optimal_strategy(self, current_score: tuple, player_profile: dict):
        """
        Evaluates all strategies and returns the optimal choice for the coach.
        """
        score_us, score_them = current_score
        best_strategy = None
        max_set_win_prob = -1.0
        strategy_analysis = {}

        for strategy_name, stats in player_profile.items():
            # 1. Calculate probability of winning just this single point
            p_point = self.calculate_point_win_probability(stats)
            
            # 2. Calculate what that point does for our chances of winning the whole set
            p_set = self.compute_set_win_expectancy(score_us, score_them, p_point)
            
            strategy_analysis[strategy_name] = {
                "point_win_prob": round(p_point, 3),
                "set_win_expectancy": round(p_set, 3)
            }

            if p_set > max_set_win_prob:
                max_set_win_prob = p_set
                best_strategy = strategy_name

        return {
            "current_score": current_score,
            "recommended_strategy": best_strategy,
            "projected_set_win_probability": round(max_set_win_prob * 100, 1),
            "full_analysis": strategy_analysis
        }


# ==========================================
# COACH'S USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Example Scenario: High School Girls Match
    # Late in the set, tight score line.
    coach_simulator = VolleyballMatchSimulator(level="High School", gender="Girls")
    
    # Input your specific player's serving profile data
    # (Easily derived from standard box scores and chart data)
    player_serve_matrix = {
        "Aggressive Jump Float (Zone 1)": {
            "ace_rate": 0.12, 
            "error_rate": 0.06, 
            "opp_perfect_pass_rate": 0.25
        },
        "High-Risk Jump Spin (Zone 6)": {
            "ace_rate": 0.22, 
            "error_rate": 0.26, 
            "opp_perfect_pass_rate": 0.15
        },
        "Safe Standing Float (Target Weak Passer)": {
            "ace_rate": 0.04, 
            "error_rate": 0.02, 
            "opp_perfect_pass_rate": 0.45
        }
    }
    
    # Current Score: Us 22, Them 23
    current_game_state = (22, 23)
    
    decision = coach_simulator.get_optimal_strategy(current_game_state, player_serve_matrix)
    
    # --- Output Report ---
    print(f"--- MATCH DECISION REPORT ({coach_simulator.level} {coach_simulator.gender}) ---")
    print(f"Score: Us {decision['current_score'][0]} | Opponent {decision['current_score'][1]}")
    print(f"RECOMMENDED STRATEGY: {decision['recommended_strategy']}\n")
    print("Strategy Breakdown:")
    for strategy, metrics in decision['full_analysis'].items():
        print(f" -> {strategy}:")
        print(f"    Rally Win Prob: {metrics['point_win_prob']*100}% | Set Win Expectancy: {metrics['set_win_expectancy']*100}%")
    print(f"\nExecuting the recommended strategy gives you a {decision['projected_set_win_probability']}% chance to win the set.")
