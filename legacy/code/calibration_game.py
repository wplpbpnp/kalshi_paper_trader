#!/usr/bin/env python3
"""
Probability Calibration Game

Practice estimating probabilities. Get scored on how well-calibrated you are.
Good calibration = when you say 70%, things happen 70% of the time.
"""

import random
import json
import os
from datetime import datetime

SAVE_FILE = "calibration_scores.json"

# Trivia questions with known probabilities
QUESTIONS = [
    # Geography/Demographics
    {"q": "What % of the world's population lives in Asia?", "a": 60, "source": "~60% (4.7B of 8B)"},
    {"q": "What % of Earth's surface is covered by water?", "a": 71, "source": "71%"},
    {"q": "What % of the world's population lives in the Northern Hemisphere?", "a": 90, "source": "~90%"},
    {"q": "What % of Earth's land is in the Northern Hemisphere?", "a": 68, "source": "~68%"},
    {"q": "What % of the world's population lives in cities?", "a": 57, "source": "~57% (2024)"},

    # US specific
    {"q": "What % of Americans have a bachelor's degree or higher?", "a": 33, "source": "~33% (2023)"},
    {"q": "What % of Americans are obese (BMI > 30)?", "a": 42, "source": "~42% (CDC)"},
    {"q": "What % of US electricity comes from renewable sources?", "a": 22, "source": "~22% (2023)"},
    {"q": "What % of Americans own a gun?", "a": 32, "source": "~32% (Gallup)"},
    {"q": "What % of Americans identify as LGBT?", "a": 7, "source": "~7.6% (Gallup 2024)"},

    # Science/Nature
    {"q": "What % of the human body is water?", "a": 60, "source": "~60%"},
    {"q": "What % of Earth's atmosphere is nitrogen?", "a": 78, "source": "78%"},
    {"q": "What % of the ocean has been explored?", "a": 5, "source": "~5%"},
    {"q": "What % of species that have ever existed are now extinct?", "a": 99, "source": "~99%"},
    {"q": "What % of your DNA do you share with a banana?", "a": 60, "source": "~60%"},

    # Economics/Business
    {"q": "What % of startups fail within 5 years?", "a": 50, "source": "~50%"},
    {"q": "What % of day traders lose money?", "a": 90, "source": "~90% (various studies)"},
    {"q": "What % of the S&P 500 is tech stocks?", "a": 30, "source": "~30% (2024)"},
    {"q": "What % of US GDP is consumer spending?", "a": 68, "source": "~68%"},
    {"q": "What % of global wealth is held by the top 1%?", "a": 46, "source": "~46% (Credit Suisse)"},

    # Sports/Games
    {"q": "What % of NFL players go bankrupt within 12 years of retirement?", "a": 78, "source": "~78% (SI)"},
    {"q": "What % of chess games at grandmaster level end in a draw?", "a": 50, "source": "~50%"},
    {"q": "What % of penalty kicks in soccer are scored?", "a": 75, "source": "~75-80%"},

    # Technology
    {"q": "What % of the world has internet access?", "a": 66, "source": "~66% (2024)"},
    {"q": "What % of emails sent are spam?", "a": 45, "source": "~45% (2024)"},
    {"q": "What % of Google searches result in no click?", "a": 65, "source": "~65%"},

    # Psychology/Behavior
    {"q": "What % of people think they're above average drivers?", "a": 80, "source": "~80% (Svenson 1981)"},
    {"q": "What % of New Year's resolutions fail by February?", "a": 80, "source": "~80%"},
    {"q": "What % of communication is non-verbal?", "a": 70, "source": "~70% (Mehrabian, disputed)"},

    # History/Misc
    {"q": "What % of medieval Europeans died from the Black Death?", "a": 33, "source": "~30-50% (usually cited as 1/3)"},
    {"q": "What % of the universe is dark matter + dark energy?", "a": 95, "source": "~95%"},
    {"q": "What % of published psychology studies replicate?", "a": 40, "source": "~40% (Replication Crisis)"},
]

# Probability estimation scenarios (Fermi-style)
SCENARIOS = [
    {"q": "You flip 10 fair coins. Probability of getting exactly 5 heads?", "a": 25, "hint": "Binomial: C(10,5)/2^10"},
    {"q": "Roll two dice. Probability the sum is 7?", "a": 17, "hint": "6/36 = 16.67%"},
    {"q": "Draw 2 cards from a deck. Probability both are hearts?", "a": 6, "hint": "(13/52)*(12/51) ≈ 5.9%"},
    {"q": "Birthday problem: 23 people in a room. Probability 2+ share a birthday?", "a": 51, "hint": "Famous result: ~50.7%"},
    {"q": "You have a 60% edge per trade. Probability of 5 wins in a row?", "a": 8, "hint": "0.6^5 = 7.8%"},
    {"q": "10% chance of rain each day. Probability it rains at least once in 7 days?", "a": 52, "hint": "1 - 0.9^7 = 52.2%"},
    {"q": "If a test is 99% accurate and 1% of people have the disease, probability you have it given positive test?", "a": 50, "hint": "Base rate fallacy: ~50%"},
    {"q": "Fair coin, 4 flips. Probability of at least 3 heads?", "a": 31, "hint": "(C(4,3) + C(4,4))/16 = 31.25%"},
    {"q": "Roll a die until you get a 6. Probability it takes exactly 3 rolls?", "a": 12, "hint": "(5/6)^2 * (1/6) = 11.6%"},
    {"q": "Monty Hall: you switch doors. Probability of winning?", "a": 67, "hint": "2/3 ≈ 66.7%"},
]


def load_scores():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, 'r') as f:
            return json.load(f)
    return {"games": [], "total_questions": 0, "calibration_data": {}}


def save_scores(data):
    with open(SAVE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def calculate_brier_score(predictions):
    """Lower is better. 0 = perfect, 0.25 = random guessing on 50/50"""
    if not predictions:
        return None
    total = 0
    for pred, actual in predictions:
        # Normalize to 0-1
        p = pred / 100
        a = actual / 100
        total += (p - a) ** 2
    return total / len(predictions)


def calculate_calibration_buckets(data):
    """Group predictions into buckets and compare to actuals"""
    buckets = {i: {"predictions": [], "actuals": []} for i in range(0, 101, 10)}

    for pred, actual in data:
        bucket = (pred // 10) * 10
        bucket = min(bucket, 90)  # Cap at 90-100 bucket
        buckets[bucket]["predictions"].append(pred)
        buckets[bucket]["actuals"].append(actual)

    return buckets


def print_calibration_report(all_data):
    """Show how well-calibrated the player is"""
    if not all_data:
        print("\nNo data yet!")
        return

    buckets = calculate_calibration_buckets(all_data)

    print("\n" + "="*60)
    print("CALIBRATION REPORT")
    print("="*60)
    print(f"{'Your Estimate':<15} {'Avg Actual':<15} {'Diff':<10} {'n':<5}")
    print("-"*60)

    for bucket in sorted(buckets.keys()):
        data = buckets[bucket]
        if data["actuals"]:
            avg_pred = sum(data["predictions"]) / len(data["predictions"])
            avg_actual = sum(data["actuals"]) / len(data["actuals"])
            diff = avg_pred - avg_actual
            n = len(data["actuals"])

            # Color coding would be nice but let's keep it simple
            sign = "+" if diff > 0 else ""
            print(f"{bucket:>3}-{bucket+10:<3}%        {avg_actual:>6.1f}%        {sign}{diff:>5.1f}%    {n:<5}")

    brier = calculate_brier_score(all_data)
    print("-"*60)
    print(f"Brier Score: {brier:.4f}")
    print("  (0 = perfect, 0.25 = coin flip guessing, lower is better)")

    # Interpretation
    if brier < 0.1:
        print("  → Excellent calibration!")
    elif brier < 0.15:
        print("  → Good calibration")
    elif brier < 0.2:
        print("  → Decent, room for improvement")
    else:
        print("  → Needs work - you're over/underconfident")


def play_trivia_round(n_questions=10):
    """Play a round of probability trivia"""
    questions = random.sample(QUESTIONS, min(n_questions, len(QUESTIONS)))
    results = []

    print("\n" + "="*60)
    print("PROBABILITY TRIVIA")
    print("Estimate the percentage (0-100)")
    print("="*60 + "\n")

    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q['q']}")

        while True:
            try:
                guess = input("Your estimate (0-100): ").strip()
                if guess.lower() == 'q':
                    return results
                guess = float(guess.replace('%', ''))
                if 0 <= guess <= 100:
                    break
                print("Please enter a number between 0 and 100")
            except ValueError:
                print("Please enter a valid number")

        actual = q['a']
        diff = abs(guess - actual)

        print(f"  Actual: {actual}% ({q['source']})")
        print(f"  You were off by: {diff:.1f}%")

        if diff <= 5:
            print("  ★ Excellent!")
        elif diff <= 10:
            print("  ✓ Good")
        elif diff <= 20:
            print("  ~ Okay")
        else:
            print("  ✗ Way off")

        print()
        results.append((guess, actual))

    return results


def play_scenario_round(n_questions=5):
    """Play probability scenarios (more mathematical)"""
    questions = random.sample(SCENARIOS, min(n_questions, len(SCENARIOS)))
    results = []

    print("\n" + "="*60)
    print("PROBABILITY SCENARIOS")
    print("Calculate or estimate the probability (0-100)")
    print("="*60 + "\n")

    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q['q']}")

        while True:
            try:
                guess = input("Your estimate (0-100): ").strip()
                if guess.lower() == 'q':
                    return results
                guess = float(guess.replace('%', ''))
                if 0 <= guess <= 100:
                    break
                print("Please enter a number between 0 and 100")
            except ValueError:
                print("Please enter a valid number")

        actual = q['a']
        diff = abs(guess - actual)

        print(f"  Actual: {actual}%")
        print(f"  Hint: {q['hint']}")
        print(f"  You were off by: {diff:.1f}%")

        if diff <= 3:
            print("  ★ Nailed it!")
        elif diff <= 10:
            print("  ✓ Good intuition")
        elif diff <= 20:
            print("  ~ In the ballpark")
        else:
            print("  ✗ Review this one")

        print()
        results.append((guess, actual))

    return results


def play_ticker_prediction(n_rounds=10):
    """
    Predict the next move of a synthetic ticker.
    The ticker has a hidden probability of going up.
    Your job: estimate that probability from the pattern.
    """
    results = []

    print("\n" + "="*60)
    print("TICKER PREDICTION")
    print("A synthetic ticker will show you its recent moves.")
    print("Estimate the probability (0-100) that the NEXT move is UP.")
    print("The ticker has a hidden 'true' probability that doesn't change.")
    print("="*60 + "\n")

    for round_num in range(1, n_rounds + 1):
        # Generate a random true probability (biased away from 50 to make it interesting)
        # Use a distribution that favors more extreme probabilities
        if random.random() < 0.5:
            true_prob = random.randint(20, 45)  # Bearish
        else:
            true_prob = random.randint(55, 80)  # Bullish

        # Generate history length (8-20 moves)
        history_len = random.randint(8, 20)

        # Generate the history based on true probability
        history = []
        for _ in range(history_len):
            if random.randint(1, 100) <= true_prob:
                history.append("▲")
            else:
                history.append("▼")

        # Display
        print(f"Round {round_num}/{n_rounds}")
        print(f"Ticker $RNG: {''.join(history)}")

        # Count actual ups for reference (player can do this too)
        actual_ups = history.count("▲")

        # Get player's estimate
        while True:
            try:
                guess = input("P(next move is UP)? (0-100): ").strip()
                if guess.lower() == 'q':
                    return results
                guess = float(guess.replace('%', ''))
                if 0 <= guess <= 100:
                    break
                print("Please enter a number between 0 and 100")
            except ValueError:
                print("Please enter a valid number")

        # Generate the next move
        next_up = random.randint(1, 100) <= true_prob
        next_move = "▲ UP" if next_up else "▼ DOWN"

        # Score
        diff = abs(guess - true_prob)

        print(f"  Next move: {next_move}")
        print(f"  True P(UP): {true_prob}%")
        print(f"  History had: {actual_ups}/{history_len} ups ({100*actual_ups/history_len:.0f}%)")
        print(f"  Your estimate was off by: {diff:.0f}%")

        if diff <= 5:
            print("  ★ Excellent read!")
        elif diff <= 10:
            print("  ✓ Good")
        elif diff <= 15:
            print("  ~ Okay")
        else:
            print("  ✗ Misread the ticker")

        print()

        # For calibration tracking, we compare guess to true_prob
        results.append((guess, true_prob))

    # End of round summary
    if results:
        avg_error = sum(abs(g - a) for g, a in results) / len(results)
        print(f"\nAverage error: {avg_error:.1f}%")

        # Check for biases
        over_estimates = sum(1 for g, a in results if g > a)
        under_estimates = sum(1 for g, a in results if g < a)

        if over_estimates > len(results) * 0.7:
            print("Tendency: You're OVERESTIMATING P(UP) - seeing too much signal")
        elif under_estimates > len(results) * 0.7:
            print("Tendency: You're UNDERESTIMATING P(UP) - too pessimistic")
        else:
            print("Tendency: Balanced")

        # Check for gambler's fallacy
        print("\nTip: The true probability is CONSTANT within each ticker.")
        print("Recent streaks don't change the underlying odds.")

    return results


def play_streak_detection(n_rounds=10):
    """
    Harder mode: some tickers are random (50/50), some are biased.
    Your job: detect which is which and estimate the bias.
    """
    results = []

    print("\n" + "="*60)
    print("STREAK DETECTION (Hard Mode)")
    print("Some tickers are FAIR (50/50), some are BIASED.")
    print("Estimate P(UP) - if you think it's fair, say 50.")
    print("="*60 + "\n")

    for round_num in range(1, n_rounds + 1):
        # 40% chance of fair ticker, 60% chance of biased
        if random.random() < 0.4:
            true_prob = 50  # Fair
            ticker_type = "FAIR"
        else:
            # Biased - but how biased?
            if random.random() < 0.5:
                true_prob = random.randint(25, 40)  # Bearish
            else:
                true_prob = random.randint(60, 75)  # Bullish
            ticker_type = "BIASED"

        # Longer history for this mode
        history_len = random.randint(15, 30)

        # Generate history
        history = []
        for _ in range(history_len):
            if random.randint(1, 100) <= true_prob:
                history.append("▲")
            else:
                history.append("▼")

        print(f"Round {round_num}/{n_rounds}")
        # Display in chunks for readability
        hist_str = ''.join(history)
        if len(hist_str) > 25:
            print(f"Ticker $???:")
            print(f"  {hist_str[:25]}")
            print(f"  {hist_str[25:]}")
        else:
            print(f"Ticker $???: {hist_str}")

        while True:
            try:
                guess = input("P(next move is UP)? (0-100): ").strip()
                if guess.lower() == 'q':
                    return results
                guess = float(guess.replace('%', ''))
                if 0 <= guess <= 100:
                    break
                print("Please enter a number between 0 and 100")
            except ValueError:
                print("Please enter a valid number")

        # Generate next move
        next_up = random.randint(1, 100) <= true_prob
        next_move = "▲ UP" if next_up else "▼ DOWN"

        diff = abs(guess - true_prob)
        actual_ups = history.count("▲")

        print(f"  Next move: {next_move}")
        print(f"  Ticker was: {ticker_type} (true P(UP) = {true_prob}%)")
        print(f"  History: {actual_ups}/{history_len} ups ({100*actual_ups/history_len:.0f}%)")
        print(f"  Your error: {diff:.0f}%")

        # Did they correctly identify fair vs biased?
        guessed_fair = 45 <= guess <= 55
        was_fair = ticker_type == "FAIR"

        if guessed_fair == was_fair:
            print("  ✓ Correctly identified fair/biased!")
        else:
            if was_fair and not guessed_fair:
                print("  ✗ Saw pattern in randomness (apophenia)")
            else:
                print("  ✗ Missed the real bias")

        print()
        results.append((guess, true_prob))

    if results:
        avg_error = sum(abs(g - a) for g, a in results) / len(results)

        # Count correct fair/biased detections
        correct_detections = sum(
            1 for g, a in results
            if (45 <= g <= 55) == (a == 50)
        )

        print(f"\nAverage error: {avg_error:.1f}%")
        print(f"Correct fair/biased detection: {correct_detections}/{len(results)}")

        if avg_error < 10:
            print("Excellent pattern detection!")
        elif avg_error < 15:
            print("Good intuition for bias")
        else:
            print("Practice distinguishing signal from noise")

    return results


def play_confidence_intervals():
    """Practice giving 90% confidence intervals"""
    questions = [
        {"q": "Height of the Eiffel Tower in meters", "a": 330},
        {"q": "Year the Roman Empire fell (Western)", "a": 476},
        {"q": "Population of Tokyo metro area in millions", "a": 37},
        {"q": "Distance from Earth to Moon in km", "a": 384400},
        {"q": "Number of bones in the human body", "a": 206},
        {"q": "Year the first iPhone was released", "a": 2007},
        {"q": "Speed of light in km/s (rounded)", "a": 300000},
        {"q": "Number of countries in the UN", "a": 193},
        {"q": "Bitcoin's all-time high price in USD (2024)", "a": 73000},
        {"q": "Age of the universe in billions of years", "a": 13.8},
    ]

    questions = random.sample(questions, min(5, len(questions)))
    hits = 0

    print("\n" + "="*60)
    print("90% CONFIDENCE INTERVALS")
    print("Give a range you're 90% sure contains the true value")
    print("(You should hit ~9/10 if well-calibrated)")
    print("="*60 + "\n")

    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q['q']}")

        try:
            low = input("  Low end: ").strip()
            if low.lower() == 'q':
                break
            low = float(low.replace(',', ''))

            high = input("  High end: ").strip()
            if high.lower() == 'q':
                break
            high = float(high.replace(',', ''))
        except ValueError:
            print("  Invalid input, skipping...")
            continue

        actual = q['a']
        hit = low <= actual <= high

        if hit:
            hits += 1
            print(f"  ✓ Actual: {actual:,} - You got it!")
        else:
            print(f"  ✗ Actual: {actual:,} - Outside your range")

        # Check if interval is too wide
        if high > 0 and (high - low) / actual > 2:
            print("  (Your interval was very wide)")

        print()

    n = len(questions)
    print(f"\nResults: {hits}/{n} hits ({100*hits/n:.0f}%)")
    if hits/n > 0.95:
        print("You might be underconfident - tighten your intervals")
    elif hits/n < 0.85:
        print("You might be overconfident - widen your intervals")
    else:
        print("Good calibration on confidence intervals!")


def main():
    scores = load_scores()
    all_calibration_data = scores.get("calibration_data", {})

    # Convert stored data back to list of tuples
    all_results = []
    for pred_str, actuals in all_calibration_data.items():
        pred = float(pred_str)
        for actual in actuals:
            all_results.append((pred, actual))

    print("\n" + "="*60)
    print("  PROBABILITY CALIBRATION TRAINER")
    print("  Practice estimating probabilities accurately")
    print("="*60)

    while True:
        print("\nModes:")
        print("  1. Trivia - Estimate real-world probabilities")
        print("  2. Scenarios - Calculate probability puzzles")
        print("  3. Confidence Intervals - Give 90% ranges")
        print("  4. Ticker Prediction - Read synthetic price action")
        print("  5. Streak Detection (Hard) - Fair vs biased tickers")
        print("  6. View Calibration Report")
        print("  7. Reset Data")
        print("  q. Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == '1':
            results = play_trivia_round()
            all_results.extend(results)
        elif choice == '2':
            results = play_scenario_round()
            all_results.extend(results)
        elif choice == '3':
            play_confidence_intervals()
        elif choice == '4':
            results = play_ticker_prediction()
            all_results.extend(results)
        elif choice == '5':
            results = play_streak_detection()
            all_results.extend(results)
        elif choice == '6':
            print_calibration_report(all_results)
        elif choice == '7':
            confirm = input("Reset all calibration data? (y/n): ")
            if confirm.lower() == 'y':
                all_results = []
                all_calibration_data = {}
                print("Data reset!")
        elif choice == 'q':
            break
        else:
            print("Invalid choice")

    # Save results
    # Convert to storable format
    calibration_data = {}
    for pred, actual in all_results:
        key = str(pred)
        if key not in calibration_data:
            calibration_data[key] = []
        calibration_data[key].append(actual)

    scores["calibration_data"] = calibration_data
    scores["total_questions"] = len(all_results)
    scores["last_played"] = datetime.now().isoformat()
    save_scores(scores)

    print(f"\nSaved {len(all_results)} total predictions")
    print("Your calibration data persists between sessions")


if __name__ == "__main__":
    main()
