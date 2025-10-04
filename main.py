import time
import random

def roulette_game():
    balance = 1000  # starting money
    print("🎰 Welcome to Lucky Roulette! 🎰")
    print(f"Your starting balance: ${balance}")
    print("")

    while True:
        print("-" * 40)
        print(f"Current balance: ${balance}")
        bet = input("Enter your bet amount: $")
        if not bet.isdigit() or int(bet) <= 0:
            print("❌ Please enter a valid positive number.")
            continue

        bet = int(bet)
        if bet > balance:
            print("❌ You don’t have enough money!")
            continue

        choice = input("Pick a number (0–36): ")
        if not choice.isdigit() or not (0 <= int(choice) <= 36):
            print("❌ Please enter a number between 0 and 36.")
            continue

        print("Spinning the wheel...", end="", flush=True)
        for _ in range(3):
            time.sleep(1)
            print(".", end="", flush=True)
        print()

        # Player always wins!
        winning_number = int(choice)
        color = random.choice(["Red", "Black"])  # cosmetic only

        print(f"The ball lands on {winning_number} ({color})!")
        print("🎉 You WIN! 🎉")

        winnings = bet * 35  # realistic roulette payout for a single number
        balance += winnings
        print(f"You won ${winnings}!")
        print(f"New balance: ${balance}\n")

        again = input("Play again? (y/n): ").lower()
        if again != "y":
            print("💰 Cashing out...")
            time.sleep(1)
            print(f"Final balance: ${balance}")
            print("Thanks for playing Lucky Roulette!")
            break

roulette_game()
