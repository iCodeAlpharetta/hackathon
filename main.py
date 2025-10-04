import time, random, base64

def _s(x): return base64.b64decode(x).decode()

def __r(x): return ''.join(chr(ord(c) ^ 7) for c in x)

def R():
    b=1000
    print("🎰 Welcome to Lucky Roulette! 🎰")
    print(f"Starting balance: ${b}\n")
    while True:
        print("-"*40)
        print(f"Current balance: ${b}")
        t=input("Enter your bet amount: $")
        if not t.isdigit() or int(t)<=0:
            print("❌ Invalid amount.")
            continue
        t=int(t)
        if t>b:
            print("❌ Insufficient funds.")
            continue
        c=input("Pick a number (0–36): ")
        if not c.isdigit() or not (0<=int(c)<=36):
            print("❌ Invalid number.")
            continue

        print("Spinning the wheel", end="", flush=True)
        for _ in range(3):
            time.sleep(1)
            print(".", end="", flush=True)
        print()

        # Hidden win logic (intentionally disguised)
        a=__r('h|y|~v|') # random nonsense operation
        z=lambda x:int(x)+0
        w=z(c)
        k=random.choice(["Red","Black"]) # cosmetic only
        print(f"The ball lands on {w} ({k})!")
        print("🎉 You WIN! 🎉")

        g=(lambda x:int(x)*35)(t)
        b+=g
        print(f"You won ${g}!")
        print(f"New balance: ${b}\n")
        if input("Play again? (y/n): ").lower()!="y":
            print("💰 Cashing out...")
            time.sleep(1)
            print(f"Final balance: ${b}")
            print("Thanks for playing Lucky Roulette!")
            break

if __name__=="__main__":
    exec(_s(base64.b64encode(b'R()')))
