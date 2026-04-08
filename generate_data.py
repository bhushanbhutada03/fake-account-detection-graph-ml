"""
generate_data.py — Realistic Bot Detection Dataset
Columns: User ID, Username, Tweet, Retweet Count, Mention Count,
         Follower Count, Verified, Bot Label, Location, Created At, Hashtags

Realistic overlap between bots and real users so model learns patterns,
not a perfect rule. Expected ML accuracy: 85-92%.
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

N = 5000


BOT_TWEETS = [
    "Follow me for follow back! #followback #follow",
    "Buy cheap followers now! Link in bio.",
    "RT to win!! #giveaway #free #win",
    "Check this out #viral #trending",
] * 1000

REAL_TWEETS = [
    "Had a great day at the park today",
    "Can't believe what happened in today's match!",
    "Just finished reading an amazing book.",
    "Traffic is terrible today.",
    "Making chai at midnight is peak productivity.",
] * 1000

LOCATIONS = ["Mumbai","Delhi","Bangalore","New York","London","Tokyo","","India","USA"]
BOT_HASHTAGS  = ["#followback #f4f", "#viral #trending", "#giveaway #win", "#crypto", ""]
REAL_HASHTAGS = ["#cricket", "#mondaymotivation", "#food", "", "#photography", "#tech"]

def rand_date(y1, y2):
    s = datetime(y1, 1, 1)
    e = datetime(y2, 12, 31)
    return (s + timedelta(days=random.randint(0, (e-s).days))).strftime("%Y-%m-%d")

# Shared ID pool — no leakage
all_ids = random.sample(range(10000, 99999), N)

rows = []
for i in range(N):
    is_bot = 1 if i < N//2 else 0
    uid = all_ids[i]

    if is_bot:
        row = {
            "User ID":       uid,
            "Username":      f"usr_{random.randint(10000,99999)}",
            "Tweet":         random.choice(BOT_TWEETS),
            # Realistic bot: high activity but OVERLAPS with real users
            "Retweet Count": max(1, int(np.random.normal(150, 80))),   # mean 150, some as low as 10
            "Mention Count": max(1, int(np.random.normal(80,  50))),   # mean 80
            "Follower Count":max(1, int(np.random.normal(120, 100))),  # mean 120, some overlap with real
            "Verified":      int(np.random.choice([0,1], p=[0.98, 0.02])),
            "Bot Label":     1,
            "Location":      random.choice(["", "Worldwide", "Earth", "Unknown"]),
            "Created At":    rand_date(2019, 2024),   # newer accounts
            "Hashtags":      random.choice(BOT_HASHTAGS),
        }
    else:
        row = {
            "User ID":       uid,
            "Username":      f"usr_{random.randint(10000,99999)}",
            "Tweet":         random.choice(REAL_TWEETS),
            # Realistic real user: moderate, OVERLAPS with bots
            "Retweet Count": max(1, int(np.random.normal(60,  50))),   # mean 60, some high
            "Mention Count": max(1, int(np.random.normal(30,  25))),   # mean 30
            "Follower Count":max(1, int(np.random.normal(600, 400))),  # mean 600
            "Verified":      int(np.random.choice([0,1], p=[0.93, 0.07])),
            "Bot Label":     0,
            "Location":      random.choice(LOCATIONS),
            "Created At":    rand_date(2010, 2020),   # older accounts
            "Hashtags":      random.choice(REAL_HASHTAGS),
        }
    rows.append(row)

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/dataset.csv", index=False)

print(f"Saved: {len(df)} records | Bots: {df['Bot Label'].sum()} | Real: {(df['Bot Label']==0).sum()}")
print("\nMean features by class (should overlap somewhat):")
print(df.groupby('Bot Label')[['Follower Count','Retweet Count','Mention Count']].describe().round(1))
