import json
import hashlib
import numpy as np
import pandas as pd

CSV_PATH = "spotify_history_cleaned.csv"

BRONZE_THRESHOLD = 5_000
SILVER_THRESHOLD = 15_000
GOLD_THRESHOLD = 30_000

df = pd.read_csv(CSV_PATH)
df["ts"] = pd.to_datetime(df["ts"])
df["date"] = df["ts"].dt.date

TOTAL_PLAYS = len(df)

summary = {
    "total_plays": int(TOTAL_PLAYS),
    "total_hours": round(df["minutes_played"].sum() / 60, 1),
    "unique_artists": int(df["artist_name"].nunique()),
    "unique_tracks": int(df["track_name"].nunique()),
    "unique_albums": int(df["album_name"].nunique()),
    "date_start": str(df["ts"].min().date()),
    "date_end": str(df["ts"].max().date()),
    "years_covered": int(df["year"].nunique()),
}

# TOP 10 ARTISTS BY ENGAGEMENT SCORE
agg = df.groupby("artist_name").agg(
    total_plays=("ms_played", "count"),
    total_minutes=("minutes_played", "sum"),
    completed=("end_type", lambda x: (x == "completed").mean()),
    active=("start_type", lambda x: (x == "active").mean()),
    quick_skip=("is_quick_skip", "mean"),
).reset_index()

qualified = agg[agg["total_plays"] >= 50].copy()
def norm(s):
    return (s - s.min()) / (s.max() - s.min())
qualified["volume_score"] = norm(np.log1p(qualified["total_minutes"]))
qualified["engagement_score"] = (
    0.55 * qualified["volume_score"]
    + 0.25 * qualified["completed"]
    + 0.12 * qualified["active"]
    + 0.08 * (1 - qualified["quick_skip"])
) * 100
top_artists_df = qualified.sort_values("engagement_score", ascending=False).head(10)

top_artists = [
    {
        "rank": i + 1,
        "artist": row["artist_name"],
        "plays": int(row["total_plays"]),
        "hours": round(row["total_minutes"] / 60, 1),
        "engagement_score": round(row["engagement_score"], 1),
        "completion_rate": round(row["completed"] * 100, 1),
    }
    for i, (_, row) in enumerate(top_artists_df.iterrows())
]

track_agg = df.groupby(["track_name", "artist_name"]).agg(
    plays=("ms_played", "count"), minutes=("minutes_played", "sum")
).reset_index().sort_values("minutes", ascending=False).head(10)
top_tracks = [
    {"rank": i + 1, "track": r["track_name"], "artist": r["artist_name"],
     "plays": int(r["plays"]), "minutes": round(r["minutes"], 1)}
    for i, (_, r) in enumerate(track_agg.iterrows())
]

album_agg = df.groupby(["album_name", "artist_name"]).agg(
    plays=("ms_played", "count"), minutes=("minutes_played", "sum")
).reset_index().sort_values("minutes", ascending=False).head(10)
top_albums = [
    {"rank": i + 1, "album": r["album_name"], "artist": r["artist_name"],
     "plays": int(r["plays"]), "hours": round(r["minutes"] / 60, 1)}
    for i, (_, r) in enumerate(album_agg.iterrows())
]

# YEARLY LISTENING HOURS
yearly = df.groupby("year")["minutes_played"].sum() / 60
yearly_hours = [{"year": int(y), "hours": round(h, 1)} for y, h in yearly.items()]

# USER BEHAVIOUR -- 168 hourly buckets, Monday 00:00 -> Sunday 23:00
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
play_counts = df.groupby(["day_of_week", "hour"]).size().reindex(
    pd.MultiIndex.from_product([day_order, range(24)], names=["day_of_week", "hour"]),
    fill_value=0
)
minutes_grid = df.groupby(["day_of_week", "hour"])["minutes_played"].sum().reindex(
    pd.MultiIndex.from_product([day_order, range(24)], names=["day_of_week", "hour"]),
    fill_value=0
)

weekly_rhythm = {
    "labels": [f"{d[:3]} {h:02d}:00" for d, h in play_counts.index],
    "plays": [int(v) for v in play_counts.values],
    "minutes": [round(float(v), 1) for v in minutes_grid.values],
    "day_names": day_order,
    "day_start_index": [i * 24 for i in range(7)],
}

# LISTENING PERSONA
late_night_ratio = df["hour"].isin([0, 1, 2, 3, 4]).mean()
morning_ratio = df["hour"].isin([5, 6, 7, 8, 9]).mean()
weekend_ratio = df["day_of_week"].isin(["Saturday", "Sunday"]).mean()
active_ratio = (df["start_type"] == "active").mean()
completed_ratio = (df["end_type"] == "completed").mean()
quick_skip_ratio = df["is_quick_skip"].mean()
top_artist_share = df["artist_name"].value_counts().iloc[0] / TOTAL_PLAYS
diversity = df["artist_name"].nunique() / TOTAL_PLAYS

PERSONAS = [
    ("Lover Era", top_artist_share > 0.15,
     f"{top_artist_share*100:.1f}% of every play you've ever logged is one artist."),
    ("Night Owl", late_night_ratio > 0.28,
     f"{late_night_ratio*100:.1f}% of your plays happen between midnight and 4am."),
    ("Weekend Wanderer", weekend_ratio > 0.35,
     f"{weekend_ratio*100:.1f}% of your listening happens on weekends."),
    ("Restless Ears", quick_skip_ratio > 0.40,
     f"You skip {quick_skip_ratio*100:.1f}% of tracks within 30 seconds."),
    ("The Curator", active_ratio > 0.60,
     f"{active_ratio*100:.1f}% of your plays were hand-picked, not autoplayed."),
    ("Passenger Mode", active_ratio < 0.30,
     f"Only {active_ratio*100:.1f}% of your plays were actively chosen -- autoplay drives the rest."),
    ("Deep Listener", completed_ratio > 0.60,
     f"{completed_ratio*100:.1f}% of your plays run all the way to the end."),
    ("Explorer", diversity > 0.05,
     f"You've played {int(diversity*TOTAL_PLAYS)} different artists -- wide, restless taste."),
    ("Comfort Loop", diversity < 0.035,
     f"Just {diversity*100:.2f} unique artists per 100 plays -- you return to favorites again and again."),
    ("Early Bird", morning_ratio > 0.25,
     f"{morning_ratio*100:.1f}% of your plays happen before 9am."),
]

matches = [(name, desc) for name, cond, desc in PERSONAS if cond]
if not matches:
    matches = [("Steady Listener", "Your habits are balanced across the day, week, and your library -- no single extreme trait stands out.")]

persona = {
    "primary": {"name": matches[0][0], "reason": matches[0][1]},
    "secondary": {"name": matches[1][0], "reason": matches[1][1]} if len(matches) > 1 else None,
    "stats_used": {
        "late_night_ratio": round(late_night_ratio * 100, 1),
        "morning_ratio": round(morning_ratio * 100, 1),
        "weekend_ratio": round(weekend_ratio * 100, 1),
        "active_ratio": round(active_ratio * 100, 1),
        "completed_ratio": round(completed_ratio * 100, 1),
        "quick_skip_ratio": round(quick_skip_ratio * 100, 1),
        "top_artist_share": round(top_artist_share * 100, 1),
        "avg_repeats_per_artist": round(TOTAL_PLAYS / df["artist_name"].nunique(), 1),
    },
}

def gen_code(artist, salt):
    """Deterministic 6-char alphanumeric code, unique per artist+salt."""
    h = hashlib.sha256(f"BUGBYTES|{artist}|{salt}".encode()).hexdigest()
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I ambiguity
    n = int(h[:12], 16)
    code = ""
    for _ in range(6):
        code += alphabet[n % len(alphabet)]
        n //= len(alphabet)
    return code

TIER_DEFS = [
    {"key": "bronze", "label": "Bronze", "threshold": BRONZE_THRESHOLD,
     "reward": "10% off code for the artist's general merch store"},
    {"key": "silver", "label": "Silver", "threshold": SILVER_THRESHOLD,
     "reward": "Access to an exclusive vinyl colorway (platform collab)"},
    {"key": "gold", "label": "Gold", "threshold": GOLD_THRESHOLD,
     "reward": "Single-use, time-sensitive concert presale code"},
]

superfan_pool = agg[agg["total_plays"] >= 50].sort_values("total_plays", ascending=False).head(14)

superfan_ladder = []
for _, row in superfan_pool.iterrows():
    artist_name = row["artist_name"]
    plays = int(row["total_plays"])

    tiers = []
    current_tier_index = -1
    for i, t in enumerate(TIER_DEFS):
        unlocked = plays >= t["threshold"]
        if unlocked:
            current_tier_index = i
        tiers.append({
            "key": t["key"],
            "label": t["label"],
            "threshold": t["threshold"],
            "reward": t["reward"],
            "unlocked": unlocked,
            "code": gen_code(artist_name, t["key"]) if unlocked else None,
        })

    if current_tier_index + 1 < len(TIER_DEFS):
        next_tier = TIER_DEFS[current_tier_index + 1]
        remaining = int(next_tier["threshold"] - plays)
        lower_bound = TIER_DEFS[current_tier_index]["threshold"] if current_tier_index >= 0 else 0
        span = next_tier["threshold"] - lower_bound
        progress_pct = round(max(0.0, min(1.0, (plays - lower_bound) / span)) * 100, 1)
        next_tier_label = next_tier["label"]
    else:
        remaining = 0
        progress_pct = 100.0
        next_tier_label = None

    superfan_ladder.append({
        "artist": artist_name,
        "plays": plays,
        "hours": round(row["total_minutes"] / 60, 1),
        "current_tier": TIER_DEFS[current_tier_index]["label"] if current_tier_index >= 0 else None,
        "next_tier": next_tier_label,
        "plays_to_next_tier": remaining,
        "progress_pct_to_next": progress_pct,
        "tiers": tiers,
    })

data = {
    "summary": summary,
    "top_artists": top_artists,
    "top_tracks": top_tracks,
    "top_albums": top_albums,
    "yearly_hours": yearly_hours,
    "weekly_rhythm": weekly_rhythm,
    "persona": persona,
    "superfan_ladder": superfan_ladder,
}

with open("wrapped_data.json", "w") as f:
    json.dump(data, f, indent=2)

print("Persona:", persona["primary"]["name"], "+", persona["secondary"]["name"] if persona["secondary"] else None)
print("JSON size (KB):", round(len(json.dumps(data)) / 1024, 1))
print("Top artist:", superfan_ladder[0]["artist"], superfan_ladder[0]["plays"], "-> tier:", superfan_ladder[0]["current_tier"])
for s in superfan_ladder[:5]:
    print(f"  {s['artist']:<20} plays={s['plays']:<7} tier={s['current_tier']} next={s['next_tier']} progress={s['progress_pct_to_next']}%")
