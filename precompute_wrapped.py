"""
Precompute every number the Wrapped-style site needs, as one compact JSON file.
No raw play-by-play rows are shipped to the browser -- only aggregates.
"""

import json
import numpy as np
import pandas as pd

df = pd.read_csv("/mnt/user-data/outputs/spotify_history_cleaned.csv")
df["ts"] = pd.to_datetime(df["ts"])
df["date"] = df["ts"].dt.date

TOTAL_PLAYS = len(df)

# ------------------------------------------------------------------
# 1. TOP-LINE SUMMARY
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 2. TOP 10 ARTISTS BY ENGAGEMENT SCORE  (same formula as before)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 3. TOP 10 TRACKS / ALBUMS (by total minutes played)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 4. YEARLY LISTENING HOURS
# ------------------------------------------------------------------
yearly = df.groupby("year")["minutes_played"].sum() / 60
yearly_hours = [{"year": int(y), "hours": round(h, 1)} for y, h in yearly.items()]

# ------------------------------------------------------------------
# 5. USER BEHAVIOUR -- one continuous line across the full week
#    (Monday 00:00 -> Sunday 23:00, 168 points, raw play counts,
#    matching the single-line/filled-area style of the reference chart)
# ------------------------------------------------------------------
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
play_counts = df.groupby(["day_of_week", "hour"]).size().reindex(
    pd.MultiIndex.from_product([day_order, range(24)], names=["day_of_week", "hour"]),
    fill_value=0
)

weekly_rhythm = {
    "labels": [f"{d[:3]} {h:02d}:00" for d, h in play_counts.index],
    "plays": [int(v) for v in play_counts.values],
    "day_names": day_order,
    "day_start_index": [i * 24 for i in range(7)],  # index where each day begins, for gridlines
}

# ------------------------------------------------------------------
# 6. LISTENING PERSONA (rule-based, computed from real behaviour)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 7. "STREAM THRESHOLD" MERCH UNLOCK  (personal play-count based)
#    NOTE: uses the user's OWN play count per artist as a stand-in for
#    a real cross-platform stream count, since this dataset is a single
#    listener's history. See write-up for the assumption this makes.
# ------------------------------------------------------------------
import hashlib

def promo_code(artist, album):
    """Deterministic 6-char alphanumeric code, unique per artist+album."""
    h = hashlib.sha256(f"BUGBYTES|{artist}|{album}".encode()).hexdigest()
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I ambiguity
    n = int(h[:12], 16)
    code = ""
    for _ in range(6):
        code += alphabet[n % len(alphabet)]
        n //= len(alphabet)
    return code

unlock_candidates = []
for _, row in agg[agg["total_plays"] >= 300].sort_values("total_plays", ascending=False).head(12).iterrows():
    artist_name = row["artist_name"]
    artist_albums = (
        df[df["artist_name"] == artist_name]
        .groupby("album_name")["minutes_played"].sum()
        .sort_values(ascending=False)
        .head(4)
    )
    albums_with_codes = [
        {"album": album, "hours": round(minutes / 60, 1), "promo_code": promo_code(artist_name, album)}
        for album, minutes in artist_albums.items()
    ]
    unlock_candidates.append({
        "artist": artist_name,
        "plays": int(row["total_plays"]),
        "hours": round(row["total_minutes"] / 60, 1),
        "albums": albums_with_codes,
    })


# ------------------------------------------------------------------
# 8. REAL "listeners of X also played Y" -- same-day co-occurrence
#    (rule-based heuristic on this listener's own data, not a trained
#    cross-user recommender -- see write-up)
# ------------------------------------------------------------------
co_occurrence = {}
for artist in [a["artist"] for a in top_artists[:5]]:
    days_with = df.loc[df["artist_name"] == artist, "date"].unique()
    same_day = df[df["date"].isin(days_with) & (df["artist_name"] != artist)]
    top3 = same_day["artist_name"].value_counts().head(3).index.tolist()
    co_occurrence[artist] = top3

# ------------------------------------------------------------------
# WRITE OUT
# ------------------------------------------------------------------
data = {
    "summary": summary,
    "top_artists": top_artists,
    "top_tracks": top_tracks,
    "top_albums": top_albums,
    "yearly_hours": yearly_hours,
    "weekly_rhythm": weekly_rhythm,
    "persona": persona,
    "unlock_candidates": unlock_candidates,
    "co_occurrence": co_occurrence,
}

with open("/home/claude/wrapped_data.json", "w") as f:
    json.dump(data, f, indent=2)

print("Persona:", persona["primary"]["name"], "+", persona["secondary"]["name"] if persona["secondary"] else None)
print("JSON size (KB):", round(len(json.dumps(data)) / 1024, 1))
print("Top artist for unlock demo:", unlock_candidates[0]["artist"], unlock_candidates[0]["plays"])
print("Sample album promo codes:", unlock_candidates[0]["albums"])
