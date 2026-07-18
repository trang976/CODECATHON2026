import pandas as pd
import numpy as np
import json

df = pd.read_csv("spotify_history_clean.csv")
df["ts"] = pd.to_datetime(df["ts"])

df["completed"] = (df["reason_end"] == "trackdone").astype(int)

artist_stats = (
    df.groupby("artist_name")
    .agg(
        play_count=("ms_played", "count"),
        total_ms=("ms_played", "sum"),
        avg_ms=("ms_played", "mean"),
        skip_rate=("skipped", "mean"),
        completion_rate=("completed", "mean"),
        first_play=("ts", "min"),
        last_play=("ts", "max"),
    )
    .reset_index()
)

artist_stats["total_hours"] = artist_stats["total_ms"] / 3_600_000
artist_stats["days_active"] = (
    artist_stats["last_play"] - artist_stats["first_play"]
).dt.days + 1


# ---------- 2. ENGAGEMENT SCORE ----------
# Normalize components 0-1, then weighted blend.
def norm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


artist_stats["norm_playcount"] = norm(artist_stats["play_count"])
artist_stats["norm_hours"] = norm(artist_stats["total_hours"])
artist_stats["norm_completion"] = artist_stats["completion_rate"]
artist_stats["norm_lowskip"] = 1 - artist_stats["skip_rate"]

# Recency boost: plays in the last 12 months of the dataset weigh more
max_date = df["ts"].max()
recent_cutoff = max_date - pd.Timedelta(days=365)
recent_counts = df[df["ts"] >= recent_cutoff].groupby("artist_name").size()
artist_stats["recent_play_count"] = (
    artist_stats["artist_name"].map(recent_counts).fillna(0)
)
artist_stats["norm_recency"] = norm(artist_stats["recent_play_count"])

artist_stats["engagement_score"] = (
    0.35 * artist_stats["norm_hours"]
    + 0.25 * artist_stats["norm_playcount"]
    + 0.20 * artist_stats["norm_completion"]
    + 0.10 * artist_stats["norm_lowskip"]
    + 0.10 * artist_stats["norm_recency"]
) * 100

artist_stats = artist_stats.sort_values("engagement_score", ascending=False)


# ---------- 3. PERKS / THRESHOLD TIER SYSTEM ----------
def tier(plays):
    if plays >= 30000:
        return "Platinum"
    if plays >= 5000:
        return "Gold"
    if plays >= 1000:
        return "Silver"
    if plays >= 100:
        return "Bronze"
    return "Listener"


# Since raw play_count won't hit 30k for a single artist realistically here,
# also compute a normalized "stream score" (ms_played-based, comparable to
# a streaming-platform's own stream count which counts a play at >=30s).
df["counts_as_stream"] = (df["ms_played"] >= 30000).astype(
    int
)  # Spotify's real 30s rule
stream_counts = df.groupby("artist_name")["counts_as_stream"].sum()
artist_stats["verified_streams"] = (
    artist_stats["artist_name"].map(stream_counts).fillna(0).astype(int)
)
artist_stats["tier"] = artist_stats["verified_streams"].apply(tier)

perk_map = {
    "Listener": [],
    "Bronze": ["Early access to new releases"],
    "Silver": ["Early access to new releases", "Discount code for merch"],
    "Gold": [
        "Early access to new releases",
        "Discount code for merch",
        "Presale concert tickets",
    ],
    "Platinum": [
        "Early access to new releases",
        "Discount code for merch",
        "Presale concert tickets",
        "Exclusive vinyl / signed item eligibility",
    ],
}
artist_stats["perks"] = artist_stats["tier"].map(perk_map)

# ---------- 4. TIME PATTERNS ----------
hourly = df.groupby("hour")["ms_played"].sum().reindex(range(24), fill_value=0)
dow_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
daily = df.groupby("day_of_week")["ms_played"].sum().reindex(dow_order, fill_value=0)
monthly = df.groupby(["year", "month"])["ms_played"].sum().reset_index()
monthly["label"] = (
    monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
)
yearly = df.groupby("year")["ms_played"].sum()

top_tracks = (
    df.groupby(["track_name", "artist_name"])
    .agg(play_count=("ms_played", "count"), total_ms=("ms_played", "sum"))
    .reset_index()
    .sort_values("total_ms", ascending=False)
    .head(20)
)

top_albums = (
    df.groupby(["album_name", "artist_name"])
    .agg(play_count=("ms_played", "count"), total_ms=("ms_played", "sum"))
    .reset_index()
    .sort_values("total_ms", ascending=False)
    .head(15)
)

# ---------- 5. EXPORT ----------
out = {
    "summary": {
        "total_plays": int(len(df)),
        "total_hours": round(df["ms_played"].sum() / 3_600_000, 1),
        "unique_artists": int(df["artist_name"].nunique()),
        "unique_tracks": int(df["track_name"].nunique()),
        "date_range": [str(df["ts"].min().date()), str(df["ts"].max().date())],
        "overall_skip_rate": round(df["skipped"].mean() * 100, 1),
    },
    "top_artists": artist_stats.head(20)[
        [
            "artist_name",
            "play_count",
            "total_hours",
            "engagement_score",
            "completion_rate",
            "skip_rate",
            "verified_streams",
            "tier",
            "perks",
        ]
    ]
    .round(2)
    .to_dict("records"),
    "top_tracks": top_tracks.assign(
        hours=lambda x: round(x["total_ms"] / 3_600_000, 2)
    )[["track_name", "artist_name", "play_count", "hours"]].to_dict("records"),
    "top_albums": top_albums.assign(
        hours=lambda x: round(x["total_ms"] / 3_600_000, 2)
    )[["album_name", "artist_name", "play_count", "hours"]].to_dict("records"),
    "hourly_hours": [round(h / 3_600_000, 2) for h in hourly.tolist()],
    "daily_hours": {d: round(v / 3_600_000, 2) for d, v in daily.items()},
    "monthly_hours": [
        {"label": r["label"], "hours": round(r["ms_played"] / 3_600_000, 2)}
        for _, r in monthly.iterrows()
    ],
    "yearly_hours": [
        {"year": int(y), "hours": round(v / 3_600_000, 1)} for y, v in yearly.items()
    ],
    "tier_distribution": artist_stats["tier"].value_counts().to_dict(),
}

with open("dashboard_data.json", "w") as f:
    json.dump(out, f, indent=2, default=str)

print("Top 10 artists by engagement score:")
print(
    artist_stats[
        [
            "artist_name",
            "play_count",
            "total_hours",
            "engagement_score",
            "tier",
            "verified_streams",
        ]
    ]
    .head(10)
    .to_string(index=False)
)
print("\nTier distribution:", artist_stats["tier"].value_counts().to_dict())
print("\nJSON written:", len(json.dumps(out)), "bytes")
