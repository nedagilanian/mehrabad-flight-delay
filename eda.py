import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("mehrabad_flights.csv")

# =========================
# Average Delay by Airline
# =========================

airline_delay = (
    df.groupby("Airline")["DelayMinutes"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    airline_delay.index,
    airline_delay.values
)

plt.title(
    "Average Flight Delay by Airline",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Airline", fontsize=12)
plt.ylabel("Average Delay (Minutes)", fontsize=12)

plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.3,
        f"{height:.1f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    "delay_by_airline.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =========================
# Average Delay by Destination
# =========================

dest_delay = (
    df.groupby("Destination")["DelayMinutes"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    dest_delay.index,
    dest_delay.values
)

plt.title(
    "Average Flight Delay by Destination",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Destination", fontsize=12)
plt.ylabel("Average Delay (Minutes)", fontsize=12)

plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.3,
        f"{height:.1f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    "delay_by_destination.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =========================
# Average Delay by Hour
# =========================

hourly_delay = (
    df.groupby("ScheduledHour")["DelayMinutes"]
    .mean()
)

plt.figure(figsize=(12, 6))

plt.plot(
    hourly_delay.index,
    hourly_delay.values,
    marker="o",
    linewidth=2
)

plt.title(
    "Average Delay by Scheduled Hour",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Hour of Day")
plt.ylabel("Average Delay (Minutes)")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "delay_by_hour.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================
# Delay Distribution
# =========================

plt.figure(figsize=(10, 6))

plt.hist(
    df["DelayMinutes"],
    bins=20
)

plt.title(
    "Distribution of Flight Delays",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Delay Minutes")
plt.ylabel("Number of Flights")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "delay_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("EDA completed successfully")