# Create synthetic data for support ticket analytics
# Output data/raw/tickets_raw.csv - main tickets table

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# Where to write the CSV
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = RAW_DIR / "support_tickets.csv"


def random_datetime(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between start and end."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def main(num_rows: int = 500) -> None:
    random.seed(42)  # reproducible synthetic data

    now = datetime.utcnow()
    start = now - timedelta(days=90)  # last ~90 days

    priorities = ["Low", "Medium", "High"]
    channels = ["Email", "Chat", "Phone", "Web"]
    categories = ["Login Issues", "Billing", "Tech Support", "Account Update", "Other"]
    agents = ["Agent A", "Agent B", "Agent C", "Agent D"]
    statuses = ["Open", "In Progress", "Resolved", "Closed"]

    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header row
        writer.writerow([
            "ticket_id",
            "created_at",
            "resolved_at",
            "priority",
            "channel",
            "category",
            "agent",
            "status"
        ])

        for i in range(1, num_rows + 1):
            ticket_id = f"T{i:04d}"
            created_at = random_datetime(start, now)

            priority = random.choice(priorities)
            channel = random.choice(channels)
            category = random.choice(categories)
            agent = random.choice(agents)

            # Decide if resolved or still open
            is_resolved = random.random() < 0.8  # ~80% resolved
            if is_resolved:
                # resolution time depends loosely on priority
                if priority == "High":
                    max_hours = 24
                elif priority == "Medium":
                    max_hours = 48
                else:
                    max_hours = 72

                resolution_hours = random.uniform(1, max_hours * 1.5)
                resolved_at = created_at + timedelta(hours=resolution_hours)

                status = random.choice(["Resolved", "Closed"])
                resolved_str = resolved_at.isoformat(timespec="seconds")
            else:
                resolved_str = ""  # blank -> not resolved
                status = random.choice(["Open", "In Progress"])

            created_str = created_at.isoformat(timespec="seconds")

            writer.writerow([
                ticket_id,
                created_str,
                resolved_str,
                priority,
                channel,
                category,
                agent,
                status
            ])

    print(f"Wrote {num_rows} synthetic tickets to {OUT_FILE}")


if __name__ == "__main__":
    main()
