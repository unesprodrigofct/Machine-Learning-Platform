"""Generate deterministic, non-sensitive data for the runnable platform example."""

from __future__ import annotations

import csv
import random
from pathlib import Path


def main() -> None:
    random_generator = random.Random(42)
    output_path = Path("data/sample/delivery_classification.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["distance_km", "prep_minutes", "weather", "order_hour", "late"])
        writer.writeheader()
        for _ in range(240):
            distance = round(random_generator.uniform(1, 18), 2)
            preparation = random_generator.randint(5, 45)
            weather = random_generator.choice(["clear", "rain", "cloudy"])
            hour = random_generator.randint(8, 23)
            late_probability = 0.12 + (distance / 50) + (preparation / 120) + (0.20 if weather == "rain" else 0)
            writer.writerow({
                "distance_km": distance,
                "prep_minutes": preparation,
                "weather": weather,
                "order_hour": hour,
                "late": int(random_generator.random() < late_probability),
            })


if __name__ == "__main__":
    main()
