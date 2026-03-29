import pandas as pd
import random
from datetime import datetime, timedelta

services = ["EC2", "S3", "RDS", "Lambda", "CloudFront"]
regions = ["us-east-1", "us-west-2", "eu-central-1"]
teams = ["ml", "backend", "frontend", "infra"]

data = []
start_date = datetime(2026, 1, 1)

for i in range(90):  # 3 months
    for service in services:
        usage = random.randint(10, 100)

        # simulate realistic cost differences
        if service == "EC2":
            cost = usage * random.uniform(0.4, 0.9)
        elif service == "RDS":
            cost = usage * random.uniform(0.5, 1.0)
        else:
            cost = usage * random.uniform(0.1, 0.5)

        data.append({
            "date": (start_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "service": service,
            "usage_hours": usage,
            "cost": round(cost, 2),
            "region": random.choice(regions),
            "team": random.choice(teams)
        })

df = pd.DataFrame(data)
df.to_csv("data/cloud_usage.csv", index=False)

print("Dataset created at data/cloud_usage.csv")