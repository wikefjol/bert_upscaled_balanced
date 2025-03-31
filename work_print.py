import pandas as pd
import matplotlib.pyplot as plt

# 1) GPU cost factors from your table
gpu_cost = {
    "t4": 0.35,
    "a40": 1.00,
    "v100": 1.31,
    "a100": 1.84,
    "a100fat": 2.20
}

# 2) Read the data file
df = pd.read_csv("all_jobs.txt", sep='|')

# 3) Parse dates and elapsed times
df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
df['Elapsed'] = pd.to_timedelta(df['Elapsed'], errors='coerce')

# 4) Keep only rows that mention a GPU in AllocTRES
df = df[df['AllocTRES'].str.contains("gpu", na=False)]
df.dropna(subset=['Start', 'Elapsed'], inplace=True)

# 5) Use extractall to find each GPU type and count
#    Pattern captures e.g. "gpu:t4=1", "gpu:a100fat=2", etc.
matches = df['AllocTRES'].str.extractall(r'(?:gpu:)([^\=]+)=(\d+)')
# matches has a MultiIndex (row + match #), with columns [0,1] for (type, count)

# Convert the count to float
matches[1] = matches[1].astype(float)
matches.rename(columns={0: 'gpu_type', 1: 'gpu_count'}, inplace=True)

# 6) We'll merge these extracted matches back to the main df
#    Because .extractall returns a multi-level index, we join by the first level (row).
df_merged = df.join(matches.droplevel(1))

# Now each row in df_merged is a single GPU type, with "gpu_type" and "gpu_count".

# 7) Compute "Used hours" per job row:  (Elapsed × gpu_count × costFactor)
def map_cost(gpu_type):
    # e.g. "t4" or "a100fat"
    # Lowercase the string to be safe
    gpu_type = gpu_type.lower().strip()
    return gpu_cost.get(gpu_type, 0.0)  # default 0 if not found

df_merged['CostFactor'] = df_merged['gpu_type'].apply(map_cost)

# Convert Elapsed to hours
df_merged['Elapsed_hours'] = df_merged['Elapsed'].dt.total_seconds() / (3600)


# "Used hours" = Elapsed × #GPUs × costFactor
df_merged['Used_hours'] = df_merged['Elapsed_hours'] * df_merged['gpu_count'] * df_merged['CostFactor']

# 8) Sum up each job's multiple GPU lines (if a job had T4 + A40 in the same job, etc.)
#    But to do a daily time series, we can group by date and sum "Used_hours"
df_merged['Date'] = df_merged['Start'].dt.date
daily_used = df_merged.groupby('Date')['Used_hours'].sum()

# 9) Fill in missing dates if you want from, say, Jan 1 onward
full_range = pd.date_range(start="2025-01-01", end=pd.to_datetime(daily_used.index.max()))
daily_used = daily_used.reindex(full_range.date, fill_value=0)
daily_used.index = pd.to_datetime(daily_used.index)

# 10) Compute a 30-day rolling sum
rolling_30d = daily_used.rolling(30).sum()

# 11) Plot
plt.plot(rolling_30d.index, rolling_30d.values)
plt.ylabel("Used [h] (30-day rolling total)")
plt.xlabel("Date")
plt.title("Rolling 30-Day GPU Usage (Cluster 'Used [h]' metric)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("rolling_gpu_usage.png", dpi=300, bbox_inches='tight')
