#!/usr/bin/env python
"""
plot_profiles.py

Reads profiling data from "profile_log.csv", replaces extremely high values in each measurement column
(with values above the 95th percentile set to NaN), averages the data per step (across seeds), and
produces one figure with five subplots arranged in a grid:

  1. Latency Breakdown for Non-Training Steps.
  2. Latency Breakdown for Training Steps.
  3. Memory Usage Breakdown for Non-Training Steps.
  4. Memory Usage Breakdown for Training Steps.
  5. Reward per Training Step.

Latency breakdown uses:
  - Non-training steps: latency_action, latency_env, latency_rb_add.
  - Training steps: the above plus latency_rb_sample, latency_target_loss, latency_train_net.

Memory breakdown uses:
  - Non-training steps: mem_nn, mem_env, mem_rb.
  - Training steps: the above plus mem_opt, mem_act, mem_grad.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Read the CSV file.
df = pd.read_csv("profile_log_torch.csv")

# List of measurement columns to filter out extreme outliers (95th percentile).
meas_cols = [
    "latency_action", "latency_env", "latency_rb_add", "latency_rb_sample",
    "latency_target_loss", "latency_train_net", "total_latency",
    "mem_nn", "mem_env", "mem_rb", "mem_opt", "mem_act", "mem_grad"
]

# For each measurement column, replace values above the 95th percentile with NaN.
for col in meas_cols:
    if col in df.columns:
        thresh = df[col].quantile(0.95)
        df.loc[df[col] > thresh, col] = np.nan

# Group the data by 'step' and average across seeds.
df_avg = df.groupby("step", as_index=False).mean()

# Recompute the training flag (0 or 1) per step.
is_train = df.groupby("step")["is_train_update"].mean().round().astype(int).reset_index()
df_avg = pd.merge(is_train, df_avg, on="step", suffixes=("_flag", ""))
df_avg["is_train_update"] = df_avg["is_train_update_flag"]
df_avg.drop(columns=["is_train_update_flag"], inplace=True)

# Split data into steps with and without training update.
df_no_train = df_avg[df_avg["is_train_update"] == 0]
df_train = df_avg[df_avg["is_train_update"] == 1]

# Define component lists.
latency_no_train = ["latency_action", "latency_env", "latency_rb_add"]
latency_train = ["latency_action", "latency_env", "latency_rb_add",
                 "latency_rb_sample", "latency_target_loss", "latency_train_net"]

memory_no_train = ["mem_nn", "mem_env", "mem_rb"]
memory_train = ["mem_nn", "mem_env", "mem_opt", "mem_act", "mem_grad", "mem_rb"]

# Create one figure with a grid:
# First two rows: two subplots each (latency and memory for non-training & training).
# Third row: one subplot spanning both columns for reward.
fig = plt.figure(constrained_layout=True, figsize=(14, 18))
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.7])

# Subplot 1: Latency Breakdown (Non-Training)
ax1 = fig.add_subplot(gs[0, 0])
ax1.stackplot(df_no_train["step"],
              [df_no_train[col] for col in latency_no_train],
              labels=latency_no_train)
ax1.set_title("Latency Breakdown (No Training Update)")
ax1.set_xlabel("Step")
ax1.set_ylabel("Latency (seconds)")
ax1.legend(loc="upper left")

# Subplot 2: Latency Breakdown (Training)
ax2 = fig.add_subplot(gs[0, 1])
ax2.stackplot(df_train["step"],
              [df_train[col] for col in latency_train],
              labels=latency_train)
ax2.set_title("Latency Breakdown (Training Update)")
ax2.set_xlabel("Step")
ax2.set_ylabel("Latency (seconds)")
ax2.legend(loc="upper left")

# Subplot 3: Memory Usage (Non-Training)
ax3 = fig.add_subplot(gs[1, 0])
ax3.stackplot(df_no_train["step"],
              [df_no_train[col] for col in memory_no_train],
              labels=memory_no_train)
ax3.set_title("Memory Usage Breakdown (No Training Update)")
ax3.set_xlabel("Step")
ax3.set_ylabel("Memory (bytes)")
ax3.legend(loc="upper left")

# Subplot 4: Memory Usage (Training)
ax4 = fig.add_subplot(gs[1, 1])
ax4.stackplot(df_train["step"],
              [df_train[col] for col in memory_train],
              labels=memory_train)
ax4.set_title("Memory Usage Breakdown (Training Update)")
ax4.set_xlabel("Step")
ax4.set_ylabel("Memory (bytes)")
ax4.legend(loc="upper left")

# Subplot 5: Reward per Training Step (line plot)
ax5 = fig.add_subplot(gs[2, :])
ax5.plot(df_train["step"], df_train["reward"], marker="o", linestyle="-", color="tab:blue")
ax5.set_title("Reward per Training Step")
ax5.set_xlabel("Step")
ax5.set_ylabel("Reward")
ax5.grid(True)

plt.suptitle("TORCH Profiling Summary", fontsize=16)
plt.savefig("profiling_summary_torch.png")
plt.show()
