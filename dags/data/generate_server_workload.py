"""
Run this locally to generate both dataset files.
Usage: cd dags/data && python generate_server_workloads.py
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 300

# Cluster 1: CPU-bound jobs
cpu_bound = pd.DataFrame({
    'cpu_percent': np.random.normal(85, 8, n//4).clip(10, 100),
    'memory_percent': np.random.normal(40, 10, n//4).clip(5, 100),
    'disk_io_mbps': np.random.normal(20, 10, n//4).clip(0, 200),
    'network_mbps': np.random.normal(15, 8, n//4).clip(0, 100),
    'runtime_seconds': np.random.normal(600, 150, n//4).clip(30, 3600),
    'gpu_util_percent': np.random.normal(10, 5, n//4).clip(0, 100),
})

# Cluster 2: Memory-bound jobs
mem_bound = pd.DataFrame({
    'cpu_percent': np.random.normal(50, 10, n//4).clip(10, 100),
    'memory_percent': np.random.normal(88, 6, n//4).clip(5, 100),
    'disk_io_mbps': np.random.normal(80, 20, n//4).clip(0, 200),
    'network_mbps': np.random.normal(30, 10, n//4).clip(0, 100),
    'runtime_seconds': np.random.normal(1200, 300, n//4).clip(30, 3600),
    'gpu_util_percent': np.random.normal(15, 8, n//4).clip(0, 100),
})

# Cluster 3: GPU-intensive jobs
gpu_bound = pd.DataFrame({
    'cpu_percent': np.random.normal(55, 12, n//4).clip(10, 100),
    'memory_percent': np.random.normal(70, 10, n//4).clip(5, 100),
    'disk_io_mbps': np.random.normal(50, 15, n//4).clip(0, 200),
    'network_mbps': np.random.normal(60, 15, n//4).clip(0, 100),
    'runtime_seconds': np.random.normal(1800, 400, n//4).clip(30, 3600),
    'gpu_util_percent': np.random.normal(90, 7, n//4).clip(0, 100),
})

# Cluster 4: Idle/lightweight jobs
idle = pd.DataFrame({
    'cpu_percent': np.random.normal(10, 5, n//4).clip(10, 100),
    'memory_percent': np.random.normal(15, 8, n//4).clip(5, 100),
    'disk_io_mbps': np.random.normal(5, 3, n//4).clip(0, 200),
    'network_mbps': np.random.normal(5, 3, n//4).clip(0, 100),
    'runtime_seconds': np.random.normal(60, 30, n//4).clip(30, 3600),
    'gpu_util_percent': np.random.normal(2, 2, n//4).clip(0, 100),
})

df = pd.concat([cpu_bound, mem_bound, gpu_bound, idle], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.round(2)

# Save training data
df.to_csv('server_workloads.csv', index=False)
print(f"Generated server_workloads.csv with {len(df)} rows")

# Save test data (5 sample workloads for prediction)
test = pd.DataFrame({
    'cpu_percent': [92.0, 45.0, 60.0, 12.0, 70.0],
    'memory_percent': [35.0, 90.0, 75.0, 10.0, 55.0],
    'disk_io_mbps': [15.0, 85.0, 45.0, 3.0, 40.0],
    'network_mbps': [10.0, 25.0, 55.0, 4.0, 30.0],
    'runtime_seconds': [500.0, 1100.0, 2000.0, 45.0, 800.0],
    'gpu_util_percent': [8.0, 12.0, 92.0, 1.0, 50.0],
})
test.to_csv('test.csv', index=False)
print(f"Generated test.csv with {len(test)} rows")
print(df.describe())