import numpy as np
import pandas as pd

# Generate random data
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=100)

# Compute basic statistics
mean = np.mean(data)
std = np.std(data)
min_val = np.min(data)
max_val = np.max(data)

# Create a DataFrame
df = pd.DataFrame({
    "Value": data
})

df.to_csv("results/random_data.csv", index=False)

print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std:.2f}")
print(f"Min: {min_val:.2f}")
print(f"Max: {max_val:.2f}")
