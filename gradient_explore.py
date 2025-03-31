import re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def extract_gradient_norms(fname):
    with open(fname, 'r') as f:
        text = f.read()
    # Extract numbers (including scientific notation)
    norm_strs = re.findall(r"Gradient norm before optimizer step:\s*([\d\.Ee+-]+)", text)
    return [float(x) for x in norm_strs]

def main(fname):
    norms = extract_gradient_norms(fname)
    # Structure data: each norm corresponds to one batch.
    df = pd.DataFrame({"Batch": np.arange(1, len(norms)+1), "Gradient Norm": norms})
    print(df)

    # Plot: Gradient norm over batches
    plt.figure(figsize=(8, 4))
    plt.plot(df["Batch"], df["Gradient Norm"], marker='o', linestyle='-')
    plt.xlabel("Batch")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Over Batches")
    plt.tight_layout()
    plt.savefig("gradient_norm_over_batches.png")
    plt.show()

    # Kernel Density Estimate of gradient norms
    kde = gaussian_kde(norms)
    x_vals = np.linspace(min(norms), max(norms), 1000)
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, kde(x_vals))
    plt.xlabel("Gradient Norm")
    plt.ylabel("Density")
    plt.title("Kernel Density Estimate of Gradient Norms")
    plt.tight_layout()
    plt.savefig("gradient_norm_density.png")
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <log_file.txt>")
        sys.exit(1)
    main(sys.argv[1])
