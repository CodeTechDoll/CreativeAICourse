import matplotlib.pyplot as plt
from matplotlib_venn import venn3

plt.figure(figsize=(8, 8))
venn3(subsets=(1, 1, 1, 1, 1, 1, 1),
    set_labels=('AI', 'ML', 'DL'),
    set_colors=('red', 'green', 'blue'),
    alpha=0.5)

plt.title("Relationship between AI, ML, and DL")
plt.show()


data = [["Classification", "Clustering"],
        ["Regression", "Dimensionality Reduction"]]

fig, ax = plt.subplots()
ax.axis("off")

table = ax.table(cellText=data,
    colLabels=["Supervised Learning", "Unsupervised Learning"],
    cellLoc="center",
    loc="center")

table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 1.5)

plt.title("Comparison of Supervised and Unsupervised Learning")
plt.show()

fig, ax = plt.subplots()
ax.axis("off")

# Agent and environment
ax.add_patch(plt.Rectangle((0.3, 0.3), 0.4, 0.4, fill=True, color="gray", alpha=0.5))
ax.text(0.45, 0.5, "Agent", fontsize=12, ha="center")
ax.add_patch(plt.Rectangle((0.7, 0.3), 0.4, 0.4, fill=True, color="gray", alpha=0.5))
ax.text(0.85, 0.5, "Environment", fontsize=12, ha="center")

# Interaction arrows
ax.annotate("Action", (0.7, 0.4), (0.3, 0.4), fontsize=12, ha="center",
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("State", (0.7, 0.6), (0.3, 0.6), fontsize=12, ha="center",
            arrowprops=dict(arrowstyle="<-", lw=1.5))
ax.annotate("Reward", (0.7, 0.5), (0.3, 0.5), fontsize=12, ha="center",
            arrowprops=dict(arrowstyle="<-", lw=1.5))

plt.title("Agent-Environment Interaction in Reinforcement Learning")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
x = np.random.rand(50)
y = 2 * x + 1 + 0.1 * np.random.randn(50)

# Fit a linear regression model
coeffs = np.polyfit(x, y, deg=1)
y_pred = np.polyval(coeffs, x)

# Create a scatter plot of the data and the best-fit line
plt.scatter(x, y, label="Data points", color="blue")
plt.plot(x, y_pred, label="Best-fit line", color="red", linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Example")
plt.legend()
plt.show()