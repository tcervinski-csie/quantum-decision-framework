"""Generate Figure 1: System Architecture diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.5)
ax.axis("off")

# Colors
box_color = "#E8EDF2"
engine_color = "#D0DCE8"
interface_color = "#C2D1E0"
arrow_color = "#3B3B3B"
text_color = "#1A1A1A"
sub_color = "#555555"

# Agent Interface box on top
agent_box = mpatches.FancyBboxPatch(
    (1.5, 3.3), 7, 0.9, boxstyle="round,pad=0.15",
    facecolor=interface_color, edgecolor="#888888", linewidth=1.2
)
ax.add_patch(agent_box)
ax.text(5, 3.75, "Agent Interface", ha="center", va="center",
        fontsize=12, fontweight="bold", color=text_color)
ax.text(5, 3.4, "quantum_decision()", ha="center", va="center",
        fontsize=9, color=sub_color, style="italic")

# Arrow from agent interface down
ax.annotate("", xy=(5, 2.65), xytext=(5, 3.25),
            arrowprops=dict(arrowstyle="->, head_width=0.3",
                          color=arrow_color, lw=1.5))

# Pipeline boxes
boxes = [
    (0.3, "Problem\nAnalyzer", "NL \u2192 Features", box_color),
    (2.7, "Decision\nEngine", "Classify + Gate", engine_color),
    (5.1, "Code\nGenerator", "Build Circuit", box_color),
    (7.5, "Executor", "Run on\nAer / IBM", box_color),
]

for x, label, sublabel, color in boxes:
    box = mpatches.FancyBboxPatch(
        (x, 1.2), 2.0, 1.4, boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="#888888", linewidth=1.2
    )
    ax.add_patch(box)
    ax.text(x + 1.0, 2.05, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color=text_color)
    ax.text(x + 1.0, 1.4, sublabel, ha="center", va="center",
            fontsize=8, color=sub_color)

# Arrows between pipeline boxes
for x_start in [2.35, 4.75, 7.15]:
    ax.annotate("", xy=(x_start + 0.35, 1.9), xytext=(x_start, 1.9),
                arrowprops=dict(arrowstyle="->, head_width=0.25",
                              color=arrow_color, lw=1.5))

# Input/Output labels
ax.text(0.05, 1.9, "Problem\ndescription", ha="center", va="center",
        fontsize=8, color=sub_color, style="italic")
ax.annotate("", xy=(0.3, 1.9), xytext=(0.22, 1.9),
            arrowprops=dict(arrowstyle="->, head_width=0.2",
                          color=arrow_color, lw=1.0))

# Stage labels at bottom
stages = [
    (1.3, "Stage 0"),
    (3.7, "Stages 1-3"),
    (6.1, "Generation"),
    (8.5, "Execution"),
]
for x, label in stages:
    ax.text(x, 0.95, label, ha="center", va="center",
            fontsize=8, color="#777777")

# Title
ax.text(5, 0.3, "Fig. 1. System architecture and data flow.",
        ha="center", va="center", fontsize=10, color=text_color)

plt.tight_layout()
plt.savefig("/home/tcervinski/Projects/quantum-ai/paper/figure1.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("/home/tcervinski/Projects/quantum-ai/paper/figure1.pdf",
            bbox_inches="tight", facecolor="white")
print("Saved figure1.png and figure1.pdf")
