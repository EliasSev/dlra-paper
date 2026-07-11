"""
Shared palette and style constants for figures (IBM colorblind-safe palette).
"""

# Categorical palettes (2, 3, 5 series).
C5 = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
C3 = ["#648FFF", "#785EF0", "#DC267F"]
C2 = ["#648FFF", "#DC267F"]

# Sequential colormap for fields / images.
CMAP = "plasma"

# Marker cycle: circle, star, triangle, square, diamond.
M = ["o", "*", "^", "s", "d"]
MS = [6, 8, 6, 6, 6]        # matching marker sizes

# Line-style cycle.
LS = ["-", "--", ":", "-."]