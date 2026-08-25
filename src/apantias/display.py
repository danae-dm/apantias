from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#
def draw_heatmap(
    data: np.ndarray,
    title: Optional[str] = None,
    cmap: str = "viridis",
    figsize: tuple[int, int] = (10, 8),
) -> tuple[Figure, Axes]:
    """
    Draw a heatmap of a 2D numpy array.

    Args:
        data: 2D numpy array to visualize
        title: Optional title for the plot
        cmap: Colormap to use (default: 'viridis')
        figsize: Figure size as (width, height) tuple

    Returns:
        tuple: (figure, axes) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)  # type: ignore[return-value]

    im = ax.imshow(data, cmap=cmap, aspect="auto")  # type: ignore[attr-defined]

    if title:
        ax.set_title(title)  # type: ignore[attr-defined]

    ax.set_xlabel("Column")  # type: ignore[attr-defined]
    ax.set_ylabel("Row")  # type: ignore[attr-defined]

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)  # type: ignore[attr-defined]
    cbar.set_label("Value")  # type: ignore[attr-defined]

    plt.tight_layout()

    return fig, ax
