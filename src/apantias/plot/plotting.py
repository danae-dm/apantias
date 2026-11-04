"""Matplotlib-based plotting helpers for notebooks and reports.

Visualization functions intended for interactive Jupyter use and for
producing publication-style figures from analysis outputs.
"""

from collections.abc import Callable
from typing import Any, Protocol, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class PlotDesign(Protocol):
    """Protocol defining the interface for plot design styles."""

    def apply_to_figure(self, fig: Figure) -> None:
        """Apply design styles to a figure."""
        ...

    def apply_to_axes(self, ax: Axes) -> None:
        """Apply design styles to axes."""
        ...


class DefaultPlotDesign:
    """Default plot design with clean, modern styling."""

    def __init__(self) -> None:
        self._apply_mpl_style()

    def _apply_mpl_style(self) -> None:
        """Apply matplotlib style parameters."""
        mpl.rcParams.update({
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.prop_cycle": cycler(
                color=[
                    "#2563eb",  # blue
                    "#dc2626",  # red
                    "#16a34a",  # green
                    "#d97706",  # amber
                    "#9333ea",  # purple
                    "#ec4899",  # pink
                    "#0891b2",  # cyan
                    "#7c3aed",  # violet
                ]
            ),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "legend.fancybox": False,
            "legend.handlelength": 1.0,
            "legend.borderaxespad": 0.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "font.size": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        })

    def apply_to_figure(self, fig: Figure) -> None:
        """Apply design styles to a figure."""
        fig.patch.set_facecolor("white")

    def apply_to_axes(self, ax: Axes) -> None:
        """Apply design styles to axes."""
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3)


# Global design instance that can be customized
_plot_design: DefaultPlotDesign | None = None


def get_plot_design() -> DefaultPlotDesign:
    """Get the current global plot design."""
    global _plot_design
    if _plot_design is None:
        _plot_design = DefaultPlotDesign()
    return _plot_design


def set_plot_design(design: PlotDesign | Callable[[], PlotDesign]) -> None:
    """Set the global plot design.

    Args:
        design: Either a PlotDesign instance or a callable that returns one.
    """
    global _plot_design
    if callable(design):
        _plot_design = cast(DefaultPlotDesign | None, design())
    else:
        _plot_design = cast(DefaultPlotDesign | None, design)


def reset_plot_design() -> None:
    """Reset to the default plot design."""
    global _plot_design
    _plot_design = DefaultPlotDesign()


def histogram(
    data: np.ndarray,
    bins: int,
    range: tuple[float, float] | None = None,
    density: bool = False,
    cumulative: bool = False,
    alpha: float = 0.7,
    label: str | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a histogram plot.

    Args:
        data: 1D array of data values.
        bins: Number of histogram bins.
        range: (min, max) range for histogram. If None, uses data range.
        density: If True, normalize histogram to form a probability density.
        cumulative: If True, compute cumulative histogram.
        alpha: Transparency of the histogram bars.
        label: Label for the histogram (used in legend).
        ax: Existing axes to plot on. If None, creates new figure.
        **kwargs: Additional keyword arguments passed to matplotlib's hist.

    Returns:
        Tuple of (figure, axes) for the plot.

    Raises:
        ValueError: If data is not 1D.
    """
    if data.ndim != 1:
        raise ValueError(f"Data must be 1D, got {data.ndim}D array")

    if ax is None:
        fig, ax = plt.subplots()  # type: ignore[assignment]
    else:
        fig = ax.figure

    # Apply design to figure and axes
    design = get_plot_design()
    if isinstance(fig, Figure):
        design.apply_to_figure(fig)
    design.apply_to_axes(ax)

    # Calculate histogram
    _, bin_edges, patches = ax.hist(  # type: ignore[unreachable, call-overload]
        data,
        bins=bins,
        range=range,
        density=density,
        cumulative=cumulative,
        alpha=alpha,
        label=label,
        **kwargs,
    )

    # Style the patches (bars)
    # patches can be:
    # - BarContainer (single dataset): directly iterable over Rectangle objects
    # - list of BarContainer (multiple datasets): iterate over list, then each BarContainer
    if isinstance(patches, list):
        # Multiple datasets - patches is a list of BarContainer objects
        for patch_list in patches:
            # Each patch_list is a BarContainer, iterate over it
            for patch in patch_list:  # type: ignore[union-attr]
                if hasattr(patch, "set_edgecolor"):
                    assert patch is not None  # for type checker
                    patch.set_edgecolor("black")  # type: ignore[attr-defined]
                    patch.set_linewidth(0.5)  # type: ignore[attr-defined]
    else:
        # Single BarContainer - iterate directly
        for patch in patches:  # type: ignore[union-attr]
            if hasattr(patch, "set_edgecolor"):
                assert patch is not None  # for type checker
                patch.set_edgecolor("black")  # type: ignore[attr-defined]
                patch.set_linewidth(0.5)  # type: ignore[attr-defined]

    # Add labels if not already set
    if ax.get_xlabel() == "":
        ax.set_xlabel("Value")  # type: ignore[call-arg]
    if ax.get_ylabel() == "":
        ax.set_ylabel("Count" if not density else "Density")  # type: ignore[call-arg]

    if label is not None:
        ax.legend()  # type: ignore[call-arg]

    return cast(tuple[Figure, Axes], (fig, ax))


# Export public API
__all__ = [
    "PlotDesign",
    "DefaultPlotDesign",
    "get_plot_design",
    "set_plot_design",
    "reset_plot_design",
    "histogram",
]
