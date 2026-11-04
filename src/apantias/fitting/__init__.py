"""Curve fitting and model-fitting utilities.

Functions for fitting detector data — e.g. linear slopes, histogram fits,
and any future regression or curve-fitting routines.
"""

from .gaussian_mixture import (
    evaluate_pdf,
    fit_three_gaussian_mixture,
    fit_to_pixel_params,
    get_fitted_parameters,
    log_likelihood,
)

__all__: list[str] = [
    "evaluate_pdf",
    "fit_three_gaussian_mixture",
    "fit_to_pixel_params",
    "get_fitted_parameters",
    "log_likelihood",
]
