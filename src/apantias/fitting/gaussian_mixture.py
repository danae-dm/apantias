"""Gaussian mixture model fitting utilities.

Provides specialized fitting routines for multi-Gaussian models with constraints.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm


class FitResult(dict[str, object]):
    """Results from fitting a 3-Gaussian mixture model.

    Keys:
        success: Whether optimization converged.
        message: Optimizer convergence message.
        params: Fitted parameters [w1, w2, mu1, mu_spacing, sigma].
        log_likelihood: Final log-likelihood value.
        negative_log_likelihood: Final negative log-likelihood (objective value).
        initial_guess: Initial parameters used.
        bounds: Parameter bounds used.
    """

    def __init__(
        self,
        success: bool,
        message: str,
        params: NDArray[np.float64],
        log_likelihood: float,
        negative_log_likelihood: float,
        initial_guess: NDArray[np.float64],
        bounds: tuple[list[float], list[float]],
    ) -> None:
        # Cast bounds to list[object] for type checker
        super().__init__({
            "success": success,
            "message": message,
            "params": params,
            "log_likelihood": log_likelihood,
            "negative_log_likelihood": negative_log_likelihood,
            "initial_guess": initial_guess,
            "bounds": bounds,  # type: ignore[dict-item]
        })


def log_likelihood(params: NDArray[np.float64], data: NDArray[np.float64]) -> float:
    """Compute the negative log-likelihood of data under a 3-Gaussian mixture model.

    The model assumes:
    - Three Gaussians: noise (peak 1), signal (peak 2), signal (peak 3)
    - Equidistant means: μ₂ - μ₁ = μ₃ - μ₂ = spacing
    - Equal variance: σ₁ = σ₂ = σ₃ = σ

    Parameters
    ----------
    params
        Model parameters: [w1, w2, mu1, mu_spacing, sigma]
        - w1, w2: weights of first and second Gaussian (w3 = 1 - w1 - w2)
        - mu1: mean of first Gaussian
        - mu_spacing: distance between consecutive means
        - sigma: common standard deviation (must be > 0)
    data
        Input data (1D array).

    Returns
    -------
    float
        Negative log-likelihood (for minimization).
        Returns inf if parameters are invalid.
    """
    w1, w2, mu1, mu_spacing, sigma = params

    # Check parameter validity
    if sigma <= 0:
        return np.inf
    if w1 < 0 or w2 < 0 or w1 + w2 > 1:
        return np.inf
    if mu_spacing <= 0:
        return np.inf

    # Compute weights
    w3 = 1.0 - w1 - w2

    # Evaluate PDFs using the constrained parameterization
    # All three means are determined by mu1 and mu_spacing
    pdf1 = norm.pdf(data, loc=mu1, scale=sigma)
    pdf2 = norm.pdf(data, loc=mu1 + mu_spacing, scale=sigma)
    pdf3 = norm.pdf(data, loc=mu1 + 2 * mu_spacing, scale=sigma)

    # Mixture PDF
    mixture_pdf = w1 * pdf1 + w2 * pdf2 + w3 * pdf3

    # Avoid log(0)
    epsilon = np.finfo(float).eps
    mixture_pdf = np.clip(mixture_pdf, epsilon, None)

    # Negative log-likelihood
    nll = -np.sum(np.log(mixture_pdf))

    return nll


def fit_three_gaussian_mixture(
    data: NDArray[np.float64],
    initial_guess: NDArray[np.float64] | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    method: str = "L-BFGS-B",
) -> FitResult:
    """Fit a 3-Gaussian mixture model with equidistant peaks to data.

    The model assumes:
    - Three Gaussians: noise (peak 1), signal (peak 2), signal (peak 3)
    - Equidistant means: μ₂ - μ₁ = μ₃ - μ₂ = spacing
    - Equal variance: σ₁ = σ₂ = σ₃ = σ

    Parameters
    ----------
    data
        Input data (1D array).
    initial_guess
        Initial parameter estimates: [w1, w2, mu1, mu_spacing, sigma].
        If None, uses heuristic initialization:
        - w1 = w2 = w3 = 1/3
        - mu1 = 10th percentile of data (near the left side, noise peak)
        - mu_spacing = (90th percentile - 10th percentile) / 2
        - sigma = std(data) / 3
    bounds
        Parameter bounds as ([w1_min, w2_min, mu1_min, spacing_min, sigma_min],
                             [w1_max, w2_max, mu1_max, spacing_max, sigma_max]).
        If None, uses sensible defaults:
        - Weights: [0.001, 0.001, ...] to [0.999, 0.999, ...]
        - Mu1: [min(data) - std(data), max(data) + std(data)]
        - Spacing: [0, max(data) - min(data)]
        - Sigma: [0.01 * std, 2 * std]
    method
        Optimization method (default: L-BFGS-B for bound-constrained optimization).

    Returns
    -------
    FitResult
        Results object containing:
        - 'success': bool indicating convergence
        - 'message': optimizer message
        - 'params': fitted parameters [w1, w2, mu1, mu_spacing, sigma]
        - 'log_likelihood': final log-likelihood
        - 'negative_log_likelihood': final negative log-likelihood (objective value)
        - 'initial_guess': initial parameters used
        - 'bounds': parameter bounds used

    Notes
    -----
    The optimization minimizes the negative log-likelihood.
    Use `scipy.optimize.minimize` directly for more control.
    """
    data = np.asarray(data).ravel()

    if len(data) < 10:
        raise ValueError("At least 10 data points are recommended for reliable fitting.")

    # Default initial guess
    if initial_guess is None:
        w1 = 0.33
        w2 = 0.33
        mu1 = float(np.percentile(data, 10))  # Near the left side (noise peak)
        mu_spacing = float((np.percentile(data, 90) - np.percentile(data, 10)) / 2)
        sigma = float(np.std(data) / 3)
        initial_guess = np.array([w1, w2, mu1, mu_spacing, sigma], dtype=np.float64)

    # Default bounds
    if bounds is None:
        weight_bounds = (0.001, 0.999)  # Keep weights away from 0 and 1 for numerical stability
        mu_bounds = (float(np.min(data) - np.std(data)), float(np.max(data) + np.std(data)))
        spacing_bounds = (0.0, float(np.max(data) - np.min(data)))
        sigma_bounds = (
            float(np.std(data) * 0.01),
            float(np.std(data) * 2),
        )

        bounds = (
            [weight_bounds[0], weight_bounds[0], mu_bounds[0], spacing_bounds[0], sigma_bounds[0]],
            [weight_bounds[1], weight_bounds[1], mu_bounds[1], spacing_bounds[1], sigma_bounds[1]],
        )

    # Optimization
    result = minimize(
        fun=log_likelihood,
        x0=initial_guess,
        args=(data,),
        method=method,
        bounds=list(zip(bounds[0], bounds[1], strict=True)),  # type: ignore[arg-type]
        options={"maxiter": 1000},  # type: ignore[dict-item]
    )

    # Extract results
    params = result.x  # type: ignore[union-attr]
    final_nll = float(result.fun)  # type: ignore[union-attr]
    final_ll = -final_nll

    return FitResult(
        success=bool(result.success),  # type: ignore[union-attr]
        message=str(result.message),  # type: ignore[union-attr]
        params=params,
        log_likelihood=final_ll,
        negative_log_likelihood=final_nll,
        initial_guess=initial_guess,
        bounds=bounds,
    )


def get_fitted_parameters(params: NDArray[np.float64]) -> dict[str, list[float] | float]:
    """Extract human-readable fitted parameters from the optimized parameters.

    Parameters
    ----------
    params
        Fitted parameters from `fit_three_gaussian_mixture`: [w1, w2, mu1, mu_spacing, sigma].

    Returns
    -------
    dict
        Dictionary with keys:
        - 'weights': [w1, w2, w3]
        - 'means': [mu1, mu2, mu3]
        - 'sigma': common standard deviation
        - 'spacing': distance between consecutive means
    """
    w1, w2, mu1, mu_spacing, sigma = params

    w3 = 1.0 - w1 - w2
    weights = [float(w1), float(w2), float(w3)]

    mu2 = mu1 + mu_spacing
    mu3 = mu1 + 2 * mu_spacing
    means = [float(mu1), float(mu2), float(mu3)]

    return {
        "weights": weights,
        "means": means,
        "sigma": float(sigma),
        "spacing": float(mu_spacing),
    }


def evaluate_pdf(x: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the fitted 3-Gaussian mixture PDF at given points.

    Parameters
    ----------
    x
        Points at which to evaluate the PDF.
    params
        Fitted parameters: [w1, w2, mu1, mu_spacing, sigma].

    Returns
    -------
    np.ndarray
        PDF values at x.
    """
    w1, w2, mu1, mu_spacing, sigma = params

    w3 = 1.0 - w1 - w2

    mu2 = mu1 + mu_spacing
    mu3 = mu1 + 2 * mu_spacing

    pdf1 = norm.pdf(x, loc=mu1, scale=sigma)
    pdf2 = norm.pdf(x, loc=mu2, scale=sigma)
    pdf3 = norm.pdf(x, loc=mu3, scale=sigma)

    return w1 * pdf1 + w2 * pdf2 + w3 * pdf3


def fit_to_pixel_params(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fit 3-Gaussian mixture to pixel time series and return flat parameter array.

    This function is designed for use with ``apply_pixelwise``. It takes a pixel's
    time series and returns a 1D array of fitted parameters.

    Parameters
    ----------
    data
        Pixel time series of shape (n_frames,). Each value represents the signal
        for that frame at this pixel.

    Returns
    -------
    np.ndarray
        Fitted parameters of shape (7,):
        [w1, w2, w3, mu1, mu2, mu3, sigma]
    """
    result = fit_three_gaussian_mixture(data)
    # Type: ignore because result is a dict with object values
    params = get_fitted_parameters(result["params"])  # type: ignore[call-arg]
    return np.array(
        [
            params["weights"][0],  # type: ignore[index]
            params["weights"][1],  # type: ignore[index]
            params["weights"][2],  # type: ignore[index]
            params["means"][0],  # type: ignore[index]
            params["means"][1],  # type: ignore[index]
            params["means"][2],  # type: ignore[index]
            params["sigma"],  # type: ignore[index]
        ],
        dtype=np.float64,
    )


__all__: list[str] = [
    "log_likelihood",
    "fit_three_gaussian_mixture",
    "get_fitted_parameters",
    "evaluate_pdf",
    "fit_to_pixel_params",
]
