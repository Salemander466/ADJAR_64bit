from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np

from adjar64 import get_logger, require_optional

log = get_logger("core.filter")


@dataclass(frozen=True)
class GainCurve:
    """
    Frequency-domain gain curve.

    Attributes:
        freqs_hz: frequency axis for the gain values (0..Nyquist), shape (F,)
        gain: gain values in [0, 1], shape (F,)
        kind: description of how the curve was constructed
        params: dict-like info for diagnostics/plotting
    """
    freqs_hz: np.ndarray
    gain: np.ndarray
    kind: str
    params: dict


# =============================================================================
# Gain curve construction
# =============================================================================

def compute_gain_curve(
    *,
    fs_hz: float,
    n_times: int,
    isi_pre_ms: Optional[np.ndarray] = None,
    isi_std_ms: Optional[float] = None,
    kind: Literal["woldorff_placeholder", "manual_lowpass"] = "woldorff_placeholder",
    manual_lowpass_hz: Optional[float] = None,
    transition_width_hz: Optional[float] = None,
) -> GainCurve:
    """
    Compute a frequency-domain gain curve.

    Supported kinds:
      - "woldorff_placeholder":
          Produces a low-pass-like gain curve parameterized by ISI distribution width.
          Narrower ISI distributions -> stronger attenuation of higher frequencies.
          This is intended as a practical placeholder consistent with Woldorff's
          qualitative description (gain depends on ISI distribution width).
      - "manual_lowpass":
          User-specified low-pass (soft transition).

    Args:
        fs_hz: sampling rate
        n_times: epoch length in samples
        isi_pre_ms: per-trial preceding ISIs, used to compute std if isi_std_ms not given
        isi_std_ms: explicit ISI standard deviation (ms); overrides isi_pre_ms if provided
        manual_lowpass_hz: cutoff for manual kind
        transition_width_hz: optional transition width for the soft knee (default depends on cutoff)

    Returns:
        GainCurve with freqs 0..Nyquist for rFFT length.

    Notes:
        - Gain curve length corresponds to rFFT bins for n_times.
        - You will multiply FFT(signal) * gain, then inverse FFT.
    """
    if fs_hz <= 0 or not np.isfinite(fs_hz):
        raise ValueError(f"fs_hz must be finite and > 0, got {fs_hz}")
    if n_times < 2:
        raise ValueError("n_times must be >= 2")

    freqs = np.fft.rfftfreq(n_times, d=1.0 / fs_hz)

    if kind == "manual_lowpass":
        if manual_lowpass_hz is None or manual_lowpass_hz <= 0:
            raise ValueError("manual_lowpass_hz must be provided and > 0 for manual_lowpass")
        gain = _soft_lowpass_gain(freqs, cutoff_hz=float(manual_lowpass_hz), transition_width_hz=transition_width_hz)
        return GainCurve(
            freqs_hz=freqs.astype(np.float64, copy=False),
            gain=gain.astype(np.float64, copy=False),
            kind="manual_lowpass",
            params={
                "cutoff_hz": float(manual_lowpass_hz),
                "transition_width_hz": float(transition_width_hz) if transition_width_hz is not None else None,
            },
        )

    if kind != "woldorff_placeholder":
        raise ValueError("kind must be one of: 'woldorff_placeholder', 'manual_lowpass'")

    # Determine ISI width parameter
    if isi_std_ms is None:
        if isi_pre_ms is None:
            raise ValueError("Provide isi_pre_ms or isi_std_ms for woldorff_placeholder gain curve")
        isi = np.asarray(isi_pre_ms, dtype=np.float64)
        if isi.ndim != 1 or isi.size < 2:
            raise ValueError("isi_pre_ms must be 1D with at least 2 values to compute std")
        if not np.isfinite(isi).all():
            raise ValueError("isi_pre_ms contains NaN/inf")
        if np.any(isi < 0):
            raise ValueError("isi_pre_ms contains negative values")
        isi_std_ms = float(np.std(isi, ddof=1))
    else:
        isi_std_ms = float(isi_std_ms)

    if not np.isfinite(isi_std_ms) or isi_std_ms < 0:
        raise ValueError("isi_std_ms must be finite and >= 0")

    # Woldorff-style placeholder mapping:
    # Use an "effective cutoff" inversely proportional to ISI std:
    #  - If ISI std is large (timing highly variable), overlap smears broadly in time and
    #    the gain should be closer to 1 for a wider band.
    #  - If ISI std is small (timing tightly clustered), overlap produces more coherent
    #    contamination, and the adjacent-response gain acts like a stronger low-pass.
    #
    # We set:
    #   f_c = alpha / (isi_std_seconds + epsilon)
    # and clamp to reasonable bounds.
    #
    # alpha chosen so that:
    #   isi_std_ms = 50ms -> f_c ~ 8-12 Hz (typical ERP low-frequency band)
    #   isi_std_ms = 150ms -> f_c ~ 3-6 Hz (more aggressive)
    #
    # You can tune alpha later to match your validation datasets.
    eps = 1e-6
    isi_std_s = isi_std_ms / 1000.0
    alpha = 0.5  # heuristic scaling constant (Hz * seconds)
    fc = alpha / (isi_std_s + eps)

    # Clamp cutoff to [0.5, Nyquist]
    nyq = fs_hz / 2.0
    fc = float(np.clip(fc, 0.5, nyq))

    # Transition width: default to 0.25 * cutoff (but not tiny)
    if transition_width_hz is None:
        tw = max(0.25 * fc, 0.5)
    else:
        tw = float(transition_width_hz)

    gain = _soft_lowpass_gain(freqs, cutoff_hz=fc, transition_width_hz=tw)

    return GainCurve(
        freqs_hz=freqs.astype(np.float64, copy=False),
        gain=gain.astype(np.float64, copy=False),
        kind="woldorff_placeholder",
        params={
            "isi_std_ms": float(isi_std_ms),
            "effective_cutoff_hz": float(fc),
            "transition_width_hz": float(tw),
            "alpha": float(alpha),
        },
    )


def _soft_lowpass_gain(freqs_hz: np.ndarray, *, cutoff_hz: float, transition_width_hz: Optional[float]) -> np.ndarray:
    """
    Soft low-pass gain curve using a logistic knee.

    gain(f) ~ 1 for f << cutoff; decays smoothly to ~0 above cutoff.

    transition_width_hz controls slope: larger width -> gentler slope.
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    if cutoff_hz <= 0:
        return np.ones_like(f)

    if transition_width_hz is None:
        transition_width_hz = max(0.25 * cutoff_hz, 0.5)

    tw = float(transition_width_hz)
    if tw <= 0:
        # Hard cutoff
        g = (f <= cutoff_hz).astype(np.float64)
        return g

    # Logistic centered at cutoff
    # g = 1 / (1 + exp((f - cutoff)/k))
    k = tw / 6.0  # map "transition width" to logistic slope scale
    k = max(k, 1e-6)
    g = 1.0 / (1.0 + np.exp((f - cutoff_hz) / k))
    return g


# =============================================================================
# Applying gain curves (FFT-domain filtering)
# =============================================================================

def apply_gain_filter_1d(x: np.ndarray, gain: GainCurve) -> np.ndarray:
    """
    Apply a GainCurve to a 1D real signal via rFFT.

    Args:
        x: 1D array length n_times
        gain: GainCurve produced for same fs_hz and n_times

    Returns:
        Filtered signal, float64.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")
    if x.size < 2:
        return x.copy()
    if gain.gain.shape[0] != np.fft.rfft(x).shape[0]:
        raise ValueError("Gain curve length does not match rFFT length for x")

    X = np.fft.rfft(x)
    Y = X * gain.gain
    y = np.fft.irfft(Y, n=x.size)
    return y.astype(np.float64, copy=False)


def apply_gain_filter(
    X: np.ndarray,
    gain: GainCurve,
    *,
    axis: int = -1,
) -> np.ndarray:
    """
    Apply a GainCurve to an array of signals along `axis` using rFFT.

    Intended for (channels, times) arrays where axis=-1.

    Args:
        X: array with time axis
        gain: GainCurve for that time axis length
        axis: axis to filter along

    Returns:
        Filtered array (float64 for real inputs).
    """
    X = np.asarray(X)
    if X.shape[axis] < 2:
        return X.astype(np.float64, copy=True)

    # Move axis to last
    Xm = np.moveaxis(X, axis, -1)
    n_times = Xm.shape[-1]

    # Validate gain length
    expected_len = np.fft.rfft(np.zeros((n_times,), dtype=np.float64)).shape[0]
    if gain.gain.shape[0] != expected_len:
        raise ValueError(
            f"Gain length mismatch: gain has {gain.gain.shape[0]}, expected {expected_len} for n_times={n_times}"
        )

    Xm = Xm.astype(np.float64, copy=False)

    Xf = np.fft.rfft(Xm, axis=-1)
    Yf = Xf * gain.gain  # broadcast over leading dims
    Ym = np.fft.irfft(Yf, n=n_times, axis=-1)

    Y = np.moveaxis(Ym, -1, axis)
    return Y.astype(np.float64, copy=False)


# =============================================================================
# Optional manual low-pass (time-domain filtering) utilities
# =============================================================================

def apply_manual_lowpass(
    X: np.ndarray,
    *,
    fs_hz: float,
    cutoff_hz: float,
    order: int = 4,
    axis: int = -1,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter. Requires SciPy.

    Args:
        X: array with time axis
        fs_hz: sampling rate
        cutoff_hz: cutoff frequency
        order: filter order
        axis: time axis
        zero_phase: if True, uses filtfilt

    Returns:
        Filtered array float64.

    Notes:
        - This is optional and separate from Woldorff gain filtering.
        - If SciPy is unavailable, you can instead compute a manual GainCurve
          using kind='manual_lowpass' and apply_gain_filter().
    """
    require_optional("scipy", "Butterworth filtering")
    from scipy.signal import butter, filtfilt, lfilter  # type: ignore

    X = np.asarray(X, dtype=np.float64)
    if fs_hz <= 0 or cutoff_hz <= 0:
        raise ValueError("fs_hz and cutoff_hz must be > 0")
    nyq = fs_hz / 2.0
    if cutoff_hz >= nyq:
        # No filtering needed
        return X.copy()

    Wn = float(cutoff_hz) / float(nyq)
    b, a = butter(int(order), Wn, btype="low", analog=False)

    if zero_phase:
        Y = filtfilt(b, a, X, axis=axis)
    else:
        Y = lfilter(b, a, X, axis=axis)
    return np.asarray(Y, dtype=np.float64)
