from __future__ import annotations

import numpy as np

from adjar64.accel.fftconv import fft_convolve_1d, fft_convolve_batch


def _direct_conv_1d(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    # NumPy convolution for reference
    y_full = np.convolve(a, b, mode="full")
    if mode == "full":
        return y_full
    if mode == "same":
        start = (b.size - 1) // 2
        end = start + a.size
        return y_full[start:end]
    if mode == "valid":
        if a.size < b.size:
            return y_full[:0]
        start = b.size - 1
        end = start + (a.size - b.size + 1)
        return y_full[start:end]
    raise ValueError("unsupported mode")


def test_fft_convolve_1d_matches_direct_full_same_valid() -> None:
    rng = np.random.default_rng(123)

    for (na, nb) in [(64, 17), (257, 33), (1000, 1), (200, 200), (50, 120)]:
        a = rng.standard_normal(na).astype(np.float64)
        b = rng.standard_normal(nb).astype(np.float64)

        for mode in ("full", "same", "valid"):
            y_fft = fft_convolve_1d(a, b, mode=mode, use_real_fft=True)
            y_ref = _direct_conv_1d(a, b, mode=mode)

            assert y_fft.shape == y_ref.shape
            # FFT numerical differences should be extremely small for float64.
            np.testing.assert_allclose(y_fft, y_ref, rtol=1e-10, atol=1e-10)


def test_fft_convolve_batch_matches_direct_same() -> None:
    rng = np.random.default_rng(456)

    n_ch = 8
    n_times = 512
    k_len = 65

    X = rng.standard_normal((n_ch, n_times)).astype(np.float64)
    h = rng.standard_normal((k_len,)).astype(np.float64)

    Y_fft = fft_convolve_batch(X, h, axis=-1, mode="same", use_real_fft=True)

    # Direct reference per channel
    Y_ref = np.zeros_like(Y_fft)
    for ch in range(n_ch):
        y_full = np.convolve(X[ch], h, mode="full")
        start = (h.size - 1) // 2
        end = start + n_times
        Y_ref[ch] = y_full[start:end]

    np.testing.assert_allclose(Y_fft, Y_ref, rtol=1e-10, atol=1e-10)
