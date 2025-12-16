ADJAR_64BIT

A 64-bit Adjacent Response (ADJAR) EEG Overlap Correction Toolkit

Project Background and Motivation

This project was initiated at the request of Prof. Johan L. Kenemans
(https://www.uu.nl/staff/JLKenemans
)
with the goal of creating a modern, fully 64-bit compatible implementation of the ADJAR (Adjacent Response) algorithm for EEG/ERP research.

The original ADJAR method, introduced in the early 1990s, remains scientifically relevant but existing implementations are either outdated, platform-limited (32-bit), or tightly coupled to legacy software environments. This project addresses those limitations by providing:

A clean, extensible Python implementation

Full 64-bit numerical support

A desktop GUI suitable for non-programmers

Reproducible testing and validation against synthetic ground-truth data

The result is a research-grade tool that can be used independently of proprietary EEG software.

What This Project Implements
Core ADJAR Functionality

Level 1 ADJAR correction

ISI-based subaveraging

Estimation of the preceding response waveform R_prev(t)

Prediction of overlap via convolution with the preceding-event distribution D_pre(t)

Correction of the conventional ERP

Level 2 ADJAR correction

Iterative refinement of overlap estimation

Trial-level overlap subtraction

Convergence monitoring using RMS criteria

Improved stability relative to Level 1

Numerical and Performance Design

Fully 64-bit float (float64) throughout

FFT-based convolution for efficiency

Optional gain filtering (Woldorff-style placeholder or manual low-pass)

Deterministic behavior suitable for scientific validation

Project Structure (High-Level)
ADJAR_64BIT/
├─ src/adjar64/
│  ├─ core/        # ADJAR math and pipelines
│  ├─ io/          # NPZ/CSV loading and exporting
│  ├─ accel/       # FFT-based acceleration
│  ├─ plots/       # Matplotlib figure builders
│  └─ gui/         # PySide6 desktop GUI
│
├─ tests/          # Automated validation tests
├─ scripts/        # Synthetic data generation and demos
├─ pyproject.toml
└─ README.md


The design strictly separates:

Scientific logic (core)

Input/output

User interface

Validation/testing

This separation ensures correctness, reproducibility, and long-term maintainability.

Validation and Testing

The implementation is validated using synthetic EEG datasets with known ground truth, ensuring that:

FFT-based convolution matches direct convolution

Level 1 correction improves results when the data matches ADJAR assumptions

Level 2 correction converges and further improves results

Numerical behavior is stable and reproducible

All tests pass using pytest:

python -m pytest

Graphical User Interface (GUI)

A desktop GUI is provided using PySide6 (Qt).
The GUI allows users to:

Load EEG/ERP datasets from NPZ

Run Level 1 and Level 2 ADJAR correction

Visually compare:

Conventional ERP

Level 1 corrected ERP

Level 2 corrected ERP

Export corrected results and metadata

The GUI is launched with:

python -m adjar64.gui.main

Intended Use and Licensing

This software is intended exclusively for research and academic use, particularly for EEG/ERP methodology development and evaluation.

Free to use

Non-commercial

Licensed under Utrecht University research licensing

No warranty is provided; users are responsible for validating suitability for their specific experimental designs

Development Notes

This project was designed, implemented, and validated by:

Jacob Won Joon Bae
Student Number: 0732567
MSc Artificial Intelligence
Utrecht University

GPT 5.2 was used as a development assistant to:

Plan the software architecture

Assist with debugging and refactoring

Help design validation strategies and tests
All scientific and implementation decisions were reviewed and executed by the author.

Academic Reference

If you use or adapt this software, please cite the original ADJAR method:

Woldorff, M. G. (1993).
Distortion of ERP averages due to overlap from temporally adjacent ERPs: Analysis and correction.
Psychophysiology, 30(1), 98–119.

The original paper is included in this repository as:

woldorff_adjar_1993.pdf

Final Remarks

This project modernizes a classic ERP correction method into a transparent, testable, and extensible research tool. It is intended to support methodological research, replication studies, and teaching, while remaining faithful to the conceptual foundations of the original ADJAR framework.

For questions, extensions, or academic collaboration, please contact the author through Utrecht University channels.