# HSSR-SLOPE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24141844.svg)](https://doi.org/10.5281/zenodo.24141844)

MATLAB and Python code of HSSR-SLOPE, an algorithm that reconstructs the
photoplethysmographic (PPG) waveform recorded at the wrist and detects its
beats, so that inter-beat intervals can be used for heart rate variability
(HRV) analysis.

The algorithm is described in:

> S. Costantini, A.M. Bianchi, *Towards Reliable PPG-based Heart Rate
> Variability Analysis: Applications to Human Activity and Emotion
> Recognition* (submitted).

## How it works

1. **Pre-processing.** The PPG signal is z-scored, normalised by its RMS
   envelope (2 s window) and band-pass filtered between 0.67 and 8 Hz.
2. **Noise surrogate.** The magnitude of the 3-axis acceleration is used as a
   reference of the motion noise.
3. **Harmonic bands.** The spectrum of the PPG signal is weighted by a Wiener
   filter built from the PPG and noise spectra. Its peak is the characteristic
   frequency of the PPG (`fpeak`). The first and second harmonic bands are
   `fpeak ± HT` and `2·fpeak ± HT`.
4. **Decomposition.** Singular Spectrum Analysis (SSA) splits the PPG signal
   into `RO` components.
5. **Reconstruction.** A component is kept if its main frequency lies in one
   of the two bands and if it is not more correlated with the noise surrogate
   than with the PPG signal (tolerance `CT`). The components kept are summed.
6. **Beat detection.** The SLOPE algorithm finds the peaks of the
   reconstructed signal with an adaptive threshold.

| Parameter | Meaning | Default |
|---|---|---|
| RO | Reconstruction Order: number of SSA components | 2 × PPG sample frequency |
| HT | Harmonics Tolerance: half-width of the harmonic bands (Hz) | 0.15 |
| CT | Correlation Tolerance of the noise criterion | 0.025 |

When `fpeak` is not given, the estimated value is shown on the spectra and can
be confirmed (space bar) or corrected (left click). This check can be switched
off.

## Repository content

```
matlab/   HSSR_SLOPE.m, SSA.m, PPG_FPDetection_Slope.m, HSSR_Optimizer.m,
          example_hssr_slope.m, example_optimizer.m,
          robustness_importance.m, synthetic_recording.m
python/   hssr_slope package, examples, tests
data/     sensitivity analysis results of the paper
```

## MATLAB

Requires the Signal Processing Toolbox and the Statistics and Machine Learning
Toolbox (the Wavelet Toolbox only if no acceleration is given).

```matlab
settings.fs_ppg = 64;   % Hz
settings.fs_acc = 32;   % Hz
[idx_peaks, ibi, fpeak] = HSSR_SLOPE(ppg, acc, settings);
```

`ibi` contains the time of each beat (s), the inter-beat interval (ms) and a
flag for the intervals marked as outliers (Grubbs test). Run
`example_hssr_slope.m` for a complete example on a simulated signal.

## Python

```
cd python
pip install .
```

```python
from hssr_slope import hssr_slope

result = hssr_slope(ppg, acc, fs_ppg=64, fs_acc=32)
result.peaks   # sample indices of the beats
result.ibi     # time (s) and inter-beat interval (ms)
```

Examples are in `python/examples`, tests can be run with `pytest`.

The Python version follows the MATLAB code step by step. Some MATLAB functions
(`envelope`, `bandpass`, `pspectrum`, `pburg`) have no exact counterpart in
SciPy and were rewritten from their documentation, so the two versions can give
slightly different beat positions on the same signal. The results reported in
the article were obtained with the MATLAB version.

## Parameter optimization

`HSSR_Optimizer.m` (MATLAB) and `optimize_parameters` (Python) select the
parameters for a given recording condition. They need a table with the
response variables measured for each tested combination of RO, CT and HT, for
instance the beat detection F1-score and the limits of agreement of the HRV
features against a reference ECG. A second-order regression model is fitted to
each response, and the combination with the highest Global Desirability is
returned.

Each response has an importance value. Raising the importance of a response
moves the optimum towards the parameters that improve it, e.g. the limits of
agreement of LF and HF for a frequency-domain analysis.

`example_optimizer` runs the procedure on `data/sensitivity_walking.csv`, the
sensitivity analysis of the Walking condition of the PPG-DaLiA dataset.

## Data

`data/Sensitivity_Analysis_PPGDalia.xlsx` and
`data/Sensitivity_Analysis_WESAD.xlsx` contain the full sensitivity analysis of
the paper: for each of the 270 combinations of RO, CT and HT, the quartiles of
the beat detection F1-score and the limits of agreement of AVNN, SDNN, RMSSD,
LF and HF, in eight conditions of the PPG-DaLiA and WESAD datasets.
`robustness_importance` (MATLAB and Python) uses them to check how much the
selected parameters depend on the importance values (Supplementary Table 4 of
the paper).

The raw recordings are not included. They are available from the authors of
PPG-DaLiA (Reiss et al., 2019) and WESAD (Schmidt et al., 2018).

## Citation

If you use this code, please cite the article above. Citation metadata are in
`CITATION.cff`.

## References

- R. Vautard, M. Ghil, Singular spectrum analysis in nonlinear dynamics, with applications to paleoclimatic time series, Physica D 35 (1989) 395–424. https://doi.org/10.1016/0167-2789(89)90077-8
- M. Ghil, M.R. Allen, M.D. Dettinger, K. Ide, D. Kondrashov, M.E. Mann, A.W. Robertson, A. Saunders, Y. Tian, F. Varadi, P. Yiou, Advanced spectral methods for climatic time series, Rev. Geophys. 40 (2002) 3-1–3-41. https://doi.org/10.1029/2000RG000092
- P. Reali, R. Lolatto, S. Coelli, G. Tartaglia, A.M. Bianchi, Information retrieval from photoplethysmographic sensors: a comprehensive comparison of practical interpolation and breath-extraction techniques at different sampling rates, Sensors 22 (2022) 1428. https://doi.org/10.3390/s22041428
- G. Derringer, R. Suich, Simultaneous optimization of several response variables, J. Qual. Technol. 12 (1980) 214–219. https://doi.org/10.1080/00224065.1980.11980968
- L. Vera Candioti, M.M. De Zan, M.S. Cámara, H.C. Goicoechea, Experimental design and multiple response optimization. Using the desirability function in analytical methods development, Talanta 124 (2014) 123–138. https://doi.org/10.1016/j.talanta.2014.01.034
- A. Reiss, I. Indlekofer, P. Schmidt, K. Van Laerhoven, Deep PPG: large-scale heart rate estimation with convolutional neural networks, Sensors 19 (2019) 3079. https://doi.org/10.3390/s19143079
- P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, K. Van Laerhoven, Introducing WESAD, a multimodal dataset for wearable stress and affect detection, in: Proc. 20th ACM Int. Conf. Multimodal Interact., ACM, 2018, pp. 400–408. https://doi.org/10.1145/3242969.3242985

## License

BSD 2-Clause, see `LICENSE`.
