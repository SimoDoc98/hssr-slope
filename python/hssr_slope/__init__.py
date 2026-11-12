"""HSSR-SLOPE: PPG waveform reconstruction and beat detection."""

from .hssr import HSSRResult, hssr_slope
from .optimizer import OptimizationResult, optimize_parameters, plot_desirability
from .slope import slope_beat_detection
from .ssa import ssa

__version__ = "1.1.0"
__all__ = ["hssr_slope", "HSSRResult", "ssa", "slope_beat_detection",
           "optimize_parameters", "plot_desirability", "OptimizationResult"]
