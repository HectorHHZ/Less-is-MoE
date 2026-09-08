"""Neuron-pruning implementations used by Less-is-MoE.

The released experiments score each intermediate neuron with mean absolute
gradient magnitude (``mean_abs_grad``).  The historical function and summary
names containing ``pure_gradient`` are retained for checkpoint compatibility.
"""

SCORE_METRIC = "mean_abs_grad"

__all__ = ["SCORE_METRIC"]
