"""Retired per-family pruning implementations, kept only as a test oracle.

These modules were the public pruning entry points until
``less_is_moe.intdim.prune`` replaced them (issue #25). They are no longer part
of the installed package and have no launchers: the equivalence suite imports
them to check, tensor by tensor, that the generic implementation still
reproduces what the published experiments ran.

Pinning their *outputs* as fixtures was the alternative. Floating-point scores
would drift with PyTorch, Transformers, or kernel changes, so the byte-exact
comparison keeps running the original code instead.

Do not add new families here. New backbones go through
``less_is_moe.intdim.discover`` and its registry.
"""
