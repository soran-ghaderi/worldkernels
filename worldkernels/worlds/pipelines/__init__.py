r"""Pipelines composed by world adapters.

Pipelines own model loading, encoding, denoising, and decoding for one model
family. Adapters under ``worldkernels/worlds/adapters/`` subclass
:class:`AbstractWorld` and compose a pipeline rather than inheriting from a
shared world base class.
"""
