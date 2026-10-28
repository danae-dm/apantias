"""Compatibility shim: archive helpers have moved to apantias.io.archive."""

from apantias.io.archive import archive, cleanup_temp

__all__ = ["archive", "cleanup_temp"]
