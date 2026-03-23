"""Legacy compatibility shim for GEMORNA's shared generation library.

The upstream `libg2m.so` binary imports `utils.utils_cds` even when running the
3'UTR generation path. This shim keeps that import path available while making
it explicit that CDS code is only present for binary compatibility.

For new 3'UTR work, do not depend on this module.
"""

from src.utils.utils_cds_legacy import *  # noqa: F401,F403
