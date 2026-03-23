"""Legacy compatibility shim for GEMORNA's shared generation library.

This project focuses on 3'UTR generation/fine-tuning, but the upstream
`libg2m.so` binary imports `models.gemorna_cds` during module initialization.
To keep the binary import working without making CDS a first-class part of this
repository, we re-export the original GEMORNA CDS classes here.

Do not build new 3'UTR training code against this module.
Use `src/models/gemorna_runtime.py`, `src/models/gemorna_utr.py`, and
`src/utils/utils_utr.py` as the canonical implementation area instead.
"""

from src.models.gemorna_cds_legacy import *  # noqa: F401,F403
