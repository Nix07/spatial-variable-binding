"""One-call repo + vendored-TransformerLens sys.path bootstrap.

The experiments need the repo root (for `analysis_utils`, `metrics`,
`vision_language_prompts`, `component`) and the vendored TransformerLens
(for `HookedVLTransformer`) on `sys.path` — and the vendored TransformerLens
path MUST be inserted before `import transformer_lens`, or the pip-installed
copy wins and lacks `HookedVLTransformer`.

Call `setup()` early, before any TransformerLens or analysis_utils import.
"""

from __future__ import annotations

import sys
from pathlib import Path

# shared/paths.py → parents[1] = repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
TLENS_ROOT = REPO_ROOT / "third_party" / "TransformerLens"


def setup() -> None:
    """Idempotently insert the repo root and the vendored TransformerLens at
    the front of sys.path. Safe to call multiple times."""
    for p in (str(TLENS_ROOT), str(REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
