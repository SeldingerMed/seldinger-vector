"""OR-Audit: vendor-neutral robotic surgical skill and safety attestation.

See ``docs/PLAN.md`` for the product thesis. The package is layered to match
the architecture in section 7.1, and the layering is load-bearing:

``domain``
    Entities, closed vocabularies, and the invariants that must hold
    everywhere. No I/O.
``audit``
    Deterministic serialization and the tamper-evident append-only trail that
    every other layer writes to.

Later phases add ``ingest``, ``deid``, ``perception``, ``scoring``,
``decision``, and ``api`` on top of these two.
"""

from __future__ import annotations

from or_audit.version import PACKAGE_VERSION as __version__

__all__ = ["__version__"]
