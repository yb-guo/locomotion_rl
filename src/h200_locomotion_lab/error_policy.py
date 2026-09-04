"""Explicit exception policy for recoverable command and probe failures.

Unknown custom ``Exception`` subclasses intentionally propagate. This keeps CLI
artifact generation resilient to expected Python/runtime failures without turning
programming defects into silent fallback values.
"""

RECOVERABLE_RUNTIME_ERRORS: tuple[type[Exception], ...] = (
    ArithmeticError,
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
