"""
Compatibility Module - Fixes for Python 3.12 and dependency issues

This module provides compatibility fixes that must be imported BEFORE
any problematic dependencies (like alpaca-trade-api).

Note: With urllib3 1.24.3, no patching is needed as it bundles six internally.
This module is kept for future compatibility fixes.
"""

def _patch_urllib3():
    """Patch urllib3 if needed (currently no-op for urllib3 1.24.3)."""
    # urllib3 1.24.3 bundles six internally, so no patching needed
    pass

# Run the patch immediately when this module is imported
_patch_urllib3()

