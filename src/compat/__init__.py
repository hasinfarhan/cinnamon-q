"""
Compatibility Module - Fixes for Python 3.12 and dependency issues

This module provides compatibility fixes that must be imported BEFORE
any problematic dependencies (like alpaca-trade-api).
"""

import sys
import types

# Fix urllib3.packages.six.moves for Python 3.12 compatibility
# This must run BEFORE urllib3 is imported
def _patch_urllib3():
    """Patch urllib3 to work with Python 3.12."""
    # Use import hook to patch urllib3 before it's fully loaded
    import importlib.util
    
    # Check if urllib3 is already in sys.modules
    if 'urllib3' in sys.modules:
        urllib3 = sys.modules['urllib3']
    else:
        # Pre-create the structure before urllib3 is imported
        # This is a bit hacky but necessary for Python 3.12
        try:
            # Create mock urllib3 module structure
            urllib3_mock = types.ModuleType('urllib3')
            urllib3_mock.packages = types.ModuleType('packages')
            urllib3_mock.packages.six = types.ModuleType('six')
            urllib3_mock.packages.six.moves = types.ModuleType('moves')
            
            # Try to import http_client from six.moves
            try:
                import six
                from six.moves import http_client
                urllib3_mock.packages.six.moves.http_client = http_client
            except (ImportError, AttributeError):
                # Create minimal mock
                urllib3_mock.packages.six.moves.http_client = types.ModuleType('http_client')
            
            # Pre-register in sys.modules (this is the key!)
            sys.modules['urllib3.packages'] = urllib3_mock.packages
            sys.modules['urllib3.packages.six'] = urllib3_mock.packages.six
            sys.modules['urllib3.packages.six.moves'] = urllib3_mock.packages.six.moves
            sys.modules['urllib3.packages.six.moves.http_client'] = urllib3_mock.packages.six.moves.http_client
            
        except Exception:
            # If pre-registration fails, try patching after import
            pass

# Run the patch immediately when this module is imported
_patch_urllib3()

