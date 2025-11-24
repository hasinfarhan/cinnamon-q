"""
Compatibility Module - Fixes for Python 3.12 and dependency issues

This module provides compatibility fixes that must be imported BEFORE
any problematic dependencies (like alpaca-trade-api).
"""

import sys
import types

# Fix urllib3.packages.six for Python 3.12 compatibility
# urllib3 1.26.x expects six to be in urllib3.packages.six, but it's not bundled
def _patch_urllib3():
    """Patch urllib3 to use system six package."""
    try:
        import six
        
        # Pre-register urllib3.packages.six before urllib3 is imported
        # This allows urllib3 to find six in the expected location
        if 'urllib3.packages' not in sys.modules:
            packages = types.ModuleType('packages')
            sys.modules['urllib3.packages'] = packages
        
        if 'urllib3.packages.six' not in sys.modules:
            # Make urllib3.packages.six point to the system six
            sys.modules['urllib3.packages.six'] = six
            
            # Also ensure six.moves is accessible
            if hasattr(six, 'moves'):
                if 'urllib3.packages.six.moves' not in sys.modules:
                    sys.modules['urllib3.packages.six.moves'] = six.moves
                    
                    # Ensure http_client is available
                    if hasattr(six.moves, 'http_client'):
                        sys.modules['urllib3.packages.six.moves.http_client'] = six.moves.http_client
        
    except ImportError:
        # If six isn't available, we can't patch
        pass

# Run the patch immediately when this module is imported
_patch_urllib3()

