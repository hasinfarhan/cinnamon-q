"""
urllib3.packages.six compatibility patch for Python 3.12

This module patches urllib3 to work with Python 3.12 by creating the
expected packages.six structure before urllib3 is imported.

This must be imported before any module that imports urllib3 (like alpaca_trade_api).
"""

import sys
import types

# Import six first
import six

# Create urllib3.packages.six structure before urllib3 is imported
# We need to do this by manipulating sys.modules
if 'urllib3' not in sys.modules:
    # Create a mock urllib3 module structure
    urllib3_mock = types.ModuleType('urllib3')
    urllib3_mock.packages = types.ModuleType('packages')
    urllib3_mock.packages.six = six
    sys.modules['urllib3'] = urllib3_mock
    sys.modules['urllib3.packages'] = urllib3_mock.packages
    sys.modules['urllib3.packages.six'] = six

# Now import urllib3 - it should use our patched structure
import urllib3

# Ensure the structure persists
if not hasattr(urllib3, 'packages'):
    urllib3.packages = types.ModuleType('packages')
if not hasattr(urllib3.packages, 'six'):
    urllib3.packages.six = six

