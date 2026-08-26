"""Flag-gated legacy research-case stack. See README.md in this package.

Nothing here runs in the shipping configuration. These modules are reachable only
through ``app/api/routes.py`` and ``app/api/finance_routes.py``, which
``app/main.py`` mounts under ``/api/v1/demo-legacy`` and only when
``ENABLE_DEMO_API=true`` (default: false).

The production document-intelligence implementation is ``app/services/vision/``.
"""
