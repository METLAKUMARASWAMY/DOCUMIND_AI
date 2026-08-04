from __future__ import annotations

import os


# Legacy research-case API tests run only under this explicit test profile.
os.environ.setdefault("ENABLE_DEMO_API", "true")
