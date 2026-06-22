# recall. — Constants / Config
# Single source of truth for shared defaults.

import os

# Default database path (relative to project root, or absolute).
# CLI and MCP server both use this, so memories are always in sync.
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "recall_p0.db")
