"""
The VS Code extension's local sidecar: indexes a Python codebase into a code graph (static
analysis, no model), retrieves context with the shared local push PPR (graphrag_core), answers
questions and proposes edits with a local model. Talks to the extension over stdio (server.py).
"""
