# Recordings

Live model responses, one file per card (`FERN-2.json`), written by `taskgen generate --mode record` with your own `ANTHROPIC_API_KEY`. Replay mode (the default) answers only from these files and fails with the command to run when a card has none. Empty until a live run is recorded.

Each file holds the response text, the model that served it, the stop reason, token usage (cache reads and writes included), the request id, and a SHA-256 of the request (model, system block, messages, output schema). A recording replays only for the same request, so changing the prompt, the harness guide, the API or the card makes it stale, and replay says which part changed.
