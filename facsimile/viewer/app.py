"""Page-image viewer with retrieval highlights.

The viewer is served as part of the main FastAPI app at ``GET /viewer``.
``launch_viewer`` starts the full server with uvicorn so the viewer is
accessible in a browser alongside the REST API.
"""


def launch_viewer(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the facsimile FastAPI server (API + viewer) with uvicorn.

    Args:
        host: Network interface to bind.  ``"0.0.0.0"`` makes the server
            reachable from other machines; ``"127.0.0.1"`` restricts to
            localhost.
        port: TCP port to listen on.
    """
    import uvicorn

    from facsimile.api.main import app

    uvicorn.run(app, host=host, port=port)
