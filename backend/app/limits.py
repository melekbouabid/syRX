"""Request body limit, checked on the Content-Length header before anything is read."""

from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class BodySizeLimit:
    """Answer 413 when a request announces a body above `max_bytes`.

    The scan route checks the size of the file it receives again, which also covers
    uploads sent without a Content-Length (chunked).
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            length = dict(scope["headers"]).get(b"content-length")
            if length is not None and length.isdigit() and int(length) > self.max_bytes:
                response = JSONResponse(
                    {"detail": f"Request body larger than {self.max_bytes} bytes."},
                    status_code=413,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
