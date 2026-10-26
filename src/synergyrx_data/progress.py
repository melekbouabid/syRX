"""Progress bars (rich), silent when output is not a terminal."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import TracebackType

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def _ignore_bytes(done: int, size: int | None) -> None:
    return None


def _ignore_items(n: int = 1) -> None:
    return None


class ProgressReporter:
    """Hands out progress callbacks; does nothing when disabled or not attached to a TTY."""

    def __init__(self, enabled: bool = True) -> None:
        self.console = Console(stderr=True)
        self.enabled = enabled and sys.stderr.isatty()
        self._bytes: Progress | None = None
        self._items: Progress | None = None

    def __enter__(self) -> ProgressReporter:
        if self.enabled:
            self._bytes = Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=True,
            )
            self._items = Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=True,
            )
            self._bytes.start()
            self._items.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for progress in (self._bytes, self._items):
            if progress is not None:
                progress.stop()
        self._bytes = self._items = None

    def download(self, label: str, total: int | None) -> Callable[[int, int | None], None]:
        """Callback for byte progress: `callback(done_bytes, total_bytes)`."""
        if self._bytes is None:
            return _ignore_bytes
        progress = self._bytes
        task: TaskID = progress.add_task(label, total=total)

        def update(done: int, size: int | None) -> None:
            progress.update(task, completed=done, total=size or total)

        return update

    def counter(self, label: str, total: int) -> Callable[[int], None]:
        """Callback for item progress: `advance(n)`."""
        if self._items is None:
            return _ignore_items
        progress = self._items
        task: TaskID = progress.add_task(label, total=total)

        def advance(n: int = 1) -> None:
            progress.advance(task, n)

        return advance
