"""Track exact temporary files created by one tool resource, never journal-supplied paths."""

import os
import stat
from pathlib import Path


class SpillRegistry:
    def __init__(self) -> None:
        self._files: dict[Path, tuple[int, int]] = {}

    def register(self, path: str, descriptor: int) -> None:
        identity = os.fstat(descriptor)
        self._files[Path(path)] = (identity.st_dev, identity.st_ino)

    def close(self) -> None:
        errors: list[Exception] = []
        for path, identity in list(self._files.items()):
            try:
                current = path.lstat()
                if stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == identity:
                    path.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                errors.append(error)
                continue
            self._files.pop(path, None)
        if errors:
            raise ExceptionGroup("Could not remove owned spill files", errors)
