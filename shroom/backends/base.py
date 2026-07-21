from abc import ABC, abstractmethod


class VLMBackend(ABC):
    @abstractmethod
    def generate(self, image, text, n=1, temperature=0.5):
        """Return a list of `n` raw model output strings for (image, text)."""
        raise NotImplementedError


class MockBackend(VLMBackend):
    """Deterministic backend for tests / harness dry-runs.

    script: either a callable(text, n) -> list[str], or a list where each
    generate() call returns the next element (itself a list[str]).
    """

    def __init__(self, script):
        self._script = script
        self._i = 0
        self.calls = []

    def generate(self, image, text, n=1, temperature=0.5):
        self.calls.append((text, n))
        if callable(self._script):
            return self._script(text, n)
        out = self._script[self._i]
        self._i += 1
        return out
