from time import perf_counter


class Timer:
    def __init__(self) -> None:
        self._handler = lambda: perf_counter() - self._start_time
        self._start_time: float = None

    def __call__(self) -> float:
        return self._handler()

    def __enter__(self):
        self._start_time = perf_counter()
        return self

    def __exit__(self, *_):
        self.stop()

    def start(self):
        if self._start_time is not None:
            raise ValueError("already started")
        self._start_time = perf_counter()
        return self

    def stop(self):
        exit_time = perf_counter() - self._start_time
        self._handler = lambda: exit_time
        return exit_time
