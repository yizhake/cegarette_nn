import sys
import logging
from typing import TextIO


class AbstractionRefinementLogger(logging.Logger):
    def __init__(
        self,
        stream: TextIO = sys.stdout,
        level=logging.DEBUG,
        show_time=True,
        show_level=False,
    ) -> None:
        super().__init__("abstraction-refinement", level=level)
        self.handlers = []
        sh = logging.StreamHandler(stream)
        log_format = ""
        if show_time:
            log_format += "[%(asctime)s] "
        if show_level:
            log_format += "%(levelname)s "
        log_format += "%(message)s"
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        sh.setFormatter(formatter)
        self.addHandler(sh)

        self.stream = stream

    def multiline_info(self, title, *lines: str):
        self.info(f"{title} >>>")
        sep = ""
        for l in lines:
            l = "\n  ".join(l.split("\n"))
            print(f"{sep}* {l}", file=self.stream, end="")
            sep = "\n"
        print(f"<<<", file=self.stream)
