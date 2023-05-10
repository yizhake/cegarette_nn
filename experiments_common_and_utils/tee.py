from io import TextIOWrapper
import sys


class Tee(TextIOWrapper):
    def __init__(self, filepath, write_mode):
        self.file = open(filepath, write_mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()
