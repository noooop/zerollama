
import sys
from pathlib import Path


def sys_logging(home=None):
    if home is None:
        home = Path.cwd()

    stdout_name = "stdout.log"
    stderr_name = "stderr.log"
    home.mkdir(parents=True, exist_ok=True)

    out_file = open(home / stdout_name, "w")
    err_file = open(home / stderr_name, "w")

    class Unbuffered:
        def __init__(self, stream, file):
            self.stream = stream
            self.file = file

        def write(self, data):
            self.stream.write(data)
            self.stream.flush()
            self.file.write(data)
            self.file.flush()

        def flush(self):
            pass

        def __del__(self):
            self.file.flush()
            self.file.close()

    sys.stdout = Unbuffered(sys.stdout, out_file)
    sys.stderr = Unbuffered(sys.stderr, err_file)