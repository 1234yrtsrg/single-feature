import atexit
import os
import time
from dataclasses import dataclass

@dataclass
class Timer:
    t0: float = None

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dt = time.time() - self.t0

class Logger:
    _log_file_path = None
    _log_file_handle = None

    @staticmethod
    def init(log_dir: str, log_name: str):
        os.makedirs(log_dir, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in log_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        Logger._log_file_path = os.path.join(log_dir, f"{timestamp}_{safe_name}_pid{pid}.log")
        Logger._log_file_handle = open(Logger._log_file_path, "a", encoding="utf-8")
        atexit.register(Logger.close)

    @staticmethod
    def close():
        if Logger._log_file_handle is not None:
            Logger._log_file_handle.close()
            Logger._log_file_handle = None

    @staticmethod
    def log_path():
        return Logger._log_file_path

    @staticmethod
    def _emit(msg: str, to_console: bool):
        text = str(msg)
        if Logger._log_file_handle is not None:
            Logger._log_file_handle.write(text + "\n")
            Logger._log_file_handle.flush()
            if to_console:
                print(text, flush=True)
            return

        print(text, flush=True)

    @staticmethod
    def log(msg: str):
        Logger._emit(msg, to_console=False)

    @staticmethod
    def console(msg: str):
        Logger._emit(msg, to_console=True)

    @staticmethod
    def log_step(title: str):
        bar = "-" * 100
        Logger.log(f"\n{bar}\n{title}\n{bar}")
