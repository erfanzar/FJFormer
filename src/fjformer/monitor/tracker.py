"""Utilities for memory profiling in JAX."""

import curses
import json
import os
import re
import subprocess
import threading
import time
from typing import Any, Dict, Optional

import IPython.display
import jax
import jax.profiler

try:
    import posix
except ModuleNotFoundError:
    posix = None


def is_notebook() -> bool:
    """Returns True if the code is being run in a notebook, False otherwise."""
    return os.environ.get("IPYTHON") is not None


def run(
    note_book: Optional[bool] = None,
    interval: float = 1,
    dir_prefix: str = "/dev/shm",
    display_output: bool = True,
) -> None:
    """Periodically prints JAX memory usage information.

    Runs the `go tool pprof` command in a loop, capturing its output and either
    printing it to stdout or displaying it in a Jupyter notebook cell.

    Args:
        note_book: Whether the code is running in a notebook. If None, it will
            be auto-detected.
        interval: The time interval (in seconds) between refreshes.
        dir_prefix: The directory to store the memory profile.
        display_output: Whether to print the output to the console or a
            Jupyter notebook cell.
    """
    if note_book is None:
        note_book = is_notebook()

    std = curses.initscr() if not note_book else None
    try:
        while True:
            if not note_book and display_output:
                std.clear()
            output = subprocess.run(
                args=["go", "tool", "pprof", "-tags", f"{dir_prefix}/memory.prof"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout.decode("utf-8")

            if display_output:
                if not note_book:
                    std.addstr(output)
                    std.refresh()
                else:
                    IPython.display.clear_output(True)
                    print(output)

            with open(f"{dir_prefix}/memory.json", "w") as fin:
                json.dump({"log": output}, fin)
            time.sleep(interval)
    except KeyboardInterrupt:
        curses.endwin()


def get_memory_information(dir_prefix: str = "/dev/shm") -> str:
    """Retrieves JAX memory usage information using `go tool pprof`.

    Args:
        dir_prefix: The directory where the memory profile is stored.

    Returns:
        The output of the `go tool pprof` command as a string.
    """
    return subprocess.run(
        args=["go", "tool", "pprof", "-tags", f"{dir_prefix}/memory.prof"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.decode("utf-8")


def initialise_tracking(interval: float = 1.0, dir_prefix: str = "/dev/shm") -> None:
    """Starts a daemon thread to periodically save the memory profile to disk.

    This function starts a background thread that continuously saves the JAX
    memory profile to a file. It ensures atomicity of the file update using
    `posix.rename` if available, otherwise falls back to a simple rename.

    Args:
        interval: The time interval (in seconds) between memory profile saves.
        dir_prefix: The directory to store the memory profile.
    """

    def _save_profile():
        while True:
            jax.profiler.save_device_memory_profile(f"{dir_prefix}/memory.prof.new")
            if posix is not None:
                os.rename(f"{dir_prefix}/memory.prof.new", f"{dir_prefix}/memory.prof")
            else:
                posix.rename(
                    f"{dir_prefix}/memory.prof.new", f"{dir_prefix}/memory.prof"
                )
            time.sleep(interval)

    thread = threading.Thread(target=_save_profile, daemon=True)
    thread.start()


def threaded_log(
    interval: float = 1.0, dir_prefix: str = "/dev/shm", save_mem_json: bool = False
) -> threading.Thread:
    """Starts a thread to periodically log memory information.

    This function launches a background thread that continuously monitors and
    either prints or displays the JAX memory usage information. Optionally,
    it can also save the memory information to a JSON file.

    Args:
        interval: The time interval (in seconds) between memory logs.
        dir_prefix: The directory to save the memory information JSON file.
        save_mem_json: Whether to save the memory information to a JSON file.

    Returns:
        The thread handling the memory logging.
    """
    note_book = is_notebook()

    def _show_memory_info():
        std = curses.initscr() if not note_book else None
        try:
            while True:
                mem_info = get_memory_information()
                if not note_book:
                    std.clear()
                    std.addstr(mem_info)
                    std.refresh()
                else:
                    IPython.display.clear_output(True)
                    print(mem_info)
                if save_mem_json:
                    with open(f"{dir_prefix}/memory.json", "w") as fin:
                        json.dump({"log": mem_info}, fin)
                time.sleep(interval)
        except KeyboardInterrupt:
            curses.endwin()

    thread = threading.Thread(target=_show_memory_info)
    return thread


def get_capacity_matrix(dir_prefix: str = "/dev/shm") -> Dict[str, Dict[str, Any]]:
    """Parses memory information and returns a dictionary with capacity details.

    This function extracts memory usage information from the output of the
    `go tool pprof` command and structures it into a dictionary. The dictionary
    maps memory pool names to their usage details, including used memory,
    usage percentage, process information, and full capacity.

    Args:
        dir_prefix: The directory where the memory profile is stored.

    Returns:
        A dictionary containing memory capacity information.
    """
    pattern = r"(\d+\.\d+\wB) \((\d+\.\d+%)\): (\w+)(\(.*?\))?"

    def _calculate_full_size(size: str, percent: str) -> float:
        size_in_gb = float(re.search(r"(\d+\.\d+)GB", size).group(1))
        percent_value = 100 / float(re.search(r"(\d+\.\d+)%", percent).group(1))
        return size_in_gb * percent_value

    matches = re.findall(pattern, get_memory_information(dir_prefix=dir_prefix))
    information = {}
    try:
        for match in matches:
            information[match[2]] = {
                "Used": match[0],
                "Usage Percent": match[1],
                "Process": match[3][1:] if match[3] else "∞",
                "Full Capacity": _calculate_full_size(match[0], match[1]),
            }
    except (ArithmeticError, AttributeError, KeyError, ValueError):
        pass
    return information
