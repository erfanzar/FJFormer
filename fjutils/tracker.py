import curses
import json
import subprocess
import time

import IPython.display
import jax
import threading
import os
import posix


def is_notebook():
    """Returns True if the code is being run in a notebook, False otherwise."""
    return os.environ.get("IPYTHON") is not None


# Edited version of Jax-SMI from https://github.com/ayaka14732/jax-smi/
def run(note_book=None, interval: float = 1, dir_prefix: str = '/dev/shm', dpr=True):
    if note_book is None:
        note_book = is_notebook()
    std = curses.initscr() if not note_book else None
    try:
        while True:
            if not note_book and dpr:
                std.clear()
            output = subprocess.run(
                args=['go', 'tool', 'pprof', '-tags', f'{dir_prefix}/memory.prof'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout.decode('utf-8')
            if not note_book and dpr:
                std.addstr(output)
                std.refresh()
            if note_book and dpr:
                IPython.display.clear_output(True)
                print(output)

            with open(f'{dir_prefix}/memory.json', 'w') as fin:
                json.dump({
                    'log': output
                }, fin)
            time.sleep(interval)
    except KeyboardInterrupt:
        curses.endwin()


def get_mem(dir_prefix: str = '/dev/shm') -> str:
    return subprocess.run(
        args=['go', 'tool', 'pprof', '-tags', f'{dir_prefix}/memory.prof'],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.decode('utf-8')


def initialise_tracking(interval: float = 1., dir_prefix: str = '/dev/shm') -> None:
    def inner():
        while True:
            jax.profiler.save_device_memory_profile(f'{dir_prefix}/memory.prof.new')
            posix.rename(f'{dir_prefix}/memory.prof.new', f'{dir_prefix}/memory.prof')
            time.sleep(interval)

    thread = threading.Thread(target=inner, daemon=True)
    thread.start()


def threaded_log(interval: float = 1., dir_prefix: str = '/dev/shm', save_mem_json: bool = False) -> threading.Thread:
    note_book = is_notebook()

    def show_():

        std = curses.initscr() if not note_book else None
        try:
            while True:
                mem_info = get_mem()
                if not note_book:
                    std.clear()
                    std.addstr(mem_info)
                    std.refresh()
                if note_book:
                    IPython.display.clear_output(True)
                    print(mem_info)
                if save_mem_json:
                    with open(f'{dir_prefix}/memory.json', 'w') as fin:
                        json.dump({
                            'log': mem_info
                        }, fin)
                time.sleep(interval)
        except KeyboardInterrupt:
            curses.endwin()

    thread = threading.Thread(
        target=show_
    )
    return thread
