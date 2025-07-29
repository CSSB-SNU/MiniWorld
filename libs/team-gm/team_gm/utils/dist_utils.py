import functools
import os


def is_rank_zero():
    return os.environ.get("RANK", "0") == "0"


def rank_zero_only(func):
    @functools.wraps(func)
    def wrapper_fn(*args, **kwargs):
        if is_rank_zero():
            return func(*args, **kwargs)
        return None

    return wrapper_fn
