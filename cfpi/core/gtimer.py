import portalocker
from gtimer import *
from gtimer import stamp as st


def stamp(*args, **kwargs):
    with portalocker.Lock(f"/tmp/{args[0]}"):
        st(*args, unique=False, **kwargs)
