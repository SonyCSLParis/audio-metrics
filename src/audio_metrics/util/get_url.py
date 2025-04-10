from pathlib import Path
import appdirs
import logging
from urllib import request

from tqdm import tqdm


PACKAGE_NAME = __name__.split(".", maxsplit=1)[0]


def progress_hook(t):
    """
    Wraps tqdm instance. Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).

    Example
    -------

    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)

    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks just transferred [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def download_and_save(url, outfile):
    with tqdm(
        unit="B",
        unit_scale=True,
        leave=True,
        miniters=1,
        desc=f"Downloading {url.split('/')[-1]} to {Path(outfile).parent}",
    ) as t:
        request.urlretrieve(
            url,
            filename=outfile,
            reporthook=progress_hook(t),
        )


def download_url(url):
    cache_dir = Path(appdirs.user_cache_dir(PACKAGE_NAME))
    name = url.rsplit("/", maxsplit=1)[-1]
    fp = cache_dir / name
    fn = fp.as_posix()
    if not fp.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.info(f"Downloading {url} to {fn}")
        try:
            download_and_save(url, fn)
        except Exception as e:
            raise Exception(f"Error downloading {url}") from e
    return fn
