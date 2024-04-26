"""The setup code for the library."""

import pathlib

from setuptools import setup


CWD = pathlib.Path(__file__).absolute().parent


def get_version() -> str:
    """Pulls the version from the __init__ file in the library.

    Raises:
        RuntimeError: Raises when no __version__ was found.

    Returns:
        str: The current version of the library.
    """
    path = CWD / "dmsp" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


def get_long_description() -> str:
    """Reads the long description from the README.md file.

    Returns:
        str: The long description.
    """

    with open("README.md") as fh:
        long_description = "".join(fh.readlines())
    return long_description


setup(
    name="dmsp",
    version=get_version(),
    long_description=get_long_description(),
)
