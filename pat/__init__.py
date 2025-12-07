"""PAT V4 - Privacy Assurance Toolkit."""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Return the installed package version or a development placeholder."""

    try:
        return version("pat")
    except PackageNotFoundError:
        return "0.0.0-dev"


__all__ = ["get_version"]

