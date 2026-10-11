"""Python package facade for the KORA C++ runtime."""

from ._core import schema_version, version

__all__ = ["schema_version", "version"]
__version__ = version()
