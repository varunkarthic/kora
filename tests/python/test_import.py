import kora


def test_native_module_exports_runtime_identity():
    assert kora.version() == kora.__version__
    assert isinstance(kora.version(), str)
    assert kora.version().count(".") == 2
    assert kora.schema_version() >= 1
