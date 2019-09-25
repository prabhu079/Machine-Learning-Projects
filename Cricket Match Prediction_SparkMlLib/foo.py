def foo():
    globals()["foo"] = lambda x: "foo {0}".format(x)
    # Exports all entries from globals which start with foo
    __all__ = [x for x in globals() if x.startswith("foo")]