from functools import wraps

class _StrictProtectedMeta(type):
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        cls._allow_internal_write = False
        cls._allow_internal_access = False
        return cls

    def __getattr__(cls, name):
        
        # Called ONLY if attribute does not exist
        if name.startswith("_") and not cls._allow_internal_access:
            raise AttributeError(
                f"Direct read of protected attribute '{name}' is prohibited. "
                "Use class getter methods."
            )
        raise AttributeError(name)

    def __setattr__(cls, name, value):
        if name in ("_allow_internal_write", "_allow_internal_access"):
            return super().__setattr__(name, value)

        if name.startswith("_"):
            if not getattr(cls, "_allow_internal_write", False):
                raise AttributeError(
                    f"Direct write to protected attribute '{name}' is prohibited. "
                    "Use class setter methods."
                )

        return super().__setattr__(name, value)

class _StrictProtectedMeta_old(type):
    # # Issue with docs
    # _INTROSPECTION_ALLOWLIST = {
    # "__doc__", "__module__", "__dict__", "__mro__",
    # "__bases__", "__qualname__", "__name__", "__annotations__",
    # "__class__", "__subclasshook__", "__init_subclass__"
    # }
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        # initialize internal flags so they exist
        cls._allow_internal_write = False
        cls._allow_internal_access = False
        return cls

    def __getattribute__(cls, name):
        # # Always allow introspection-critical attributes (For sphinx documentation)
        # if name in _StrictProtectedMeta._INTROSPECTION_ALLOWLIST:
        #     return super().__getattribute__(name)

        # Always allow these flags internally
        if name in ("_allow_internal_access", "_allow_internal_write"):
            return super().__getattribute__(name)

        # Protect ANY class attribute starting with "_"
        if name.startswith("_"):
            if not getattr(cls, "_allow_internal_access", False):
                raise AttributeError(
                    f"Direct read of protected attribute '{name}' is prohibited. "
                    "Use class getter methods."
                )

        return super().__getattribute__(name)

    def __setattr__(cls, name, value):
        # allow internal flags
        if name in ("_allow_internal_write", "_allow_internal_access"):
            return super().__setattr__(name, value)

        if name.startswith("_"):
            if not getattr(cls, "_allow_internal_write", False):
                raise AttributeError(
                    f"Direct write to protected attribute '{name}' is prohibited. "
                    "Use class setter methods."
                )

        return super().__setattr__(name, value)

def _internal_classmethod(method):
    @wraps(method)
    def wrapper(cls, *args, **kwargs):
        prev_access = cls._allow_internal_access
        prev_write = cls._allow_internal_write
        cls._allow_internal_access = True
        cls._allow_internal_write = True
        try:
            return method(cls, *args, **kwargs)
        finally:
            cls._allow_internal_write = prev_write
            cls._allow_internal_access = prev_access
    return classmethod(wrapper)
        
class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)
