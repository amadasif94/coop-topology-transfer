"""Private function bindings, not module monkey-patches.

The existing IEEE33 runner already uses this technique for its model factory.
It lets the active IEEE33 source and checkpoint identity remain untouched while
reusing exactly the same loops. Closures (including zero-argument super) survive.
"""
from types import FunctionType


def bind(function, namespace):
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result
