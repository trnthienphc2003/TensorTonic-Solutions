def scalar_expression_partials(a: float, b: float, c: float, h: float) -> tuple[float, float, float, float]:
    """
    Returns the expression value and numerical partials for a, b, and c.
    """
    a, b, c, h = map(float, (a, b, c, h))
    def _d(a, b, c):
        return a * b + c

    orig_val = _d(a, b, c)
    da = (_d(a + h, b, c) - orig_val) / h
    db = (_d(a, b + h, c) - orig_val) / h
    dc = (_d(a, b, c + h) - orig_val) / h
    return orig_val, da, db, dc