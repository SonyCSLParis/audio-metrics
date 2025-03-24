def apa(d_y_x, d_y_xp, d_x_xp):
    d_y_x = max(0, d_y_x)
    d_y_xp = max(0, d_y_xp)
    d_x_xp = max(0, d_x_xp)
    numerator = d_y_xp - d_y_x
    denominator = d_x_xp
    if abs(numerator) > denominator:
        denominator = abs(numerator)
    if denominator <= 0:
        return 0.0
    return 1 / 2 + numerator / (2 * denominator)
