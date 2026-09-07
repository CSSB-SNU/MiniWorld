"""Minimal mmCIF writing helpers: value quoting, key-value blocks and loops."""


def cif_value(v):
    """Render one value using mmCIF quoting rules."""
    s = "" if v is None else str(v)
    if s == "":
        return "?"
    if "\n" in s:
        return None  # caller must emit a ; text block
    if any(c in s for c in " \t") or s[0] in "_$[];'\"#":
        if '"' not in s:
            return f'"{s}"'
        if "'" not in s:
            return f"'{s}'"
        return None
    return s


def emit_keyvalue(out, category, pairs):
    width = max(len(k) for k, _ in pairs) + len(category) + 2
    for key, val in pairs:
        tag = f"{category}.{key}"
        rendered = cif_value(val)
        if rendered is None:
            out.append(tag)
            out.append(";" + str(val))
            out.append(";")
        else:
            out.append(f"{tag.ljust(width)} {rendered}")
    out.append("#")


def emit_loop(out, category, columns, rows):
    out.append("loop_")
    for col in columns:
        out.append(f"{category}.{col}")
    rendered = []
    for row in rows:
        cells = []
        for cell in row:
            r = cif_value(cell)
            # loop cells cannot be ; blocks here; collapse newlines defensively
            cells.append(r if r is not None else '"' + str(cell).replace("\n", " ") + '"')
        rendered.append(cells)
    if rendered:
        widths = [max(len(r[i]) for r in rendered) for i in range(len(columns))]
        for cells in rendered:
            out.append(" ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip())
    out.append("#")
