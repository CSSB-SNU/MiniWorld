"""Minimal mmCIF tokenizer, enough to pull single categories out of PDBx files."""


def tokenize(lines):
    """Yield (token, was_quoted) for mmCIF value tokens across `lines`."""
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith(";"):
            # multi-line semicolon text field
            buf = [line[1:]]
            i += 1
            while i < n and not lines[i].startswith(";"):
                buf.append(lines[i])
                i += 1
            i += 1
            # mmCIF semicolon text field: preserved verbatim, no stripping
            yield "\n".join(buf), True
            continue
        i += 1
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        j = 0
        m = len(stripped)
        while j < m:
            c = stripped[j]
            if c in " \t":
                j += 1
            elif c == "#":
                break
            elif c in "'\"":
                # quoted: terminator is quote followed by whitespace or EOL
                k = j + 1
                while k < m:
                    if stripped[k] == c and (k + 1 == m or stripped[k + 1] in " \t"):
                        break
                    k += 1
                yield stripped[j + 1 : k], True
                j = k + 1
            else:
                k = j
                while k < m and stripped[k] not in " \t":
                    k += 1
                yield stripped[j:k], False
                j = k


def parse_category(lines, category):
    """Extract one mmCIF category as (tags, rows).

    `category` is the dotted prefix without the trailing dot, e.g. "_chem_comp".
    Returns (tags, rows) where tags are the bare item names and rows is a list of
    equal-length value lists. Handles both `loop_` and key-value forms.
    Returns (None, None) if the category is absent, and raises Truncated if the
    block appears cut off mid-way.
    """
    prefix = category + "."
    start = None
    for idx, line in enumerate(lines):
        if line.startswith(prefix):
            start = idx
            break
    if start is None:
        return None, None

    # Is this a loop_? Walk back over blank/comment lines.
    k = start - 1
    while k >= 0 and not lines[k].strip():
        k -= 1
    is_loop = k >= 0 and lines[k].strip() == "loop_"

    if not is_loop:
        # key-value form: consecutive `_category.tag value` lines
        tags, vals = [], []
        idx = start
        while idx < len(lines):
            line = lines[idx]
            if not line.startswith(prefix):
                if line.strip():
                    break
                idx += 1
                continue
            toks = list(tokenize([line]))
            tags.append(toks[0][0][len(prefix) :])
            if len(toks) > 1:
                vals.append(toks[1][0])
                idx += 1
                continue
            # value lives on the following line(s), possibly a ; text block
            nxt = idx + 1
            while nxt < len(lines) and not lines[nxt].strip():
                nxt += 1
            if nxt >= len(lines):
                raise Truncated(category)
            if lines[nxt].startswith(";"):
                end = nxt + 1
                while end < len(lines) and not lines[end].startswith(";"):
                    end += 1
                if end >= len(lines):
                    raise Truncated(category)
                buf = [lines[nxt][1:], *lines[nxt + 1 : end]]
                vals.append("\n".join(buf))
                idx = end + 1
            else:
                got = list(tokenize([lines[nxt]]))
                if not got:
                    raise Malformed(f"{category}.{tags[-1]}: no value")
                vals.append(got[0][0])
                idx = nxt + 1
        return tags, [vals]

    # loop_ form: tag lines, then a flat token stream chunked by len(tags)
    tags = []
    idx = start
    while idx < len(lines) and lines[idx].startswith(prefix):
        tags.append(lines[idx].strip()[len(prefix) :])
        idx += 1
    if idx < len(lines) and lines[idx].strip().startswith("_"):
        # another category interleaved -> not a pure loop over this category
        return None, None

    data_lines = []
    terminated = False
    while idx < len(lines):
        s = lines[idx].strip()
        if s.startswith("#") or s == "loop_" or (s.startswith("_") and not lines[idx].startswith(";")):
            terminated = True
            break
        data_lines.append(lines[idx])
        idx += 1
    if not terminated:
        raise Truncated(category)

    toks = [t for t, _q in tokenize(data_lines)]
    ncol = len(tags)
    if ncol == 0 or len(toks) % ncol != 0:
        raise Malformed(f"{category}: {len(toks)} tokens for {ncol} columns")
    rows = [toks[i : i + ncol] for i in range(0, len(toks), ncol)]
    return tags, rows


class Truncated(Exception):
    pass


class Malformed(Exception):
    pass
