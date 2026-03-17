import re
from typing import Callable, List, Optional


def fairy_tale(
    text: str,
    min_reps: int = 10,
    min_block: int = 2,
    max_block: int = 300,
    *,
    normalize_case: bool = False,
    collapse_identical_lines: bool = True,
    # NEW: collapse "one-char repeats" (optionally with a small set of framing chars) if long enough
    long_run_min: int = 300,
    debug: bool = True,
) -> str:
    """
    Remove repeated subsequences from LLM outputs.

    Collapses:
      1) Very long one-character runs on a single line (e.g. "-----" or table-rule spam),
         if run length >= long_run_min. Keeps the line but trims the long run.
      2) Runs of identical lines (by normalized content) if run length >= min_reps.
      3) Consecutive repeated blocks (length L in [min_block..max_block]) if repeated >= min_reps.

    Notes:
      - Matching uses normalization (whitespace collapsed, line endings removed).
      - Output preserves original first occurrences (formatting kept).
      - This only collapses *consecutive* repeats (A A A ...).
    """
    # Add you regex here
    text = clean_text(text, n1=300, n2=40, n3=10)
    lines = text.splitlines(True)
    log = print if debug else (lambda *args, **kwargs: None)

    norm_fn = _make_normalizer(normalize_case=normalize_case)

    # NEW: sanitize extreme single-line char runs first (before block logic)
    if long_run_min and long_run_min > 0:
        lines = [
            _collapse_long_char_runs_in_line(ln, long_run_min=long_run_min, log=log)
            for ln in lines
        ]

    if collapse_identical_lines:
        lines = _collapse_equal_line_runs(lines, min_reps=min_reps, key=norm_fn, log=log)

    lines = _collapse_repeated_blocks(lines, min_block, max_block, min_reps=min_reps, key=norm_fn, log=log)

    if collapse_identical_lines:
        lines = _collapse_equal_line_runs(lines, min_reps=min_reps, key=norm_fn, log=log)  # sweep again

    return "".join(lines)


def _make_normalizer(*, normalize_case: bool) -> Callable[[str], str]:
    def norm(s: str) -> str:
        s = s.rstrip("\r\n")
        s = re.sub(r"\s+", " ", s).strip()
        if normalize_case:
            s = s.lower()
        return s
    return norm


_LONG_RUN_CHARS = r"\-=_~.\*#\+:"  # '-' '=' '_' etc. (NOT including '|')


def _preview_text(s: str, limit: int = 120) -> str:
    compact = s.replace("\r", "\\r").replace("\n", "\\n")
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _collapse_long_char_runs_in_line(
    line: str,
    *,
    long_run_min: int,
    keep: int = 30,
    log: Callable[..., None] = print,
) -> str:
    """
    Shrink any run of the same character (from _LONG_RUN_CHARS) repeated
    >= long_run_min times down to `keep` occurrences.

    Keeps line endings unchanged.
    """
    # Preserve newline exactly
    m_nl = re.search(r"(\r\n|\n|\r)$", line)
    nl = m_nl.group(1) if m_nl else ""
    core = line[:-len(nl)] if nl else line

    if long_run_min <= keep:
        return line  # no point

    # Compile threshold-specific regex: (c) repeated >= long_run_min
    rx = re.compile(rf"([{_LONG_RUN_CHARS}])\1{{{long_run_min-1},}}")

    def repl(m: re.Match) -> str:
        ch = m.group(1)
        original_len = len(m.group(0))
        log(
            f"[DEDUPE] Collapsed long char run: char={ch!r}, from={original_len} to={keep}, "
            f"line='{_preview_text(core)}'"
        )
        return ch * keep

    core2 = rx.sub(repl, core)
    return core2 + nl


def _collapse_equal_line_runs(
    lines: List[str],
    min_reps: int,
    key: Callable[[str], str],
    *,
    log: Callable[..., None] = print,
) -> List[str]:
    out: List[str] = []
    run: List[str] = []
    run_key: Optional[str] = None

    def flush() -> None:
        nonlocal run, run_key
        if not run:
            return
        if run_key is not None and len(run) >= min_reps:
            log(
                f"[DEDUPE] Collapsed identical line run: reps={len(run)}, "
                f"line='{_preview_text(run[0])}'"
            )
            out.append(run[0])
        else:
            out.extend(run)
        run = []
        run_key = None

    for ln in lines:
        k = key(ln)
        if run_key is None:
            run_key = k
            run = [ln]
        elif k == run_key:
            run.append(ln)
        else:
            flush()
            run_key = k
            run = [ln]

    flush()
    return out


def _collapse_repeated_blocks(
    lines: List[str],
    min_block: int,
    max_block: int,
    *,
    min_reps: int,
    key: Callable[[str], str],
    log: Callable[..., None] = print,
) -> List[str]:
    out: List[str] = []
    n = len(lines)
    i = 0

    keys = [key(ln) for ln in lines]

    if min_block < 1:
        min_block = 1
    if max_block < min_block:
        max_block = min_block

    while i < n:
        found = False
        max_L = min(max_block, (n - i) // 2)

        for L in range(max_L, min_block - 1, -1):
            block_keys = keys[i:i + L]
            reps = 1

            while i + (reps + 1) * L <= n and keys[i + reps * L:i + (reps + 1) * L] == block_keys:
                reps += 1

            if reps >= min_reps:
                block_preview = _preview_text("".join(lines[i:i + min(L, 2)]))
                log(
                    f"[DEDUPE] Collapsed repeated block: block_len={L}, reps={reps}, "
                    f"preview='{block_preview}'"
                )
                out.extend(lines[i:i + L])
                i += reps * L
                found = True
                break

        if not found:
            out.append(lines[i])
            i += 1

    return out


def cap_repeat_1char(s: str, n1: int, keep=1, *, debug: bool = True) -> str:
    if n1 <= 0:
        return re.sub(r"(.)\1*", "", s)

    rx = re.compile(rf"(.)\1{{{n1},}}")

    def repl(m: re.Match) -> str:
        ch = m.group(1)
        full = m.group(0)
        if debug:
            print(
                f"[SUBSEQ] 1-char run removed: char={ch!r}, "
                f"len={len(full)}, kept={keep}"
            )
        return ch * keep

    return rx.sub(repl, s)


def cap_repeat_2chars(s: str, n2: int, keep=1, *, debug: bool = True) -> str:
    if n2 <= 0:
        return re.sub(r"(..)\1*", "", s)

    rx = re.compile(rf"(..)(?:\1){{{n2},}}")

    def repl(m: re.Match) -> str:
        unit = m.group(1)
        full = m.group(0)
        reps = len(full) // len(unit)
        if debug:
            print(
                f"[SUBSEQ] 2-char repeat removed: unit={unit!r}, "
                f"reps={reps}, kept={keep}"
            )
        return unit * keep

    return rx.sub(repl, s)


def cap_repeat_3plus(s: str, n3: int, keep=1, *, debug: bool = True) -> str:
    if n3 <= 0:
        return re.sub(r"(.{3,}?)(?:\1)*", "", s)

    rx = re.compile(rf"(.{{3,}}?)(?:\1){{{n3},}}")

    def repl(m: re.Match) -> str:
        unit = m.group(1)
        full = m.group(0)
        reps = len(full) // len(unit)
        if debug:
            print(
                f"[SUBSEQ] 3+ repeat removed: unit='{unit[:40]}...', "
                f"unit_len={len(unit)}, reps={reps}, kept={keep}"
            )
        return unit * keep

    return rx.sub(repl, s)


def clean_text(s: str, n1: int, n2: int, n3: int, *, debug: bool = True) -> str:
    s = cap_repeat_1char(s, n1, debug=debug)
    s = cap_repeat_2chars(s, n2, debug=debug)
    s = cap_repeat_3plus(s, n3, debug=debug)
    return s

