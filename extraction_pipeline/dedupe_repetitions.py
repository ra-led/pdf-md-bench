import re
from typing import Callable, List, Optional


def fairy_tame(
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


# NEW:
# Reduce pathological "one-character repeats" that exceed a threshold.
# Example: "|-------|-----....(1000 dashes).....-----|"
#
# Strategy:
# - Detect any run of a single repeated char among a conservative set
#   that commonly appears in LLM artifacts / separators.
# - If run length >= long_run_min, shrink it to a shorter run (keeps structure).
#
# This avoids deleting legitimate normal MD table separator lines like:
# "| --- | --- |" because those runs are short.
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
