import re

dash_payload = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+|[A-Za-zА-Яа-яЁё])\.\s*-\s*(.+?)\s*$"
)

def fairy_tame(text, min_reps=10):
    lines = text.splitlines(True)

    lines = _collapse_same_payload_runs(lines, min_reps=min_reps)
    lines = _collapse_repeated_blocks(lines, 2, 300, min_reps=min_reps)
    lines = _collapse_same_payload_runs(lines, min_reps=min_reps)  # sweep again

    return "".join(lines)

def _payload(line):
    m = dash_payload.match(line.rstrip("\r\n"))
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(1)).strip().lower()

def _collapse_same_payload_runs(lines, min_reps=10):
    """
    Схлопывает подряд идущие строки с одинаковым payload
    ТОЛЬКО если подряд их >= min_reps.
    Если серия короче — оставляет все строки серии.
    """
    out = []
    run = []          # накопленные строки серии
    run_payload = None

    def flush_run():
        nonlocal run, run_payload
        if not run:
            return
        if run_payload is not None and len(run) >= min_reps:
            out.append(run[0])   # оставить только первое
        else:
            out.extend(run)      # оставить всю серию
        run = []
        run_payload = None

    for ln in lines:
        p = _payload(ln)
        if p is None:
            flush_run()
            out.append(ln)
            continue

        if run_payload is None:
            run_payload = p
            run = [ln]
        elif p == run_payload:
            run.append(ln)
        else:
            flush_run()
            run_payload = p
            run = [ln]

    flush_run()
    return out

def _collapse_repeated_blocks(lines, min_block, max_block, min_reps=10):
    """
    Схлопывает подряд повторяющиеся блоки строк длины L (min_block..max_block)
    ТОЛЬКО если блок повторился подряд >= min_reps раз.
    Оставляет только первое появление блока.
    """
    out = []
    i = 0
    n = len(lines)

    while i < n:
        found = False
        max_L = min(max_block, (n - i) // 2)

        for L in range(max_L, min_block - 1, -1):
            block = lines[i:i+L]
            reps = 1
            while i + (reps + 1) * L <= n and lines[i + reps*L : i + (reps+1)*L] == block:
                reps += 1

            if reps >= min_reps:
                out.extend(block)   # оставить только первый блок
                i += reps * L       # пропустить остальные
                found = True
                break

        if not found:
            out.append(lines[i])
            i += 1

    return out
