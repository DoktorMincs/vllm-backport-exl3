#!/usr/bin/env python3
"""Soma tempo de kernel por nome dentro de janelas de tempo do trace nsys.

Uso:
  prof_window.py --db trace.sqlite --schema
  prof_window.py --db trace.sqlite --window decode=1793110890.5:1793110900.2 \
                 --window prefill=1793110900.2:1793110908.6 [--top 25]

As janelas sao epoch (segundos); o script converte para o relogio do trace.
"""

import os as _os

EXL3_ROOT = _os.environ.get("EXL3_PORT", _os.path.expanduser("~/exl3-port"))
MODEL_DIR = _os.environ.get(
    "MODEL_DIR",
    _os.path.expanduser("~/models/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw"),
)
MM3_REF = _os.environ.get("MM3_REF", _os.path.expanduser("~/mm3-oracle/ref.py"))
VLLM_REPO = _os.environ.get("VLLM_REPO", _os.path.expanduser("~/vllm-backport"))

import argparse
import sqlite3
import sys


def table_cols(cur, table):
    return [r[1] for r in cur.execute("PRAGMA table_info('%s')" % table)]


def find_table(cur, needle):
    for (name,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        if needle.upper() in name.upper():
            return name
    return None


def all_tables(cur):
    return [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def schema(cur):
    for t in all_tables(cur):
        if any(k in t.upper() for k in ("KERNEL", "SESSION", "STRING", "NAME", "PROCESS")):
            print(t, "|", ",".join(table_cols(cur, t)))


def session_start(cur):
    for cand in ("TARGET_INFO_SESSION_START_TIME", "SESSION_START_TIME"):
        if cand in all_tables(cur):
            cols = table_cols(cur, cand)
            row = cur.execute("SELECT * FROM %s LIMIT 1" % cand).fetchone()
            print("# session table %s %s -> %s" % (cand, cols, row), file=sys.stderr)
            if row:
                for name, val in zip(cols, row):
                    if isinstance(val, int) and val > 1e15:  # ns epoch ou ns desde boot
                        if val > 1e18:      # ns desde epoch
                            return val / 1e9
                        return None
                    if isinstance(val, (int, float)) and val > 1e9:
                        return float(val)
    return None


def name_map(cur):
    """id -> texto, seja StringIds ou tabela de nomes desmangados."""
    if "StringIds" in all_tables(cur):
        return ("StringIds", "value", {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")})
    return (None, None, {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--window", action="append", default=[])
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--start-epoch", type=float, default=None,
                    help="epoch da primeira amostra do trace (se nao der para ler)")
    args = ap.parse_args()

    db = sqlite3.connect(args.db)
    cur = db.cursor()
    if args.schema:
        schema(cur)
        return 0

    kt = find_table(cur, "KERNEL")
    cols = table_cols(cur, kt)
    print("# kernel table %s: %s" % (kt, cols), file=sys.stderr)
    name_col = "demangledName" if "demangledName" in cols else (
        "nameId" if "nameId" in cols else ("shortName" if "shortName" in cols else None))
    dev_col = "deviceId" if "deviceId" in cols else None
    sid, _sval, smap = name_map(cur)
    print("# nomes via %s" % sid, file=sys.stderr)

    t0 = session_start(cur)
    if t0 is None:
        t0 = args.start_epoch
    if t0 is None:
        print("ERRO: nao consegui achar o inicio do trace; passe --start-epoch", file=sys.stderr)
        return 2
    print("# session start epoch=%.3f" % t0, file=sys.stderr)

    for spec in args.window:
        label, rng = spec.split("=", 1)
        a, b = [float(x) for x in rng.split(":")]
        lo, hi = (a - t0) * 1e9, (b - t0) * 1e9
        q = ("SELECT %s, %s, SUM(end-start) AS t, COUNT(*) AS n "
             "FROM %s WHERE start >= ? AND end <= ? GROUP BY 1,2 ORDER BY t DESC"
             % (name_col, dev_col, kt))
        rows = cur.execute(q, (lo, hi)).fetchall()
        total = sum(r[2] for r in rows)
        print("\n=== %s  [%.2f s de janela]  tempo de kernel somado=%.2f s" %
              (label, b - a, total / 1e9))
        per_dev = {}
        agg = {}
        for name, dev, t, n in rows:
            disp = smap.get(name, str(name)) if isinstance(name, int) else (name or "?")
            agg[disp] = agg.get(disp, 0.0) + t
            per_dev[dev] = per_dev.get(dev, 0.0) + t
        print("-- por GPU (s):", ", ".join("gpu%s=%.1f" % (d, v / 1e9)
                                          for d, v in sorted(per_dev.items(),
                                                             key=lambda x: -(x[1] or 0))[:12]))
        print("-- top %d kernels:" % args.top)
        for disp, t in sorted(agg.items(), key=lambda x: -x[1])[:args.top]:
            print("  %7.3f s  %5.1f%%  %s" % (t / 1e9, 100.0 * t / total if total else 0, disp[:110]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
