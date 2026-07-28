"""Rich terminal monitor for connector training runs.

  python scripts/connector/monitor.py                # one snapshot (default logs)
  python scripts/connector/monitor.py --watch 30     # refresh every 30s
  python scripts/connector/monitor.py --logs /tmp/set_mac.log /tmp/seg_mac.log

Shows per log: run state (process, elapsed, ETA), intra-epoch batch progress (if the
trainer prints it), an epoch table, ASCII curves for loss / IoU / dirty / cleanOK / corR,
and reference lines (floor, v1, v3 baselines).
"""
import argparse
import os
import re
import subprocess
import time

REF = {"floor": 0.213, "v1": 0.304, "v3-BIO": 0.320}
EP = re.compile(r"\[ep (\d+)\] loss=([\d.]+) dev: iou=([\d.]+) \(fl=([\d.]+), tau=([\d.]+), "
                r"g=([\d.]+)\) dirty=([\d.]+) cleanOK=([\d.]+) gateRec=([\d.]+) "
                r"corR=([-\d.]+) corS=([-\d.]+).*\[([\d.]+)m\]")
BATCH = re.compile(r"\[ep (\d+) b(\d+)/(\d+)\] loss=([\d.]+)")


def spark(vals, width=48, lo=None, hi=None):
    if not vals:
        return "(no data)"
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    rng = (hi - lo) or 1e-9
    chars = "▁▂▃▄▅▆▇█"
    pts = vals[-width:]
    return "".join(chars[min(7, int((v - lo) / rng * 7.999))] for v in pts)


def curve(name, vals, fmt="{:.3f}", ref=None):
    if not vals:
        return f"  {name:8} (пока нет данных)"
    cur, best = vals[-1], (max(vals) if name != "loss" else min(vals))
    extra = ""
    if ref:
        marks = " ".join(f"{k}={v}" for k, v in ref.items())
        extra = f"   [{marks}]"
    return (f"  {name:8} {spark(vals)}  now={fmt.format(cur)} "
            f"best={fmt.format(best)}{extra}")


def proc_info(pattern):
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        pids = [p for p in out.stdout.split() if p.strip()]
        if not pids:
            return None
        ps = subprocess.run(["ps", "-o", "etime=,pcpu=,rss=", "-p", pids[0]],
                            capture_output=True, text=True).stdout.split()
        if len(ps) >= 3:
            return dict(pid=pids[0], etime=ps[0], cpu=ps[1] + "%",
                        ram=f"{int(ps[2]) // 1024}MB")
    except Exception:
        pass
    return None


def show_log(path):
    name = os.path.basename(path)
    if not os.path.exists(path):
        print(f"── {name}: (лог ещё не создан)")
        return
    text = open(path, errors="ignore").read()
    done = "DONE" in text
    eps = [EP.search(l) for l in text.splitlines()]
    eps = [m for m in eps if m]
    rows = [dict(ep=int(m.group(1)), loss=float(m.group(2)), iou=float(m.group(3)),
                 dirty=float(m.group(7)), cleanOK=float(m.group(8)),
                 corR=float(m.group(10)), mins=float(m.group(12))) for m in eps]
    state = "✅ DONE" if done else "🟢 running"
    print(f"── {name}  {state}  эпох: {len(rows)}/12")
    # intra-epoch batch progress (only future runs print it)
    bts = [BATCH.search(l) for l in text.splitlines()[::-1][:50]]
    bts = [m for m in bts if m]
    if bts and not done and (not rows or int(bts[0].group(1)) > rows[-1]["ep"]):
        m = bts[0]
        frac = int(m.group(2)) / max(1, int(m.group(3)))
        bar = "█" * int(frac * 30) + "░" * (30 - int(frac * 30))
        print(f"  внутри эпохи {m.group(1)}: [{bar}] {frac:.0%}  running-loss={m.group(4)}")
    elif rows and not done:
        # estimate position inside the current epoch from wall-clock
        per_ep = (rows[-1]["mins"] / rows[-1]["ep"]) if rows[-1]["ep"] else 8
        mtime_min = (time.time() - os.path.getmtime(path)) / 60
        frac = min(0.99, mtime_min / per_ep)
        bar = "█" * int(frac * 30) + "░" * (30 - int(frac * 30))
        print(f"  эпоха {rows[-1]['ep'] + 1} (оценка по времени): [{bar}] ~{frac:.0%} "
              f"(~{per_ep:.1f} мин/эпоха)")
    if rows:
        print(f"  {'ep':>3} {'loss':>8} {'iou':>7} {'dirty':>7} {'cleanOK':>8} {'corR':>7} {'мин':>6}")
        for r in rows[-6:]:
            flag = " ←" if r["iou"] == max(x["iou"] for x in rows) else ""
            print(f"  {r['ep']:>3} {r['loss']:>8.4f} {r['iou']:>7.4f} {r['dirty']:>7.3f} "
                  f"{r['cleanOK']:>8.2f} {r['corR']:>7.3f} {r['mins']:>6.1f}{flag}")
        print(curve("loss", [r["loss"] for r in rows]))
        print(curve("iou", [r["iou"] for r in rows], ref=REF))
        print(curve("dirty", [r["dirty"] for r in rows]))
        print(curve("cleanOK", [r["cleanOK"] for r in rows], "{:.2f}"))
        print(curve("corR", [r["corR"] for r in rows]))
        if not done:
            per_ep = rows[-1]["mins"] / max(1, rows[-1]["ep"])
            left = (12 - rows[-1]["ep"]) * per_ep
            print(f"  ETA до конца прогона: ~{left:.0f} мин")
    else:
        tail = [l for l in text.splitlines() if l.strip()][-2:]
        for l in tail:
            print(f"  {l[:100]}")
        print("  (эпоха 1 ещё считается — таблица появится после первой eval)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*",
                    default=["/tmp/set_mac.log", "/tmp/seg_mac.log", "/tmp/v3_mac.log"])
    ap.add_argument("--watch", type=int, default=0, help="refresh period, seconds")
    args = ap.parse_args()
    while True:
        if args.watch:
            os.system("clear")
        print(f"╔══ CONNECTOR TRAINING MONITOR ── {time.strftime('%H:%M:%S')} ══╗")
        pi = proc_info("train_connector")
        print(f"  процесс: " + (f"pid {pi['pid']}  {pi['etime']}  CPU {pi['cpu']}  RAM {pi['ram']}"
                                if pi else "не запущен"))
        print(f"  референсы: floor={REF['floor']}  v1={REF['v1']}  v3-BIO={REF['v3-BIO']}\n")
        for lg in args.logs:
            show_log(lg)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
