"""The declared fresh-seed evidence run for ada_mj's lift: seeds 1000-1029, both grips, once,
from a clean checkout. Each run is the installed command, ``ada-lift --seed S --grip G --record``.
Writes <out>/<grip>/seed-<S>/lift.json, <out>/run.log, and <out>/summary.json (atomically)."""
import json, os, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

repo, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
SEEDS, GRIPS = list(range(1000, 1030)), ("enveloping", "fingertip")
out.mkdir(parents=True, exist_ok=False)
log = open(out / "run.log", "w", buffering=1)
def say(msg): log.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}\n")
git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
py = str(repo / ".venv/bin/python")
versions = subprocess.run([py, "-c", "import importlib.metadata as m, json; print(json.dumps({p: m.version(p) for p in ('ada-mj','ssrobot','ssgrasping','sstsr','sscbirrt','mujoco','ada-assets','numpy')}))"], capture_output=True, text=True).stdout.strip()
say(f"command {' '.join(sys.argv)}")
say(f"commit {git('rev-parse', 'HEAD')} clean={git('status', '--porcelain') == ''}")
say(f"host {platform.node()} {platform.platform()} load {os.getloadavg()}")
say(f"versions {versions}")
say(f"declared: seeds {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)}), grips {GRIPS}: {len(SEEDS) * len(GRIPS)} runs")
start, rows = time.time(), []
for grip in GRIPS:
    for seed in SEEDS:
        d = out / grip / f"seed-{seed}"
        r = subprocess.run([str(repo / ".venv/bin/ada-lift"), "--seed", str(seed), "--grip", grip, "--record", str(d)], capture_output=True, text=True, cwd=out)
        rec = json.loads((d / "lift.json").read_text()) if (d / "lift.json").exists() else None
        p = rec["privileged"] if rec else {}
        failed = None if rec is None else next((s["name"] + ":" + str(s["reason"]) for s in rec["grasp"]["stages"] if not s["succeeded"]), None)
        rows.append({"grip": grip, "seed": seed, "exit": r.returncode, "grasp": rec and rec["grasp"]["state"], "failed_stage": failed,
                     "held": p.get("held"), "rise_m": p.get("rise_m"), "max_tilt_rad": p.get("max_tilt_rad")})
        say(f"{len(rows)}/{len(SEEDS) * len(GRIPS)} {grip} seed {seed}: exit {r.returncode} grasp {rows[-1]['grasp']} held {rows[-1]['held']} rise {p.get('rise_m')} tilt {p.get('max_tilt_rad')} failed {failed} elapsed {time.time() - start:.0f}s")
assert len(rows) == 60
summary = {"commit": git("rev-parse", "HEAD"), "seeds": [SEEDS[0], SEEDS[-1]], "runs": rows, "by_grip": {}}
for grip in GRIPS:
    g = [r for r in rows if r["grip"] == grip]
    held = [r for r in g if r["held"]]
    summary["by_grip"][grip] = {"runs": len(g), "grasp_succeeded": sum(r["grasp"] == "succeeded" for r in g), "held": len(held),
        "rise_m_held_min_max": [min(r["rise_m"] for r in held), max(r["rise_m"] for r in held)] if held else None,
        "max_tilt_rad_held_max": max(r["max_tilt_rad"] for r in held) if held else None,
        "failed_stages": sorted({r["failed_stage"] for r in g if r["failed_stage"]})}
tmp = out / "summary.json.partial"; tmp.write_text(json.dumps(summary, indent=1) + "\n"); tmp.replace(out / "summary.json")
say(f"by_grip {json.dumps(summary['by_grip'])}")
say("EXIT=0")
