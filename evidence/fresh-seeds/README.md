# Fresh-seed lift evidence

This is the declared evidence for ADA's grasp-and-lift slice (#57). It is a single run on
seeds never used before, from a clean checkout of the merged commit.

## The declaration
- **Seeds:** 1000–1029 (30).
- **Grips:** `enveloping` (what `ada-lift` uses) and `fingertip` (for comparison): 60 runs.
- **Each run:** the installed command, `ada-lift --seed S --grip G --record <dir>`.
- **Held:** the grasp succeeded, and at the end of the 10 s hold the cylinder has risen
  at least 5 cm (`ada_mj.lift`). There is no pass threshold; the counts are reported as found.

## The run
- **Code:** ada_mj `9eaa328dcda43dac67abe54fbb49728375c4125c`, with a clean tree, synced
  with `uv sync --locked`.
- **Key versions:** sstsr 4.5.0, sscbirrt 3.16.0, MuJoCo 3.14.0, numpy 2.5.4; the rest
  are pinned by the lock.
- **Host:** macOS 26.6.2 on arm64, 2026-11-12. It took 136 s.
- **Runner:** [`evidence_run.py`](evidence_run.py), kept as it ran (sha256
  `e77a3c5c8816f5413cc6b789b21a75525933eb44f76f1e273d1c240317c4c310`).
- **Outputs:**
  - [`run.log`](run.log): the command, commit, versions and one line per run;
  - [`summary.json`](summary.json): every run and the counts per grip (sha256
    `a8f4e4497aa9db623c1979cd32d5e4d02523726974e4b64412a1e84f0cf7da17`);
  - `<grip>/seed-<S>/lift.json`: each run's full report.

The runner doesn't stop on a failed run. It writes `EXIT=0` once the 60 runs finish,
whatever their exits, and it doesn't check that the version probe succeeded. For this
run, all three were checked: all 60 runs exited 0, each wrote its `lift.json`, and
`run.log` records the versions.

## Results

| Grip | Grasp succeeded | Held | Height when held | Largest tilt over the hold (median / max) |
|---|---|---|---|---|
| enveloping | 30/30 | 30/30 | 59.1–61.5 mm | 5.4° / 6.7° |
| fingertip | 30/30 | 24/30 | 55.4–62.7 mm | 11.2° / 14.7° |

Six fingertip runs reported success but did not lift the cylinder:
- **Seeds 1000, 1005, 1009, 1013 and 1029:** it was left on the table, risen less than 3 mm.
- **Seed 1011:** it was flung off the table, which has no floor below it, so its recorded
  rise is −388 m.

The enveloping grasp held every seed. The fingertip grasp's misses and larger tilt are
what sstsr#204's enveloping family was added to avoid.

## Reproduce

```bash
git clone https://github.com/personalrobotics/ada_mj && cd ada_mj
git checkout 9eaa328dcda43dac67abe54fbb49728375c4125c && uv sync --locked
python3 <this directory>/evidence_run.py "$PWD" runs/<new directory>
```

The runner refuses an output directory that already exists. Outcomes are expected to
match on the same platform. MuJoCo's results can differ in the last bits across
platforms, so on another platform compare the counts, not the bytes.
