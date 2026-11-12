# ada_mj

ADA is the Personal Robotics Lab's assistive feeding robot: a Kinova JACO2 arm with its
two-finger hand. ada_mj is ADA's application on
[ssrobot](https://github.com/personalrobotics/ssrobot) and
[ssgrasping](https://github.com/personalrobotics/ssgrasping), in MuJoCo.

It does one thing so far. ADA grasps a cylinder standing on a table and lifts it, in open space.

The earlier application, built on mj_manipulator, remains available at
[`v0.1.0`](https://github.com/personalrobotics/ada_mj/releases/tag/v0.1.0). That version
includes the console and the feeding demo, which this version does not have yet.

## What it shows

`ada-lift` stands a 6 cm, 0.1 kg cylinder on the over-bed table and registers its pose
from the simulator, in place of perception. It then:
1. grasps the cylinder with ssgrasping's `grasp`, from sstsr's enveloping side grasps
   for the JACO2 (sstsr#204), in which the palm comes close and the fingers fold around
   the cylinder;
2. lifts it 6 cm;
3. holds still for 10 s.

The JACO2's opening does not tell whether it has the cylinder (ssgrasping#47). So the
grasp's `acquired` and `retained` are unobserved, and ADA proceeds on them by a declared
policy (`assume_unobserved`), recorded in every run.

Whether the cylinder is held is scored from the simulator's state, for evaluation only.
It is held when, at the end of the hold, it is still attached and has risen at least
5 cm. The run also records the cylinder's largest tilt over the hold, and when ssrobot
first found it moved in the hand.

`--grip fingertip` uses sstsr's other side grasps instead, which pinch the cylinder at
the fingertips. They are there for comparison: the JACO2's curling fingers pull the
cylinder in and turn it as they close, and it often slips.

## Limitations

- **Held-object collision geometry is not conservative after the JACO2 seats the object
  in its palm** (ssrobot#242). So ADA demonstrates grasping and lifting in open space
  only, not carrying or placing an object near obstacles.
- **The model's fingers barely squeeze a cylinder narrower than about 5 cm**
  (ada_assets#50). They reach their closed target wrapped around it, so it is held
  lightly, tilts in the hand and creeps down. That is why the cylinder is 6 cm.

## Use

```bash
uv sync --locked
uv run ada-lift --seed 0 --record runs/seed0
```

- `--seed` places the cylinder; seed 0 is the reference pose.
- `--grip` chooses `enveloping` (the default) or `fingertip`.
- `--record` writes the run's report, `lift.json`.

The command exits 0 when the grasp succeeds.

ada_mj is its own uv project (`[tool.uv.workspace] members = []`), so in a robot-code
checkout, sync it from its own directory.

## Develop

```bash
uv run pytest            # the end-to-end lift; its report goes to artifacts/
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run python scripts/check_boundary.py   # none of the legacy stack loads
```
