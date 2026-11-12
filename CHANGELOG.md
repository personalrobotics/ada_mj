# Changelog

## Unreleased

ada_mj moves onto ssrobot and ssgrasping.

- **Removed:** the mj_manipulator application, with its console and feeding demo.
  They remain at `v0.1.0`.
- **Added:** `ada-lift`, in which ADA grasps a 6 cm cylinder with the JACO2's
  enveloping side grasp (sstsr#204), lifts it in open space and holds it. It proceeds
  on the transitions the hand cannot observe by a recorded policy (ssgrasping#47).
- **Limitations:** held-object collision geometry is not conservative once the
  object is seated in the palm (ssrobot#242), and the model's fingers barely squeeze
  cylinders narrower than about 5 cm (ada_assets#50).
- **Project:** its own uv project and lock, with pinned dependencies.
- **CI:** lint, type checks, the end-to-end test, and a check that the installed wheel
  loads none of the legacy stack.

## v0.1.0

The last release on mj_manipulator.
