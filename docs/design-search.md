# 2D gripper design search

`graspsim2d` searches a bounding box of design variables for the finger shape of
a 1-DoF, two-finger angular gripper that can pick up filled cereal boxes and
sheets of cardboard. It generates several distinct finger concepts, screens
thousands of variants of each with a fast analytic grasp model, checks the
leaders in a planar MuJoCo simulation (close, lift, hold, shake, hold), feeds
what the simulation learned back into the screen, and reports the best
all-rounder and the best design for each object.

The bounding box is centred on Parth's finger
(`UMI6_paddles_paralel_pockets_V01.stp`): two pivots 38 mm apart, geared 1:1
(equal and opposite rotation), the tip 150 mm from the pivot line, and a flat
face that is parallel to the other finger's when closed, made of a wide paddle
and a narrow tab. That design is concept C1, and it is always simulated as
drawn for reference. Every candidate must also clear the FT Dynamic bracket
(top) above the jaw: each finger is cut to clear it, as Parth's is.

Everything is 2D: the plane of the jaw (finger opening direction x, lift
direction z). Each pad is extruded its own width out of the plane, and the
contact width is the pad's or the object's, whichever is narrower.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # pinned versions the first pass used
pip install -e .                     # optional: adds the `graspsim2d` command
```

Without `pip install -e .`, run from the repo root as
`PYTHONPATH=. python -m graspsim2d ...`.

## Commands

```bash
python -m graspsim2d describe                      # design space, concepts, objects
python -m graspsim2d screen baseline_paddle        # analytic screen of a concept's nominal design (C1: Parth's finger)
python -m graspsim2d simulate self_aligning --objects hbo_28oz,b_flute_3mm --plot traces.png
python -m graspsim2d search --out runs/first_pass  # the full ideate / simulate / iterate loop
python -m graspsim2d search --quick --out runs/smoke   # tiny budget, one round, about 2 minutes
python -m graspsim2d search --quick --rounds 2 --out runs/smoke2   # tiny budget, two rounds
python -m graspsim2d figures runs/first_pass       # redraw the figures of a finished run
```

`screen` and `simulate` take a concept name (its nominal design), a design
from a finished run as `runs/first_pass/results.json:r2-03`, or a JSON file
with design variables. `simulate --xml DIR` writes the MJCF models so a run can
be opened in the MuJoCo viewer (`python -m mujoco.viewer --mjcf=DIR/hbo_28oz.xml`).

`--config DIR` points every command at another config directory;
`--design-space FILE`, `--objects FILE`, `--scenario FILE`, `--concepts FILE`
and `--search FILE` override single files.

The full search with the shipped budget (2 rounds) takes roughly 15 to 30
minutes on 8 cores. It uses every core unless `workers` in `search.yaml` says
otherwise.

## Configuration

All inputs are YAML files in `config/`. A run copies them into its output
directory, so every result keeps the inputs that produced it.

| file | what it holds |
|---|---|
| `design_space.yaml` | the bounding box: a range (or fixed value) for each design variable |
| `concepts.yaml` | the finger concepts, which variables each one searches, any narrower ranges, starting values, and which concept is the reference |
| `objects.yaml` | materials and objects, with sources for every value |
| `scenario.yaml` | presentation, the bracket keep-out, motion sequence, shake, pass criteria and simulation settings |
| `search.yaml` | search budget |

### Swapping in a new bounding box

Edit `config/design_space.yaml` (or pass `--design-space my_box.yaml`) and
rerun `search`. Each entry is

```yaml
finger_reach: {range: [6, 7.5], unit: in}   # converted to the schema unit (mm)
pad_angle:    {value: 4, unit: deg}         # fixed value
```

Accepted units: mm, cm, m, in (lengths); deg, rad (angles); N*m (also N.m,
Nm) and lbf*in (torque); N*m/rad; shoreA; 1/m (curvature); 1 (fractions).
A variable left out of the file is fixed at the schema nominal from
`graspsim2d/schema.py`, and an unknown name or unit is an error, so a typo
cannot silently drop a variable. Concept narrowing in `concepts.yaml` is
intersected with the bounding box, so it never widens a range you set (if the
two do not overlap, the concept uses your range). `describe` prints the box
that will actually be searched.

The variables:

| variable | unit | meaning |
|---|---|---|
| `pivot_spacing` | mm | distance between the two finger pivot axes (fixed at 38: the bracket carries both pivots) |
| `finger_reach` | mm | fingertip distance forward of the pivot line, fingers parallel |
| `finger_thickness` | mm | in-plane depth of the finger beam |
| `crank_offset` | mm | inward offset of the distal finger section (0 = straight) |
| `crank_position` | 1 | where the crank starts, as a fraction of the reach |
| `pad_length` | mm | length of the (distal) pad face; the paddle, for the paddle style |
| `pad_thickness` | mm | elastomer thickness (at the crown apex for crowned pads) |
| `pad_angle` | deg | pad face angle to the finger axis; + puts the distal end further inward |
| `pad_tip_margin` | mm | fingertip to the distal end of the pad; for the paddle style, the tab length plus a 2 mm gap |
| `pad_width` | mm | out-of-plane width of the pad (the paddle, for the paddle style) |
| `tab_width` | mm | paddle style: out-of-plane width of the tab carrying the face on to the tip |
| `pad_shore` | Shore A | pad hardness, converted to a modulus (Gent 1958) |
| `pad_crown_curvature` | 1/m | pad face curvature (0 = flat) |
| `finger_torque` | N*m | drive torque at each finger pivot (the 1:1 gears give both fingers the same), one value for every object |
| `max_opening` | deg | opening angle of each finger at the start of a grasp |
| `pad2_offset`, `pad2_length`, `pad2_angle`, `pad2_protrusion` | mm, mm, deg, mm | dual-zone concept: the proximal box pad |
| `pivot_depth`, `pivot_stiffness`, `pivot_range` | mm, N*m/rad, deg | self-aligning concept: the pad's pivot pin and centring spring |

To add a variable, add a `VariableDef` to `SCHEMA` in `graspsim2d/schema.py`
and use it in `graspsim2d/geometry.py`.

### Concepts

| concept | idea |
|---|---|
| C1 `baseline_paddle` | Parth's finger: straight, a wide paddle (49.9 to 98.8 mm from the pivot line, 118 mm wide) and a 21 mm tab carrying the same face to the tip, parallel to the other finger's face when closed (2.4 mm gap) |
| C2 `cranked_flat` | the distal section is offset inward, so the fingers can be parallel on a thin sheet while the proximal part clears a box |
| C3 `crowned` | convex pad face; the contact point rolls along the pad as the jaw angle changes |
| C4 `dual_zone` | a tip pad for sheets plus a proximal box pad; the closing angle picks the zone |
| C5 `self_aligning` | the pad turns on a passive pin with a centring spring and stops, so it lies flat on any face |

A new concept is an entry in `concepts.yaml` (which variables it searches and
which pad style it uses); a new pad style needs code in `geometry.py` and,
for the dynamic check, `mjsim.py`. `nominal:` gives a concept its own starting
values where the baseline's do not suit it (C2 to C5 start with a 40 mm pad
5 mm from the tip). `reference: true` (set on C1) puts the concept's nominal
design, Parth's finger as drawn, through the dynamic check on every object in
round 1 and reports it as `baseline`.

### Objects

Cereal boxes are presented upright on a table and gripped across their depth
(the narrow dimension) from above. Sheets hang from a holder that lets go once
the jaw has closed, and are pinched near their free top edge. Every value in
`objects.yaml` carries its source and spread in a comment. An object with
`enabled: false` stays in the file but is skipped.

### Scenario and pass criteria

`scenario.yaml` sets the motion: close for 0.9 s at up to 1 rad/s, settle,
lift 80 mm, hold, shake for 3 s (0.5 g lateral at 2 Hz plus 0.5 g vertical at
3 Hz, with 0.5 s ramps at each end), and hold again. Keep the ramps at least one
lateral period long: shorter ramps overshoot the specified acceleration
(0.25 s gives 0.68 g for 0.5 g) and kick the sheets' bending modes. A grasp
passes when every margin is at least 1:

| criterion | analytic screen | dynamic check |
|---|---|---|
| slip | friction hold capacity against weight plus vertical shake, with `analytic.friction_safety_factor` | grip point moves less than `max_slip_mm` (2 mm) relative to the jaw |
| rotation | pitch moment capacity against the lateral shake, with `analytic.pitch_safety_factor` | object turns less than `max_rotation_deg` (10 deg) |
| crushing | peak pad pressure against the material's allowable pressure | same, from the simulated pad elements |
| unsupported panel | pad force on the box face above the settled fill line against `unsupported_panel_force_N` | same |
| creasing | bending moment at the pinch against the sheet's crease moment | same, from the hinged sheet chain |
| bare finger | the finger body and pad mounts stay clear of the object, with no interference allowed | they put at most 1 N on the object (`BODY_CONTACT_LIMIT` in `mjsim.py`) |

Box insertion depth (how far the box goes between the fingers) is chosen per
object by the robot: `insertion: best` tries `insertion_scan` depths between
"pads just reach the top edge" and the deepest allowed insertion. When that
pad-registered depth is out of reach, as it is for Parth's paddle, the scan
runs shallower instead, down to the box top at mid-pad. Sheets use a fixed
`insertion_depth_mm`. `gripper.torque_mode: fixed` gives every object the
same drive torque.

`gripper.keep_out` lists circles in the jaw plane that fingers and objects
must stay out of. The shipped one is the FT Dynamic bracket (top)
(MV-UMI01-3DP-001 A): it carries both pivots, and its 35 mm hub is centred on
the jaw centreline 31 mm from the pivot line. Parth's finger clears the hub with
an R 18.36 cutout, a 0.86 mm gap, which `clearance_mm` keeps for every
candidate. Because the bracket sets the pivots, `pivot_spacing` is fixed at
38 mm; a range there implies a new bracket and gear pair. The geometry sweeps
each circle through the finger's frame over the whole stroke, from where the
fingers meet to full opening, and cuts that region out of the finger beam. A design is infeasible if a pad or pad
mount would have to be cut, or if less than `min_ligament_mm` (5 mm) of finger
is left beside the cutout. With `blocks_objects: true` an object's top must
also stay below the circle, which caps box insertion at 100.6 mm for a 150 mm
finger. `clearance_mm` adds to every radius.

## How it works

### Geometry and kinematics (`geometry.py`)

A finger is a polygon in its own frame (u along the finger, v inward) built
from the design variables; it turns about its pivot by the jaw angle phi (+
opens). The fingers are geared 1:1, so both turn by the same angle in opposite
directions. Closing on a box needs a diverging jaw (phi of +5 to +20 deg);
closing on a thin sheet needs the fingertips to meet, so the jaw converges
slightly. A design is infeasible when its pads overlap, leave the finger, sit
where they can never touch, or collide with the keep-out. The paddle style
puts the paddle and the tab on one face line with one rigid mount, so they
stay coplanar whatever the pad angle. The keep-out cutout is drawn in every
figure.

### Analytic screen (`analytic.py`)

For each object and insertion depth: the object must fit between the fully
open fingers, the pads (not the finger body) must touch it first, and the tips
must not meet above it. The jaw then closes until the pad compression balances
the drive torque. Pads are a Winkler foundation: bonded elastomer compression
modulus E_c = E(1 + 2S^2) (S is the shape factor, capped by the bulk modulus)
over the pad thickness, in series with the object's surface stiffness (board
through-thickness modulus; box panel on the cereal bed). Self-aligning pads
turn until their contact moment balances the centring spring. From the pressure
distribution the screen computes the margins in the table above. The friction
hold accounts for the moment the friction force puts on the jaw, and for the
squeeze-out force at a box's top corner. One design on all 13 objects takes
about 0.1 to 0.25 s.

### Dynamic check (`mjsim.py`)

A planar MuJoCo model: the gripper base on x and z slides under stiff position
servos; two finger hinges coupled 1:1 by a joint equality (the gear pair: the
hinge axes are mirrored, so equal joint angles are equal and opposite
rotations) and driven by torque motors at `finger_torque`, with damping that
limits the closing speed; each pad split into elements (one every
`element_pitch_mm`) on spring slide joints with the pad's foundation stiffness
and contact width, riding a shear joint with the elastomer's shear stiffness
(and a pin with stiff stops for self-aligning pads). A box is a
rigid body on a table. A sheet is a chain of rigid segments joined by hinges
with the board's bending stiffness, hanging from a holder that is released
when the jaw has closed. Friction uses MuJoCo's elliptic cones. Each run
records slip, rotation, pressure, panel force, bending moment, bare-finger load and friction use
(the largest share of μ times the normal force either pad uses) from the start
of the lift, so the closing impact (up to twice the static force at 1 rad/s)
counts only as the `closing_impact_pressure_kPa` metric. One run takes about
4 to 6 s.

### Search loop (`search.py`)

Each round:

1. **Ideate and screen.** Per concept, a Latin hypercube over the concept's
   variables (plus the nominal design and the previous round's leaders), then
   differential evolution on the all-rounder objective
   J = sum over objects of min(margin, 1) + 0.25 * min over objects of min(margin, 2).
   J counts the objects passed and rewards spare margin on the weakest one.
2. **Specialise.** For each object, differential evolution on that object's
   score alone, starting from the archive's best design for it.
3. **Simulate.** The top `finalists_per_concept` distinct designs of every
   concept and each object's specialist (and, in round 1, the reference design
   as drawn) are run through the MuJoCo check on every object. A design
   already checked in an earlier round (typically a
   specialist the new round could not beat) is not run again; it keeps its
   results, and the new reason is added to its `why` list.
4. **Learn.** Per object and criterion, the ratio of dynamic to analytic margin
   (25th percentile of all runs so far, clipped to [0.25, 2]) corrects the
   screen in the next round. The dynamic check's slip is a displacement, so the
   screen's friction hold margin is corrected with the share of the available
   friction the pads use in MuJoCo instead. Sheet bending is the clearest case
   for corrections: with the default shake, 1.5 mm chipboard (first bending
   mode near 2 to 3 Hz) sees about 2.3 times the rigid-sheet crease moment,
   0.5 mm chipboard about 0.3 times, and corrugated sheets about 1 to 1.1 times.

The best all-rounder is the finalist that passes the most objects in the
dynamic check (ties go to the higher dynamic J). The best design per object is
the finalist with the highest dynamic margin on that object.

### Outputs

`search --out DIR` writes:

| file | contents |
|---|---|
| `results.json` | the all-rounder, best design per object, best per concept, the reference design (`references`), and every finalist with its per-object dynamic and analytic results |
| `round1.json`, `round2.json`, ... | per-round summaries, the learned corrections, and the earlier designs each round reselected (`carried`) |
| `archive.json` | the analytic evaluations kept from each round (concept, variables, per-object scores): every Latin hypercube sample, the final evolved population and each per-object result; intermediate differential-evolution evaluations are not stored |
| `search.log` | progress and the final report |
| `*.yaml` | the inputs of the run |
| `figures/` | `first-pass-best-designs.png` (all-rounder and per-object best, closed on each object), `baseline-as-drawn.png` (Parth's finger on each object, MuJoCo results), `concepts-nominal.png` (analytic screen results), `concepts-best.png` (MuJoCo results), `object-library.png`, `pass-matrix.png`, `iteration-progress.png`, `designs-in-bounds.png`, `shake-traces.png` (all-rounder time histories, including the share of available friction used) |

## Tests

```bash
PYTHONPATH=. python -m pytest -q
```

They cover the config loader (unit conversion, fixed values, unknown names),
the jaw kinematics, the pad model, the object library, the analytic force
balance, agreement between the analytic and MuJoCo static grip, the baseline
against the CAD (pad positions and widths, 2.4 mm closed gap), the keep-out
(rejecting a paddle that reaches into the bracket, capping insertion), the
contact-width cap, and the gear coupling (driving one finger carries the
other).

## Limits of the model

- 2D: no torsion about the jaw axis, no out-of-plane tipping, no box bulging
  along its width. Each pad is extruded at constant width; Parth's face tapers
  (about 148 mm wide near the cutout to 88 mm at the end of the paddle), so
  the paddle uses its mean width.
- The keep-out is treated as full height. In the CAD the bracket only covers
  the upper part of the finger and the lower paddle passes under it, so the
  2D model is conservative there.
- Boxes are rigid bodies; panel denting and crushing are judged by force and
  pressure limits, not simulated deformation. Cereal does not flow.
- Sheets bend only in the plane, at hinges every `sheet_segment_mm`; creasing
  is judged at the first hinge below the pinch.
- The drive is an ideal torque source with viscous speed limiting; no gearbox
  backlash, compliance or controller dynamics.
- Material values are typical published values with wide spreads (see the
  comments in `objects.yaml`); humidity can halve corrugated board strength.
