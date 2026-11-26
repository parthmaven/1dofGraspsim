# 2D gripper design search

`graspsim2d` searches a bounding box of design variables for the finger shape of
a 1-DoF, two-finger angular gripper that can pick up filled cereal boxes and
sheets of cardboard. It generates several distinct finger concepts, screens
thousands of variants of each with a fast analytic grasp model, checks the
leaders in a planar MuJoCo simulation (close, lift, hold, shake, hold), feeds
what the simulation learned back into the screen, and reports the best
all-rounder and the best design for each object.

The bounding box is centred on the platform finger
(`Finger, Left, MV-UMI01-3DP-001 A.stp`): two pivots 38 mm apart, geared 1:1
(equal and opposite rotation), the tip 150 mm from the pivot line, and a flat
face that is parallel to the other finger's when closed. The contact surface is
the inner face of the finger's 3 mm blade, 4 mm inboard of the paddle part (the
`Paddle, Gripper, MV-UMI01 A` plate that carries the pivot barrel), so the
paddle part never touches the object. Along its length the face is a wide main
face, a recess and a narrow tip pad, which the model calls the paddle, the notch
and the tab. That design is concept C1, and it is always simulated as drawn for
reference. Each finger can open 90° from where the fingers meet, and a grasp
opens only as far as its object needs. Every candidate must clear the hub of
the FT Dynamic bracket (top) at every angle of that swing: each finger is cut
to clear it, as the platform finger's notch does. It must also clear both FT
Dynamic brackets, top and bottom, over the same swing.

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
python -m graspsim2d screen baseline_paddle        # analytic screen of one design on every object (C1: the platform finger)
python -m graspsim2d simulate self_aligning --objects hbo_28oz,b_flute_3mm --plot traces.png   # MuJoCo check
python -m graspsim2d simulate baseline_paddle --fidelity analytic --out runs/baseline   # either level, results saved
python -m graspsim2d sweep baseline_paddle --vary finger_torque=1:12:12 --out runs/torque   # 1-D sweep, analytic
python -m graspsim2d sweep crowned --vary finger_torque=2,4,6,8 --vary pad_shore=30,50,70 --fidelity mujoco --out runs/grid
python -m graspsim2d sweep baseline_paddle --vary finger_torque=0.25:5:20 --fidelity mujoco --out runs/baseline_torque   # the platform finger against torque
python -m graspsim2d search --out runs/first_pass  # the full ideate / simulate / iterate loop
python -m graspsim2d --design-space config/design_space_stroke45.yaml search --out runs/first_pass_stroke45   # the same at a 45° stroke
python -m graspsim2d search --quick --out runs/smoke   # tiny budget, one round
python -m graspsim2d search --quick --rounds 2 --out runs/smoke2   # tiny budget, two rounds
python -m graspsim2d figures runs/first_pass       # redraw the figures of a finished search, or of a one- or two-variable sweep
```

`screen`, `simulate` and `sweep` take a concept name (its nominal design), a
design from a finished run as `runs/first_pass/results.json:r2-03`, or a JSON
file with design variables, and `--objects` picks objects by id (default: all).
`screen` is `simulate --fidelity analytic`. `simulate --xml DIR` writes the MJCF
models so a run can be opened in the MuJoCo viewer
(`python -m mujoco.viewer --mjcf=DIR/hbo_28oz.xml`).

`sweep --vary name=a,b,c` lists values and `name=lo:hi:n` spaces n values
evenly, in the units `describe` prints; repeated `--vary` options make a grid
(their Cartesian product) around the given design. Values outside the bounding
box are allowed and flagged. `sweep` runs at analytic fidelity unless
`--fidelity mujoco`.

`simulate`, `sweep` and `search` use every core; `--workers N` limits them
(`workers` in `search.yaml` also does for `search`). With `--out DIR`, `screen`,
`simulate` and `sweep` save their results there with a copy of their inputs, as
`search` always does (see Outputs).

`--config DIR` points every command at another config directory;
`--design-space FILE`, `--objects FILE`, `--scenario FILE`, `--concepts FILE`
and `--search FILE` override single files.

The full search with the shipped budget (2 rounds) takes about 21 minutes on
8 cores (1,252 s at the 90° stroke and 1,244 s at 45°).

## Configuration

All inputs are YAML files in `config/`. A run copies them into its output
directory, so every result keeps the inputs that produced it.

| file | what it holds |
|---|---|
| `design_space.yaml` | the bounding box: a range (or fixed value) for each design variable |
| `concepts.yaml` | the finger concepts, which variables each one searches, any narrower ranges, starting values, and which concept is the reference |
| `objects.yaml` | materials, boxes, sheets and the box tables to load, with sources for every value |
| `general_mills_cereal_boxes.csv` | a box table: one cereal box per row (see Objects) |
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
| `pad_tip_margin` | mm | fingertip to the distal end of the pad; for the paddle style, the notch plus the tab |
| `pad_width` | mm | out-of-plane width of the pad (the paddle, for the paddle style) |
| `tab_width` | mm | paddle style: out-of-plane width of the tab carrying the face on to the tip |
| `tab_gap` | mm | paddle style: length of the recessed notch between the paddle and the tab |
| `pad_shore` | Shore A | pad hardness, converted to a modulus (Gent 1958) |
| `pad_crown_curvature` | 1/m | pad face curvature (0 = flat) |
| `finger_torque` | N*m | drive torque at each finger pivot (the 1:1 gears give both fingers the same), one value for every object |
| `max_opening` | deg | stroke of each finger: its largest opening, measured from where the fingers meet (fixed at 90: the fingers open more than 90°). A grasp opens only as far as the object needs |
| `pad2_offset`, `pad2_length`, `pad2_angle`, `pad2_protrusion` | mm, mm, deg, mm | dual-zone concept: the proximal box pad |
| `pivot_depth`, `pivot_stiffness`, `pivot_range` | mm, N*m/rad, deg | self-aligning concept: the pad's pivot pin and centring spring |

To add a variable, add a `VariableDef` to `SCHEMA` in `graspsim2d/schema.py`
and use it in `graspsim2d/geometry.py`.

### Concepts

| concept | idea |
|---|---|
| C1 `baseline_paddle` | the platform finger: straight, a wide paddle (49.9 to 101.5 mm from the pivot line, 95 mm mean width), a 28 mm recessed notch, and a 16 mm tab carrying the same face to the tip, parallel to the other finger's face when closed (2.5 mm gap) |
| C2 `cranked_flat` | the distal section is offset inward, so the fingers can be parallel on a thin sheet while the proximal part clears a box |
| C3 `crowned` | convex pad face; the contact point rolls along the pad as the jaw angle changes |
| C4 `dual_zone` | a tip pad for sheets plus a proximal box pad; the closing angle picks the zone |
| C5 `self_aligning` | the pad turns on a passive pin with a centring spring and stops, so it lies flat on any face |

A new concept is an entry in `concepts.yaml` (which variables it searches and
which pad style it uses); a new pad style needs code in `geometry.py` and,
for the dynamic check, `mjsim.py`. `nominal:` gives a concept its own starting
values where the baseline's do not suit it (C2 to C5 start with a 40 mm pad
5 mm from the tip). `reference: true` (set on C1) puts the concept's nominal
design, the platform finger as drawn, through the dynamic check on every object in
round 1 and reports it as `baseline`.

### Objects

Cereal boxes stand upright on a table. The gripper comes down on the top end
(the shortest end) and closes across the depth, so the box's height x depth
side panel lies in the jaw plane and the width is out of plane.
`grasp_axis: width` in `box_defaults` (or on one box) closes across the width
instead. Cereal settles at the bottom: the contents fill the box up to the fill
height, so the centre of mass sits below mid-height, which sets the pitch lever
arm and the box inertia, and the panel above the fill line is backed only by
air (a pad there may push at most `unsupported_panel_force_N`).

Boxes come from the `boxes` list in `objects.yaml` and from the CSV box tables
listed under `box_tables`, one box per row, with the columns `product_title,
size_label, upc, net_mass_g, box_mass_g, height_mm, width_mm, depth_mm,
fill_height_mm, com_height_mm, bulk_density_g_cm3, source, confidence`. A blank
`fill_height_mm` falls back to the bulk density (net mass over density and
footprint), then to `fill_fraction` (0.8); a blank `com_height_mm` uses the
carton-plus-contents estimate; a blank `box_mass_g` uses the carton blank area
times the board grammage. `config/general_mills_cereal_boxes.csv` is a copy of
the General Mills catalog (`docs/general-mills-cereal-boxes.csv` in the project
store), which so far holds Cheerios Family Size 18 oz: 325 x 196 x 94 mm, 626 g
with the carton, settled fill 234 mm (72 %, 91 mm headspace) and centre of
mass 126 mm above the bottom, both modelled from its bulk density. To take a
newer catalog, copy it over that file or point `file:` at it (a relative path
is relative to `objects.yaml`). A table box's id is the prefix `gm_` plus its
UPC, unless the table entry's `names` (keyed by UPC) gives a short id and label,
as it does for `gm_cheerios_family_18oz`. Rows with confidence `placeholder`
are labelled so.

Sheets hang from a holder that lets go once the jaw has closed, and are
pinched near their free top edge. Every value in `objects.yaml` carries its
source and spread in a comment. An object or box table with `enabled: false`
stays in the file but is skipped.

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
pad-registered depth is out of reach, as it is for the platform finger, the scan
runs shallower instead, down to the box top at mid-pad. Sheets use a fixed
`insertion_depth_mm`. `gripper.torque_mode: fixed` gives every object the
same drive torque.

`gripper.keep_out` lists circles in the jaw plane that fingers and objects
must stay out of. The shipped one is the FT Dynamic bracket (top)
(MV-UMI01-3DP-001 A): it carries both pivots, and its 35 mm hub is centred on
the jaw centreline 31 mm from the pivot line. The platform finger clears the
hub with an R 18.5 notch, a 1.0 mm gap, which `clearance_mm` keeps for every
candidate. Because the bracket sets the pivots, `pivot_spacing` is fixed at
38 mm; a range there implies a new bracket and gear pair. The geometry sweeps
each circle through the finger's frame over the whole stroke, from where the
fingers meet to full opening, and cuts that region out of the finger beam. A design is infeasible if a pad or pad
mount would have to be cut, or if less than `min_ligament_mm` (5 mm) of finger
is left beside the cutout. With `blocks_objects: true` an object's top must
also stay below the circle, which caps box insertion at 100.5 mm for a 150 mm
finger. `clearance_mm` adds to every radius.

`keep_out.brackets` holds the plan footprints of both FT Dynamic brackets,
taken from the CAD. In the assembly the finger spans 1–162 mm below the top
datum. The top bracket spans 1–14.5 mm. The bottom bracket spans 14.5–67.3 mm:
the hub tube, and a plate that rings both pivots and runs 38 mm back behind
them. The brackets are not cut from the fingers. Instead, each candidate must
clear them over its whole stroke, or it is infeasible, and its least gap is
recorded. Every candidate keeps the platform finger's root, the first
`root_length_mm` (46 mm) ahead of the pivot line, which carries the hinge and
is shaped around the brackets in 3D: at their heights it has material only on
its outer side. In 3D the platform jaw (finger, paddle part and hinge
interfaces) clears both brackets at jaw angles from −1.5° to 130.5°
(`root_clear_deg`): at −2° and from 130.75° the finger meets them. A
candidate's stroke must lie in that band. The 2D check covers the rest of the
beam, which the design variables change, and every pad and pad mount wherever
it sits. It leaves out the hub, which the circle above handles, and the disc
of `hinge_radius_mm` (18.5 mm) around each pivot, where the bracket's pivot
lobe and the finger's hinge share the axis at different heights.

Past the stroke, `widest_opening` steps the hub and the brackets on until the
first touch, and each finalist records the result as `widest_opening_deg`,
with what stops it (`widest_opening_limit`). Parts beyond the root sit more
than 46 mm from the pivot, and no bracket material outside the hub is more
than 38.1 mm from one, so the limit is always the root's 130.5° less the angle
at which the fingers meet.

## Architecture

```text
config/  design_space.yaml  concepts.yaml  objects.yaml + box tables (.csv)  scenario.yaml  search.yaml
            |                  |              |                              |              |
         schema.py ------------+          objects.py                    scenario.py     search.py
       (GripperDesign)                   (GraspObject)                   (Scenario)    (search loop)
            |                                 |                              |              |
         geometry.py: finger polygons,        |                              |              |
         1:1 gearing, swept hub keep-out,     |                              |              |
         bracket check                        |                              |              |
            |                                 |                              |              |
            +---------------> simulator.py: make_simulator("analytic" | "mujoco").run(design, object, scenario)
                                 |  analytic.py (screen)      mjsim.py (MuJoCo check)
                                 v
                              GraspOutcome: passed, score, limiting criterion, margins, metrics
                                 |
              cli.py: describe | screen | simulate | sweep (sweep.py) | search | figures (plots.py)
                                 |
                              runs/<name>/: results, inputs, figures
```

| module | role |
|---|---|
| `schema.py` | the typed design-variable schema (`SCHEMA`, `GripperDesign`), the bounding-box loader (units, fixed values, unknown names rejected) and the concepts |
| `objects.py` | the object library: materials, boxes from YAML and CSV box tables (fill height, centre of mass, grasp axis) and sheets, as `GraspObject`s |
| `scenario.py` | presentation, drive, keep-out, motion and shake, pass criteria and simulation settings, as a `Scenario` |
| `geometry.py` | the parametric finger: one polygon per finger for each pad style, the 1:1 geared jaw kinematics, the stroke from where the fingers meet, the FT-hub keep-out swept over the stroke and cut from the beam, the ligament, the least hub clearance, the bracket check over the stroke, feasibility |
| `pads.py` | the elastomer: Shore A to modulus, bonded compression modulus |
| `analytic.py` | level 0, the quasi-static screen (`solve_grip`) |
| `mjsim.py` | level 1, the planar MuJoCo model (`build_model`) and run (`simulate`) |
| `simulator.py` | one interface over both levels: `AnalyticSimulator` and `MujocoSimulator` return the same `GraspOutcome`; `run_batch` spreads designs x objects over all cores |
| `sweep.py` | grids over design variables around one design |
| `search.py` | the concept search (below); its MuJoCo checks go through `MujocoSimulator` |
| `plots.py` | the figures |
| `cli.py` | the commands; every saved run gets a copy of its inputs, box tables included |

Both levels score a grasp the same way: one margin per pass criterion (table
above), each at least 1 to pass; the score is the smallest margin and the
limiting criterion is the one that sets it. A new object is a YAML entry or a
box-table row; a new criterion is a margin in `analytic.py` and `mjsim.py`; a
new simulation level is a class with `fidelity` and `run()` registered in
`make_simulator`.

`.github/workflows/ci.yml` runs the tests and a smoke run of every command
(`describe`, `screen`, `simulate`, `sweep`, `search --quick`) on every pull
request and push to `main`, and keeps the run directories as a build artifact.

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
stay coplanar whatever the pad angle, with a recessed notch of `tab_gap`
between them. The stroke runs from where the fingers meet (`meet_deg`: the
angle at which the closing fingers first touch on the centreline, or the closed
limit) to `max_opening` beyond it. The keep-out cutout is drawn in every
figure. Every finalist records its least clearance to the hub over the stroke
(`hub_clearance_mm`) and to the brackets (`bracket_clearance_mm`, with
`bracket`), each with the opening from where the fingers meet at which it
occurs. The final report lists them, and `bracket-clearance.png` plots them
against opening.

### Analytic screen (`analytic.py`)

For each object and insertion depth, the jaw first opens only as far as the
object needs. The approach opening is the smallest jaw angle at which both
fingers clear the object by `FIT_CLEARANCE` (1 mm) at every height down to that
depth. It is recorded as `approach_opening_deg` (the jaw angle) and
`approach_stroke_deg` (the opening from where the fingers meet). An object that
needs more than the stroke does not fit. Closing from there, the pads (not the
finger body) must touch it first, and the tips must not meet above it. The jaw
then closes until the pad compression balances the drive torque. The pad face
is sampled at points, so on a stiff contact the torque can jump past the
drive's within one bisection step. The final load is therefore scaled to the
drive torque: in the deepest rigid validation case below, 98.0 N at 5 N·m. Pads are a Winkler foundation: bonded elastomer compression
modulus E_c = E(1 + 2S^2) (S is the shape factor, capped by the bulk modulus)
over the pad thickness, in series with the object's surface stiffness (board
through-thickness modulus; box panel on the cereal bed). Self-aligning pads
turn until their contact moment balances the centring spring. From the pressure
distribution the screen computes the margins in the table above. The friction
hold accounts for the moment the friction force puts on the jaw, and for the
squeeze-out of a box by a tilted pad face. Where the face is tilted to the box
side by θ, the contact normal tilts by the box's share of the overlap, so the
squeeze also pushes the box toward the fingertips. At the box's top edge,
which meets the face edge-on, that share is all of it, so a grasp on the top
edges alone needs μ ≥ tan θ. A face that closes toward the tips gets no credit
for wedging the box in. The screen also reports the jaw opening at which an
object passes between the fingertips (`pass_opening`), and the least grip per
finger that hold and pitch each need, without and with their safety factors
(`min_grip_hold_N`, `min_grip_hold_sf_N`, `min_grip_pitch_N`,
`min_grip_pitch_sf_N`). Near a box's top, pitch can set the grip rather than
the weight. One design on all 14
objects takes about 0.2 to 0.45 s.

### Validation: the platform finger on Cheerios Family Size

The CAD study of the platform finger on the 94 mm deep Cheerios Family Size box
(`docs/general-mills-cereal-boxes.md` in the project store, with the true-scale
render `media/cereal-boxes/cheerios-family-size-with-gripper.png`) found the following:

- The jaws must open 17.4° each (a 94 mm tip gap) to pass the box.
- They then meet only its top edges, because the faces are tilted at the jaw angle.
- The squeeze therefore pushes the box out, so holding needs μ ≥ tan θ.

`test_cheerios_family_size_validation` checks the screen against it. The test
models the stock face as rigid printed plastic (the pad at 90 A, with the face
kept where the CAD has it) on a rigid box edge, as the study does:

| case (CAD study) | CAD study | screen |
|---|---|---|
| box just passes between the tips | 17.4° per jaw, 94 mm tip gap | 17.45°, 94.0 mm |
| tip-pad pinch as drawn: box top 18 mm inside the tips | first touch at 20.0°, 107 mm tip gap, contact 131 mm from the pivot line, μ ≥ tan 20.0° = 0.36 | insertion 20.8 mm: box top 17.9 mm inside the tips, first touch at 19.97° (tan 0.363) on the tab at 131.0 mm, top edge only; loaded at 19.72°, μ ≥ 0.358 |
| deepest contact on the tip pad | 20.7°, contact at 126 mm, μ ≥ 0.38 | 20.2°, 128.4 mm (the model's tab starts at 129.5 mm) |
| corner on the arm and main face | first touch at 24.6° to 46.5°, μ ≥ 0.46 to 1.05 | main face at insertions of 52 to 99 mm: first touch at 26.0° to 46.6° (tan 0.488 to 1.057), corner 100 to 55 mm out; loaded, μ ≥ 0.48 to 1.04. The model's notch is recessed finger body, so the depths that would put the corner on the CAD's arm (41.5 to 49 mm) are rejected |
| deepest main-face case: box top 51 mm below the pivots, 65 mm inside the open tips | first touch at 46.5°, contact 55 mm from the pivot line, μ ≥ 1.05 | insertion 99 mm: 65.0 mm inside the tips, first touch at 46.58° (tan 1.057), 55.4 mm; loaded at 46.11°, μ ≥ 1.04 |

The study's μ is the tangent of the angle at first touch. The screen's is the
tangent of the loaded face angle, which sits a quarter to half a degree further
closed, so it reads slightly lower.

The model's 6 mm, 40 A pad sits on a box wall with the cereal-backed stiffness,
top edge included. With that pad, the tip-pad pinch dents the wall about 8 mm
under 5 N·m. The jaw then closes to about 16°, and the finger body meets the
box. The screen's own pick puts the top edge on the main face (insertion
74 mm, 29.6° per jaw), where the squeeze needs μ ≥ 0.57, more than the
carton's 0.5. So the hold fails (margin 0.63), as the CAD study expects of
the bare main face.

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
counts only as the `closing_impact_pressure_kPa` metric. Each run starts 1°
beyond the object's approach opening (`START_MARGIN`; `start_opening_deg` in the
metrics), not at the full stroke. The closing speed is damping-limited, so the
speed at contact is the same either way. The close lasts `close_s`, or longer
when the jaw starts wide: the free travel from the start opening to where the
fingers meet at `closing_speed_rad_s`, plus `squeeze_s` (`close_time_s` in the
metrics). The finger body collides as cut: where the hub cut runs through a
body rectangle, `collision_rects` slices it across its length into strips at
most 3 mm wide, each holding only the material present over its whole width,
so the hub notch stays open. A dropped object scores 0, and the run keeps every
other margin measured up to the drop, with slip at 0. One run takes about 4 to
6 s.

A run that blows up is never scored. MuJoCo would reset it to t = 0 and carry
on, so its automatic reset is off, and every step checks MuJoCo's bad-state
warnings and whether time ran backwards. The run is flagged and rerun at half,
then a quarter, of `timestep_s`, with every contact and joint time constant
kept. The metrics record how many tries blew up (`resets`), when the first did
(`reset_s`), and the timestep of the run that was scored (`timestep_s`). If all
three blow up, the result is "simulation went unstable", with no margins.

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
   mode near 2 to 3 Hz) sees about 3 times the rigid-sheet crease moment
   (interquartile range 2.7 to 4.4 across the simulated designs), 0.5 mm
   chipboard about 0.3 times, and corrugated sheets about 1.1 to 1.3 times
   (medians; more on some pad layouts).

The best all-rounder is the finalist that passes the most objects in the
dynamic check (ties go to the higher dynamic J). The best design per object is
the finalist with the highest dynamic margin on that object.

### Outputs

`screen --out DIR` and `simulate --out DIR` write `results.json` (the design
and one outcome per object), `results.csv` (one row per object: pass, score,
limiting criterion, each margin and the key metrics) and the inputs.
`sweep --out DIR` writes `sweep.json` (the spec, every grid point's design and
every outcome), `sweep.csv` (one row per point and object), `summary.csv` (one
row per point: objects passed, J, smallest margin and the object that sets it),
`figures/sweep.png` and the inputs. For one varied variable, `sweep.png` plots
each object's smallest margin, each box's empty-panel margin (open markers
where the box dropped) and the objects passed; for two, the objects passed over
the grid. `figures DIR` redraws it.

`search --out DIR` writes:

| file | contents |
|---|---|
| `results.json` | the all-rounder, best design per object, best per concept, the reference design (`references`), and every finalist with its per-object dynamic and analytic results, its hub and bracket clearances, its widest clear opening and its length as drawn (`drawn_length_mm`) |
| `round1.json`, `round2.json`, ... | per-round summaries, the learned corrections, and the earlier designs each round reselected (`carried`) |
| `archive.json` | the analytic evaluations kept from each round (concept, variables, per-object scores): every Latin hypercube sample, the final evolved population and each per-object result; intermediate differential-evolution evaluations are not stored |
| `search.log` | progress and the final report |
| `*.yaml`, `*.csv` | the inputs of the run, box tables included |
| `figures/` | `first-pass-best-designs.png` (all-rounder and per-object best, closed on each object), `baseline-as-drawn.png` (the platform finger on each object, MuJoCo results), `concepts-nominal.png` (analytic screen results), `concepts-best.png` (MuJoCo results), `object-library.png`, `pass-matrix.png`, `iteration-progress.png`, `designs-in-bounds.png`, `bracket-clearance.png` (the brackets in plan with the all-rounder swung through its stroke, and the least gap to the hub and each bracket against opening), `shake-traces.png` (all-rounder time histories, including the share of available friction used) |

## Tests

```bash
PYTHONPATH=. python -m pytest -q
```

They cover the config loader (unit conversion, fixed values and the nominal
they set, unknown names), the jaw kinematics, the pad model, the object
library, the analytic force balance, the least grip for hold and pitch,
agreement between the analytic and MuJoCo static grip, the baseline against the
CAD (pad positions and widths, 2.4 mm closed gap, 1.0 mm hub clearance over the
stroke), the keep-out (rejecting a paddle that reaches into the bracket,
capping insertion), the 90° stroke from where the fingers meet (each grasp's
approach opening lies inside it), the brackets (the baseline clears both over
the whole stroke; a full-width root would not), the contact-width cap, the
drawn length of a cranked tip, the gear coupling (driving one finger carries
the other), the close time growing with the start opening, the collision model
keeping the hub notch open, dropped runs keeping every margin, the box tables
(given and fallback fill height and centre of mass, missing columns, ids by
UPC), the grasp axis, the simulator interface (both levels match direct calls),
sweeps (grid order, saved files, agreement with the screen, redrawing a sweep's
figure), run directories keeping their box tables, and the Cheerios Family Size
validation above, including the first-touch and loaded angles. CI runs them on
every pull request.

## Limits of the model

- 2D: no torsion about the jaw axis, no out-of-plane tipping, no box bulging
  along its width. Each pad is extruded at constant width; the platform
  finger's face tapers (139 mm high at 55 mm from the pivot line to 97 mm at
  85 mm, then the 27 mm arm edge to 101.5 mm), so the paddle uses the mean
  width over its length (95 mm), which keeps the contact area.
- The hub keep-out is treated as full height. In the CAD the hub and its tube
  occupy the top 67 mm of the finger's height, and the finger's lower blade
  reaches back under the tube; no object can reach that region, so the
  contact model loses nothing, but the cutout is applied to the whole beam.
- The brackets are checked in plan. Every candidate is taken to keep the
  platform finger's root (the first 46 mm ahead of the pivot line, with the
  hinge), whose 3D shape clears both brackets. A candidate that changes the
  root needs a 3D check. Pads and pad mounts are checked wherever they sit, as
  if they spanned the brackets' heights.
- Boxes are rigid bodies; panel denting and crushing are judged by force and
  pressure limits, not simulated deformation. Cereal does not flow.
- A box side has one surface stiffness, the cereal-backed panel's, up to and
  including its top edge. The real top edge is braced by the glued top flaps
  and is much stiffer, and the cereal crush pressure limit is applied even in
  the empty headspace, where the panel force limit is the real constraint.
- Sheets bend only in the plane, at hinges every `sheet_segment_mm`; creasing
  is judged at the first hinge below the pinch.
- The drive is an ideal torque source with viscous speed limiting; no gearbox
  backlash, compliance or controller dynamics.
- Material values are typical published values with wide spreads (see the
  comments in `objects.yaml`); humidity can halve corrugated board strength.
