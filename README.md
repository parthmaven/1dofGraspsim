# 1dofGraspsim

## Goal

This project aims to build a utility that simulates gripper designs (size, shape, and mechanism inputs) against different grasping tasks.

## Current objective

A 2D design search. Given a bounding box of design variables (a range for each), the tool ideates, simulates, and iterates gripper shapes that can grasp a predefined set of objects:

- Filled cereal boxes of various sizes.
- Sheets of cardboard of various thicknesses.

## Required capabilities

- Good rigid-body dynamics.
- A toolset to generate and draw out mechanisms.
- Friction.
- Some compliance.

## Nominal gripper geometry

- 1-DoF pinch gripper with two synced fingers.
- The two finger pivots are 38 mm apart.
- The gripper tip is 150 mm forward of the pivots.

## Status

The simulation stack has not been chosen yet because a literature review is in progress.
