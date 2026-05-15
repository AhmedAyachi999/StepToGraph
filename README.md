# StepToGraph

This project takes a STEP file, converts the solid into an attributed adjacency graph, and uses that graph to approximate manufacturing feature decomposition.

The main idea comes from the paper:

`Manufacturing feature recognition method based on graph and minimum non-intersection feature volume suppression`  
DOI: `10.1007/s00170-023-11031-x`

## Idea

A B-rep solid is turned into a graph:

- each face becomes a node
- each shared edge between two faces becomes a graph edge
- each graph edge is labeled as `convex`, `concave`, or `smooth`

The paper's key decomposition idea is:

- do not split on every convex edge
- keep only convex edges that form closed loops
- treat those loops as feature boundaries
- remove those boundaries from the graph
- the remaining connected components become candidate feature substructures

This project implements that pipeline with OpenCascade through `cadquery`.

## What The Project Produces

Given a STEP file, the project writes:

- `aag.json`: the attributed adjacency graph
- `aag.svg`: a 2D plot of the graph
- `feature_candidates/`: one JSON and one SVG per detected candidate substructure
- `colored_features.step`: the original solid with faces colored by detected group
- `colored_features.gltf`: a colored 3D export of the same solid
- `colored_features.html`: a simple browser viewer for the colored 3D model

## How The Method Works

The main function is in `step_graph.py`:

1. Load the STEP solid.
2. Enumerate all faces.
3. Find every shared edge between two faces.
4. For each shared edge, classify it as convex or concave by probing the local material side of the solid.
5. Keep only convex edges that survive a closed-loop filter.
6. Remove those boundary edges from the graph.
7. Take connected components of the remaining graph as candidate features.

This is a decomposition stage, not full feature recognition.

That means:

- the graph split is geometry-based
- the exported substructures are candidate features
- the names are still heuristic labels, not full template-matched manufacturing feature names

## How To Run

Use the virtual environment in the project:

```powershell
venv\Scripts\python.exe main.py s14-08.stp
```

## How To Read The Code

Start here:

- `main.py`
- `step_graph.py`

Inside `step_graph.py`, read these functions in order:

1. `build_attributed_adjacency_graph`
2. `_shared_edges`
3. `_boundary_edge_ids`
4. `_connected_groups`
5. `export_colored_model`

That is the full method from input solid to decomposed graph and colored output.

## Limits

This project does not yet implement the full final recognition stage from the paper:

- no subgraph-template matching
- no minimum non-intersection volume suppression loop
- no exact manufacturing feature naming

So the current output is best understood as:

`geometry-based feature candidate decomposition from a STEP B-rep`
