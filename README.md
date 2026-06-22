# STEP Cleaning Hotspot Predictor

This project analyzes STEP files, simulates surface cleaning, and predicts which faces are likely to stay dirty after cleaning.

The current desktop workflow is:

1. Open a STEP/STP file.
2. Predict learned hotspot faces with the trained LightGBM ranking model.
3. Run the cleaning simulation with a selected water force and compare the simulation result visually.

The pink faces in the UI are the model prediction. The colored simulation result is produced by running the cleaning simulation.

## Current Model

The active model is a **LightGBM LambdaRank model**:

- Model type: `lightgbm.LGBMRanker`
- Objective: `lambdarank`
- Saved model: `cache/lightgbm_hotspot_model.txt`
- Inference code: `features/hotspot_prediction.py`
- Training code: `train_hotspot_ranker.py`
- Current training cache: `cache/abc_cleaning_retention_2000_sample2mb_all_axis_wf05_rich`

This is a learning-to-rank model. It does not simply classify a face as dirty or clean. For each STEP object, it ranks all faces and tries to put the faces with the most retained particles at the top.

The target learned by the model is:

```text
retained_particle_marker_count
```

That means: after the cleaning simulation, how many particle markers stayed on each face.

The "stayed dirty after cleaning" result is the ground truth target. It is not the only input. The model also uses surface type, geometry, cleaning-simulation signals, concavity/convexity, and neighbor-face context.

## Training Parameters

These are the parameters used for the current trained model and the CSV data it was trained from.

| Parameter | Value used | Meaning |
| --- | --- | --- |
| Training data folder | `cache/abc_cleaning_retention_2000_sample2mb_all_axis_wf05_rich` | Folder containing the generated CSV files used for the current model. |
| STEP source dataset | `abc_datasets` | ABC STEP files were used as the training source. |
| Number of sampled STEP files | `2000` | The simulation was run on 2000 deterministic ABC STEP files. |
| Sampling seed | `42` | Makes the 2000-file sample repeatable. |
| Maximum STEP file size | `2 MB` | Large STEP files were skipped to keep simulation runtime practical. |
| Spray mode | `all-axis` | Water was sprayed from all six axis directions, not only top/bottom. |
| Water directions | `+X, -X, +Y, -Y, +Z, -Z` | These directions match the simplified UI prediction setup. |
| Water force | `0.5` | Multiplier for direct water dose during simulation. Lower means gentler cleaning and more retained particles. |
| Mesh linear deflection | `0.8` | Controls triangle mesh resolution for STEP surface sampling. Smaller values create denser meshes. |
| Mesh angular deflection | `0.5` | Controls angular mesh refinement. Smaller values preserve curved surfaces with more triangles. |
| Flow steps | `28` | Number of runoff propagation steps used after water reaches the surface. |
| Flow retention | `0.82` | Fraction of water/soil that continues through the simulated flow process. |
| Cleaning rate | `0.28` | Strength of dust removal when water reaches a sample. |
| Deposition rate | `0.20` | Strength of dirty-water redeposition onto downstream surfaces. |
| Remaining particle threshold | `0.02` | A particle marker is counted as retained if its remaining dust is at least this value. |
| Neighbor scope | `all` | Neighbor context was written for all faces, not only dirty faces. |
| Top surfaces per object | `4` | CSV reports keep the top four retained faces per object for ranking comparison. |
| Exact edge convexity | Enabled | CAD edge classification is used where possible to count convex, concave, and neutral boundaries. |
| Target column | `retained_particle_marker_count` | The value the model learns to rank. Higher means dirtier. |
| Ranking group | STEP object / dataset | Faces are ranked only against other faces from the same object. |
| Relevance labels | Top retained faces receive highest relevance | LambdaRank trains from ranked relevance labels, not raw regression alone. |
| Ranking cutoff | `top_k = 4` | Evaluation checks whether the model finds the top four dirtiest faces. |
| Maximum faces per object | `2000` | Very large ranking groups are capped while keeping relevant faces. |
| Model | `LGBMRanker` | LightGBM gradient-boosted decision-tree ranking model. |
| Objective | `lambdarank` | Ranking objective optimized for ordering faces correctly. |
| Metric | `ndcg` | Ranking metric that rewards correct ordering near the top of the list. |
| Trees | `n_estimators = 320` | Number of boosted trees. |
| Learning rate | `0.035` | Step size for each boosting iteration. |
| Leaves per tree | `num_leaves = 15` | Limits tree complexity. |
| Minimum child samples | `8` | Prevents leaves with too few training examples. |
| Row sampling | `subsample = 0.9` | Each tree uses 90% of rows. |
| Column sampling | `colsample_bytree = 0.9` | Each tree uses 90% of features. |
| L1 regularization | `reg_alpha = 0.05` | Penalizes overly complex trees. |
| L2 regularization | `reg_lambda = 0.2` | Additional regularization for stability. |
| Random state | `42` | Makes training and evaluation repeatable. |

## Feature Groups

The model does not use only surface-form statistics. It uses these feature groups:

| Feature group | Examples | Why it matters |
| --- | --- | --- |
| Surface form | `form_type` | Planes, cylinders, cones, spheres, tori, and other faces behave differently during cleaning. |
| Global form retention priors | `form_retained_particle_ratio`, `form_retained_particle_share_total` | Tells the model which surface forms tended to stay dirty in the training simulations. |
| Face size | `sample_count`, `surface_area` | Larger faces can retain more particles simply because they contain more sampled area. |
| Per-face cleaning signals | `area_weighted_water_dose`, `area_weighted_cleaning_dose`, `area_weighted_redeposition` | Describes how much water and cleaning actually reached the face. |
| Per-face hotspot geometry | `area_weighted_poor_drainage`, `area_weighted_concavity`, `area_weighted_hiddenness` | Captures trapped, shielded, or poorly draining regions. |
| Mesh boundary shape | `boundary_concave_count`, `boundary_mean_angle_score`, `boundary_max_concavity_score` | Sharp or concave boundaries can trap retained dirt. |
| Exact CAD edge convexity | `exact_boundary_convex_count`, `exact_boundary_concave_count` | Uses CAD edge classification where available, instead of only mesh estimates. |
| Neighbor context | `neighbor_area_weighted_concavity_mean`, `neighbor_area_weighted_cleaning_dose_mean`, `neighbor_boundary_concave_count` | Dirty regions often depend on nearby faces and boundaries, not only the current face. |

The full feature list is defined in `FEATURE_COLUMNS` in `train_hotspot_ranker.py`.

## Reproducing The Current Training Data

Generate the ABC cleaning-retention CSV files:

```powershell
.\occwl-env\python.exe analyze_cleaning_particle_retention.py `
  --dataset-dir abc_datasets `
  --output-dir cache\abc_cleaning_retention_2000_sample2mb_all_axis_wf05_rich `
  --sample-files 2000 `
  --sample-seed 42 `
  --max-file-size-mb 2 `
  --spray-mode all-axis `
  --water-force 0.5 `
  --top-surfaces-per-object 4 `
  --neighbor-scope all
```

Train the current ranking model:

```powershell
.\.venv311\Scripts\python.exe train_hotspot_ranker.py `
  --face-csv cache\abc_cleaning_retention_2000_sample2mb_all_axis_wf05_rich\face_particle_retention.csv `
  --neighbor-csv cache\abc_cleaning_retention_2000_sample2mb_all_axis_wf05_rich\dirty_surface_neighbor_context.csv `
  --form-csv cache\abc_cleaning_retention_2000_sample2mb_all_axis_wf05_rich\form_particle_retention.csv `
  --top-k 4 `
  --model-output cache\lightgbm_hotspot_model.txt
```

Run ABC holdout evaluation:

```powershell
.\.venv311\Scripts\python.exe train_hotspot_ranker.py `
  --holdout-test-size 0.2 `
  --top-k 4 `
  --model-output cache\lightgbm_hotspot_model.txt `
  --holdout-predictions-output cache\hotspot_ranker_holdout_predictions_wf05_abc2000.csv `
  --importance-output cache\hotspot_ranker_feature_importance_wf05_abc2000_holdout.csv
```

Latest measured results from the current run:

| Evaluation | Result |
| --- | --- |
| ABC grouped holdout top-4 accuracy | `72.55%` |
| ABC grouped holdout exact top-4 match | `44.51%` |
| ABC grouped holdout NDCG@4 | `80.10%` |
| Parca test top-4 accuracy | `64.91%` |
| Parca macro top-4 accuracy | `66.67%` |
| Parca exact top-4 match | `26.67%` |

## Desktop UI

Open the native OpenCascade viewer:

```powershell
.\occwl-env\python.exe desktop_ui.py
```

Open a specific file:

```powershell
.\occwl-env\python.exe desktop_ui.py step_datasets\perfect_L_no_holes.step
```

The simplified UI contains these controls:

| Control | Meaning |
| --- | --- |
| `Open STEP File` | Opens a STEP/STP file from disk. |
| `Predict Hotspots` | Loads `cache/lightgbm_hotspot_model.txt` and colors the model-ranked hotspot faces in pink. |
| `Run Cleaning Simulation` | Runs the cleaning simulation and colors the faces where particles stayed after cleaning. |
| `Water Force` | Sets the cleaning force from `0.000` to `1.000`. The default is `0.500`. |

The UI currently uses all-axis water directions for model prediction and simulation. This matches the current model training data.

## Architecture

- `main.py`: command-line entry point.
- `desktop_ui.py`: Tk/OpenCascade desktop viewer.
- `analyze_cleaning_particle_retention.py`: batch simulation runner that generates training CSV files.
- `train_hotspot_ranker.py`: LightGBM LambdaRank training and evaluation.
- `features/hotspot_prediction.py`: model inference for new STEP files.
- `features/cleaning_simulation/`: meshing, cleaning simulation, runoff, redeposition, and export helpers.
- `features/edge_classification/`: convex, concave, and neutral edge classification.
- `features/hole_finding/`: inward-cylinder and analytic surface-form detection.
- `step_datasets/`: local STEP examples.
- `abc_datasets/`: ABC STEP training dataset.

## Dependencies

Use the OpenCascade environment for STEP parsing and the desktop UI:

```powershell
.\occwl-env\python.exe desktop_ui.py
```

Use the Python 3.11 ML environment for LightGBM training:

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv311\Scripts\python.exe train_hotspot_ranker.py
```

If the UI environment needs to run model prediction, install the ML dependencies there as well:

```powershell
.\occwl-env\python.exe -m pip install -r requirements-ml.txt
```
