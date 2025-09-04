## Dataset

The dataset for PaTS consists of solved planning problem instances from the Blocksworld domain.

### Generation Process

The dataset is generated using the `data/generate_dataset.sh` script. This script automates:

1.  **Problem Generation**: Creates PDDL problem files (`.pddl`).
2.  **Plan Generation**: Uses the Fast Downward planner to find a solution (`.plan`).
3.  **State Extraction**: Uses VAL to validate the plan and log all state changes (`.val.log`).
4.  **Parsing and Encoding**: The `data/parse_and_encode.py` script processes the logs and PDDL files. It reconstructs the state trajectory and encodes each state into the vector format specified by the `--encoding_type` flag (`bin` or `sas`).
5.  **Dataset Splitting**: `data/analyze_dataset_splits.py` analyzes the plan length distribution and creates stratified `train_files.txt`, `val_files.txt`, and `test_files.txt`.

### Data Structure and Format

All generated data for `N` blocks with a specific encoding is organized within a directory like `data/blocks_<N>-<encoding>/` (e.g., `data/blocks_4-sas/`). Key files include:

- `pddl/`, `plans/`, `val_out/`: Directories containing the raw PDDL, plan, and VAL log files.
- `trajectories_text/`: Human-readable state trajectories.
- `trajectories_bin/`:
  - `...traj.<encoding>.npy`: NumPy array `(L, F)` of encoded states (e.g., `.traj.sas.npy`).
  - `...goal.<encoding>.npy`: NumPy array `(F,)` of the encoded goal state.
- `encoding_info_<N>.json`: **Crucial file** describing the encoding used (type, feature dimension, manifest path, etc.). This file makes the dataset self-describing.
- `predicate_manifest_<N>.txt`: For `bin` encoding only, this lists all predicates in order, defining the feature map.
- `train_files.txt`, `val_files.txt`, `test_files.txt`: Lists of problem basenames for each data split.

### State Encoding

PaTS supports multiple state encoding schemes, controlled by the `--encoding_type` flag in `parse_and_encode.py`.

#### Binary Predicate Encoding (`--encoding_type bin`)

- **Representation**: A sparse binary vector where each element corresponds to a ground predicate (e.g., `(on-table b1)`). `1` means true, `0` means false.
- **Size**: Scales quadratically with the number of blocks, O(n²).
- **Configuration**: Defined by a `predicate_manifest_<N>.txt` file. The `encoding_info_<N>.json` file will specify the type as `bin` and point to this manifest.

#### SAS+ Position Vector Encoding (`--encoding_type sas`)

- **Representation**: A dense integer vector where the _index_ represents the block and the _value_ represents its position.
  - `vector[i]` corresponds to block `b(i+1)`.
  - Value `0`: The block is on the table.
  - Value `j > 0`: The block is on top of block `bj`.
  - Value `-1`: The block is being held by the arm.
- **Example (4 blocks)**: The state "A on B, B on table, C on D, D on table" (with A=b1, B=b2, C=b3, D=b4) is encoded as `[2, 0, 4, 0]`.
- **Size**: Scales linearly with the number of blocks, O(n).
- **Configuration**: The `encoding_info_<N>.json` file will specify the type as `sas` and list the block order. No separate manifest file is needed.
