## Dataset

The dataset for PaTS consists of solved planning problem instances from the chosen domain (e.g., Blocksworld).

### Generation Process

The dataset is generated using the `data/generate_dataset.sh` script. This script automates:

1.  **Problem Generation**: PDDL problem files (`.pddl`) are created.
2.  **Plan Generation**: Fast Downward finds a solution plan (`.plan`).
3.  **Plan Validation & State Extraction**: VAL validates the plan and its verbose output (`.val.log`) details state changes.
4.  **Parsing and Encoding**: `data/parse_and_encode.py` processes PDDL files and VAL logs. It is controlled by the `--encoding_type` flag (`binary` or `sas`). It reconstructs the state trajectory, encodes each state into the chosen vector format, and saves the encoded data.
5.  **Dataset Splitting**: `data/analyze_dataset_splits.py` creates `train_files.txt`, `val_files.txt`, `test_files.txt`.

### Data Structure and Format

All generated data for `N` blocks is organized within `data/blocks_<N>/`. Key files per problem instance `blocks_<N>_problem_<M>`:

- `pddl/blocks_<N>_problem_<M>.pddl`: PDDL problem.
- `plans/blocks_<N>_problem_<M>.plan`: Expert plan.
- `val_out/blocks_<N>_problem_<M>.val.log`: VAL output.
- `trajectories_text/blocks_<N>_problem_<M>.traj.txt`: Human-readable trajectory.
- `trajectories_bin/blocks_<N>_problem_<M>.traj.<encoding>.npy`: NumPy array `(L, F)` of encoded states (e.g., `.traj.binary.npy` or `.traj.sas.npy`).
- `trajectories_bin/blocks_<N>_problem_<M>.goal.<encoding>.npy`: NumPy array `(F,)` of the encoded goal state.
- `encoding_info_<N>.json`: **Crucial file** describing the encoding used for this dataset (type, feature dimension, and path to manifest if applicable).
- `predicate_manifest_<N>.txt`: For `binary` encoding, this lists all predicates in order, defining the feature map.
- `train_files.txt`, `val_files.txt`, `test_files.txt`: Lists of problem basenames for each split.

### State Encoding

PaTS supports multiple state encoding schemes, controlled by the `--encoding_type` flag in `parse_and_encode.py`.

#### Binary Predicate Encoding (`--encoding_type binary`)

- **Representation**: A long binary vector where each element corresponds to a specific ground predicate (e.g., `(on-table b1)`, `(on b1 b2)`). `1` means true, `0` means false.
- **Size**: Scales quadratically with the number of blocks, O(n²).
- **Configuration**: This encoding is defined by a `predicate_manifest_<N>.txt` file, which lists every possible predicate in a fixed order. The `encoding_info_<N>.json` file will point to this manifest.

#### SAS+ Position Vector Encoding (`--encoding_type sas`)

- **Representation**: A compact integer vector where the _index_ represents the block and the _value_ represents its position.
  - `vector[i]` corresponds to block `b(i+1)`.
  - Value `0`: The block is on the table.
  - Value `j > 0`: The block is on top of block `bj`.
  - Value `-1`: The block is being held by the arm.
- **Example (4 blocks)**: The state "A on B, B on table, C on D, D on table" (with A=b1, B=b2, C=b3, D=b4) is encoded as `[2, 0, 4, 0]`.
- **Size**: Scales linearly with the number of blocks, O(n).
- **Configuration**: The `encoding_info_<N>.json` file will specify the type as `sas` and list the block order. No separate manifest file is needed.

### Encoding Information (`encoding_info_<N>.json`)

This JSON file is the primary source of truth for how states are encoded for a given `num_blocks`. It is read by the `BlocksWorldValidator` and other components to adapt to the current scheme.
