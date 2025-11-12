import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..pats_dataset import PaTSDataset

# Relative imports from the PaTS framework
from ..PlannableModel import PlannableModel

# Suppress some transformers warnings for cleaner output during benchmarking
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Global model and tokenizer instances to avoid reloading for every prediction call.
# This optimization is crucial for LLMs during repeated inference.
_llama_model = None
_llama_tokenizer = None


def get_llama_model_and_tokenizer(model_id: str):
    """
    Loads and caches the Llama model and tokenizer.
    Uses 4-bit quantization if CUDA is available.
    """
    global _llama_model, _llama_tokenizer
    if _llama_model is None or _llama_tokenizer is None:
        print(f"Loading Llama model '{model_id}' and tokenizer. This may take a while...")

        _llama_tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Ensure pad_token is set for generation if not present
        if _llama_tokenizer.pad_token is None:
            # Llama models usually have EOS as pad token if not explicitly set
            _llama_tokenizer.pad_token = _llama_tokenizer.eos_token
        # Set padding side for generation (important for autoregressive models)
        _llama_tokenizer.padding_side = "left"  # Llama-3-Instruct prefers left padding for generation

        _llama_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # Keep this for bfloat16 precision
            device_map="auto",  # Keep this for automatic device mapping
        )
        _llama_model.eval()  # Set to evaluation mode
        print("Llama model loaded.")
    return _llama_model, _llama_tokenizer


class LlamaWrapper(PlannableModel):
    """
    A wrapper for out-of-the-box Llama inference for PaTS.
    Performs one-shot learning by embedding an example in the prompt.
    """

    def __init__(
        self,
        model_id: str,
        domain_config: dict,
        device: torch.device,
        encoding_type: str,
        dataset_dir: Path,  # This is raw_data_dir
        processed_data_dir: Path,
        use_few_shot_example: bool = True,
    ):
        """
        Initializes the LlamaWrapper.

        :param model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        :param domain_config: Domain configuration dictionary.
        :param device: The Torch device to run the model on.
        :param encoding_type: The state encoding type ('bin' or 'sas').
        :param dataset_dir: Path to the dataset directory to load a one-shot example.
        :param processed_data_dir: Path to the processed data directory for the specific encoding.
        """
        super().__init__(Path(model_id), device)  # model_path for PlannableModel is model_id
        self.model_id = model_id
        self.encoding_type = encoding_type
        self.domain_config = domain_config
        self.state_vec_dim: int = -1  # Will be inferred from dataset
        self.example_initial_state: Optional[np.ndarray] = None
        self.processed_data_dir = processed_data_dir  # Store it
        self.example_goal_state: Optional[np.ndarray] = None
        self.example_plan_trajectory: Optional[np.ndarray] = None
        self.dataset_dir = dataset_dir  # Path to data/blocks_N/
        self.use_few_shot_example = use_few_shot_example

        # Extract domain-specific parameters from domain_config
        if self.domain_config.get("num_blocks") is not None:
            self.num_blocks = self.domain_config["num_blocks"]
            self.block_names: List[str] = [f"b{i + 1}" for i in range(self.num_blocks)]
        elif self.domain_config.get("robots") is not None:
            self.num_robots = self.domain_config["robots"]
            self.num_objects = self.domain_config["objects"]
            self.num_rooms = self.domain_config["rooms"]
        else:
            raise ValueError("domain_config must contain 'num_blocks' or 'robots/objects/rooms'.")

    def load_model(self):
        """
        Loads the Llama model and tokenizer, and prepares the one-shot example.
        """
        self.model, self.tokenizer = get_llama_model_and_tokenizer(self.model_id)

        # Infer state dimension from dataset regardless of few-shot/zero-shot
        temp_dataset = None
        try:
            # Use a dummy split file name to just load the first problem for state_dim inference
            # and to get an example problem-solution pair.
            # Using 'train_files.txt' is appropriate as examples should come from training data.
            temp_dataset = PaTSDataset(
                raw_data_dir=self.dataset_dir,
                processed_data_dir=self.processed_data_dir,
                split_file_name="train_files.txt",
                encoding_type=self.encoding_type,
            )
            self.state_vec_dim = temp_dataset.state_dim
        except Exception as e:
            print(f"Error loading PaTSDataset for state_dim inference: {e}")
            # Fallback for state_vec_dim if dataset loading fails
            self._create_dummy_example()  # This will set state_vec_dim and dummy examples

        if self.state_vec_dim <= 0:
            raise RuntimeError("Failed to infer state dimension. Please check dataset.")

        # Load one-shot example ONLY if use_few_shot_example is True
        if self.use_few_shot_example and temp_dataset:
            try:
                if len(temp_dataset) > 0:
                    # Load one example for one-shot inference from the training set.
                    # It's crucial that this example has at least 2 states (S0, S1) for the prompt.
                    for i in range(len(temp_dataset)):
                        example_item = temp_dataset[i]
                        if example_item["expert_trajectory"].shape[0] >= 2:
                            self.example_initial_state = example_item["expert_trajectory"][0]
                            self.example_goal_state = example_item["goal_state"]
                            self.example_plan_trajectory = example_item["expert_trajectory"]
                            print(
                                f"Loaded one-shot example from {example_item['id']} (trajectory length: {self.example_plan_trajectory.shape[0]})"  # type: ignore
                            )
                            break
                    if self.example_plan_trajectory is None:
                        print(
                            "Warning: No suitable one-shot example (min 2 states) found in training data. Creating dummy."
                        )
                        self._create_dummy_example()
                else:
                    print("Warning: No training data available to load one-shot example. Creating dummy.")
                    self._create_dummy_example()
            except Exception as e:
                print(f"Error loading one-shot example: {e}. Creating dummy.")
                self._create_dummy_example()
        else:  # If not using few-shot, ensure example fields are None
            self.example_initial_state = None
            self.example_goal_state = None

        print(
            f"LlamaWrapper loaded. State dimension: {self.state_vec_dim}, Encoding: {self.encoding_type}, Few-shot: {self.use_few_shot_example}"
        )

    def _create_dummy_example(self):
        """Creates a minimal dummy example if no real one can be loaded."""
        print("Creating dummy one-shot example due to data loading issues or no suitable example found.")

        if self.encoding_type == "sas":
            if hasattr(self, "num_blocks"):  # Blocksworld
                dummy_dim = self.num_blocks
                dummy_s0 = np.full(dummy_dim, 0, dtype=np.int8)  # All on table
                dummy_s1 = (
                    np.array([2] + [0] * (dummy_dim - 1), dtype=np.int8) if dummy_dim >= 1 else np.array([], dtype=np.int8)
                )  # b1 on b2, others on table
            elif hasattr(self, "num_robots"):  # Grippers
                dummy_dim = self.num_robots + self.num_objects
                dummy_s0 = np.full(dummy_dim, 1, dtype=np.int8)  # All in room1
                dummy_s1 = (
                    np.array([1] * self.num_robots + [-1] + [1] * (self.num_objects - 1), dtype=np.int8)
                    if dummy_dim >= 1
                    else np.array([], dtype=np.int8)
                )  # r1 in room1, ball1 held by r1-left
            else:
                dummy_dim = 5  # Fallback
                dummy_s0 = np.zeros(dummy_dim, dtype=np.int8)
                dummy_s1 = np.ones(dummy_dim, dtype=np.int8)
        else:  # Binary encoding
            # A rough heuristic for binary, or use a fixed small number
            dummy_dim = self.state_vec_dim if self.state_vec_dim > 0 else 20  # Use inferred dim if available, else fallback
            dummy_s0 = np.zeros(dummy_dim, dtype=np.int8)
            dummy_s1 = np.ones(dummy_dim, dtype=np.int8)
            warnings.warn(f"Creating dummy binary example with dimension {dummy_dim}. This may not be valid.")

        self.state_vec_dim = dummy_dim

        self.example_initial_state = dummy_s0
        self.example_goal_state = dummy_s1  # Arbitrary goal
        self.example_plan_trajectory = np.array([dummy_s0, dummy_s1], dtype=np.int8)

    def _vector_to_string(self, vec: np.ndarray) -> str:
        """
        Converts a numpy array vector to a string representation for the prompt.
        Example: `"[0 1 0 0 1]"` for binary, `"[2 0 0]"` for SAS+.
        """
        # Ensure integer types for consistent string representation
        return "[" + " ".join(map(str, vec.astype(int).tolist())) + "]"

    def _string_to_vector(self, text: str) -> Optional[np.ndarray]:
        """
        Parses a string from Llama's output into a numpy array vector.
        Expected format: `"[num1 num2 ... numF]"` or ` "num1 num2 ... numF"`.
        Handles potential malformed output.
        """
        try:
            # Remove brackets and commas, then split by space
            cleaned_text = text.strip().replace("[", "").replace("]", "").replace(",", " ").strip()
            parts = [p for p in cleaned_text.split() if p]  # Filter out empty strings from multiple spaces

            if len(parts) != self.state_vec_dim:
                # print(f"  Warning: Parsed vector has incorrect dimension. Expected {self.state_vec_dim}, got {len(parts)}. Raw text: '{text}'")
                return None  # Dimension mismatch

            if self.encoding_type == "bin":
                # Convert to integer and then binarize (0 or 1)
                # This handles cases where Llama might output floats (e.g., "0.9" -> 1, "0.1" -> 0)
                return np.array([int(float(p) > 0.5) for p in parts], dtype=np.int8)
            elif self.encoding_type == "sas":
                # Convert to integer, then clamp values to valid SAS+ range [-1, num_blocks]
                vector = np.array([int(float(p)) for p in parts], dtype=np.int8)
                if hasattr(self, "num_blocks"):  # Blocksworld
                    max_val = self.num_blocks
                    min_val = -1
                elif hasattr(self, "num_rooms"):  # Grippers
                    max_val = self.num_rooms
                    min_val = -(2 * self.num_robots)
                else:  # Fallback
                    max_val = 10
                    min_val = -1
                vector = np.clip(vector, min_val, max_val)
                return vector
            else:
                return None  # Should not happen, as encoding_type is checked in init

        except ValueError:
            # print(f"  Error parsing string to vector: {e}. Raw text: '{text}'")
            return None  # Parsing error (e.g., non-numeric token)
        except Exception:
            # print(f"  Unexpected error in _string_to_vector: {e}. Raw text: '{text}'")
            return None  # General unexpected error

    def predict_sequence(self, initial_state_np: np.ndarray, goal_state_np: np.ndarray, max_length: int) -> List[List[int]]:
        """
        Predicts a sequence of states (plan) using Llama in an autoregressive loop.

        :param initial_state_np: The initial state as a NumPy array.
        :param goal_state_np: The goal state as a NumPy array.
        :param max_length: Maximum number of steps in the predicted plan.
        :return: A list of state lists representing the predicted plan.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Llama model or tokenizer not loaded. Call load_model() first.")

        # Convert current problem's initial and goal states to string format
        current_state_str = self._vector_to_string(initial_state_np)
        goal_state_str = self._vector_to_string(goal_state_np)

        domain_name_for_prompt = "Blocksworld" if hasattr(self, "num_blocks") else "Grippers"

        # Base instruction for the AI
        instruction = f'You are an expert planning AI. Your task is to generate a sequence of states to solve a planning problem in {domain_name_for_prompt}. You operate on numerical state representations. Given an initial state and a goal state, predict ONLY the next state in the plan, in the exact format "[num1 num2 ... numF]". Do not output any other text or explanation.'

        # Conditionally add the example for few-shot
        example_section = ""
        if self.use_few_shot_example:
            if (
                self.example_initial_state is None
                or self.example_plan_trajectory is None
                or self.example_goal_state is None
            ):
                # This should ideally not happen if load_model was successful, but as a safeguard
                print("  Warning: Few-shot example requested but not available. Falling back to zero-shot.")
                self.use_few_shot_example = False  # Disable few-shot for this run
            else:
                example_S0_str = self._vector_to_string(self.example_initial_state)
                example_S1_str = self._vector_to_string(self.example_plan_trajectory[1])
                example_SG_str = self._vector_to_string(self.example_goal_state)
                example_section = f"""
Example:
INITIAL_STATE: {example_S0_str}
GOAL_STATE: {example_SG_str}
NEXT_STATE: {example_S1_str}
"""

        # Construct the full prompt
        prompt = f"""<s>[INST] {instruction}{example_section}
Now, generate for the following:
INITIAL_STATE: {current_state_str}
GOAL_STATE: {goal_state_str}
NEXT_STATE: [/INST]"""

        generated_plan_list_of_lists: List[List[int]] = [initial_state_np.tolist()]  # type: ignore

        for step in range(max_length - 1):  # max_length includes S0, so iterate for max_length-1 steps
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # Generate tokens
            # max_new_tokens should be enough to cover the state vector string,
            # e.g., state_vec_dim * (max digits per num + 1 for space) + 2 for brackets + buffer
            # A vector [1, 2, 3, 4] with 4 blocks in SAS+ could have max value 4 (on b4), so 1 digit.
            # Max 2 digits for -1, 0, so ~3 characters per number including space. (num_blocks * 3) + 5
            max_new_tokens_per_step = self.state_vec_dim * 3 + 5

            # Using torch.no_grad() for inference
            with torch.no_grad():
                output_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens_per_step,
                    pad_token_id=self.tokenizer.pad_token_id,  # Use pad_token_id as generated pad token
                    do_sample=False,  # Greedy decoding for deterministic planning
                    eos_token_id=self.tokenizer.eos_token_id,  # Stop generating at EOS token
                )

            # Decode the generated tokens
            # Only consider the new tokens generated after the prompt
            # output_tokens[0] is the entire sequence including input prompt
            new_tokens = output_tokens[0, inputs["input_ids"].shape[1] :]
            predicted_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Attempt to parse the predicted text into a numerical vector
            next_state_np = self._string_to_vector(predicted_text)

            if next_state_np is None:
                # If parsing fails, it's an invalid prediction, stop plan generation
                # print(f"  Llama: Failed to parse predicted state at step {step + 1}. Predicted text: '{predicted_text}'. Stopping plan generation.")
                break

            # Convert to list for the plan storage
            next_state_list = next_state_np.tolist()
            generated_plan_list_of_lists.append(next_state_list)  # type: ignore

            # Update current state string for the next iteration's prompt
            current_state_str = self._vector_to_string(next_state_np)

            # Check for stagnation (predicted state is identical to previous state)
            # This prevents infinite loops if the model gets stuck.
            if len(generated_plan_list_of_lists) >= 2 and np.array_equal(
                generated_plan_list_of_lists[-1], generated_plan_list_of_lists[-2]
            ):
                # print(f"  Llama: Stagnation detected at step {step + 1}. Stopping.")
                break

            # Check if goal is reached. This is an early stopping condition.
            if np.array_equal(next_state_np, goal_state_np):
                # print(f"  Llama: Goal reached at step {step + 1}.")
                break

        return generated_plan_list_of_lists

    @property
    def model_name(self) -> str:
        """Returns a descriptive name for the Llama model."""
        return f"PaTS_Llama_{self.encoding_type}"

    @property
    def state_dim(self) -> int:
        """Returns the feature dimension of the states the model expects/produces."""
        if self.state_vec_dim == -1:
            raise RuntimeError("State dimension not inferred. Call load_model() first.")
        return self.state_vec_dim
