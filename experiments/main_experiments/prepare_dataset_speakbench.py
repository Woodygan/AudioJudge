import pandas as pd
import json
import os
import shutil
from datasets import load_dataset
from tqdm import tqdm

import os
import json
import shutil
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
import soundfile as sf


def prepare_speakbench_dataset(
    huggingface_repo,
    output_path,
    speakbench508_json_path="datasets/speakbench508_dataset.json",
    audio_dir="speakbench_audio",
):
    """
    Prepare a JSON dataset for the pronunciation evaluation by combining data from Huggingface
    and saving all columns except 'idx' and 'instruction' as audio files.
    Also copies instruction audio from the speakbench508 dataset if instruction_id matches.

    Args:
        huggingface_repo (str): The repository name on Huggingface
        output_path (str): Path where the output JSON will be saved
        speakbench508_json_path (str): Path to the speakbench508 dataset JSON file
        audio_dir (str): Directory to save the audio files
    """
    # Create audio directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Created audio directory: {audio_dir}")

    # Load the speakbench508 dataset for instruction audio paths
    print(f"Loading speakbench508 dataset from: {speakbench508_json_path}")
    with open(speakbench508_json_path, "r") as f:
        speakbench508_data = json.load(f)
    speakbench508_df = pd.DataFrame(speakbench508_data)

    # Create a dictionary mapping instruction_id to instruction_path for faster lookup
    instruction_id_to_path = {}
    for _, row in speakbench508_df.iterrows():
        if "instruction_id" in row and "instruction_path" in row:
            instruction_id_to_path[row["instruction_id"]] = row["instruction_path"]

    print(
        f"Loaded {len(instruction_id_to_path)} instruction audio paths from speakbench508 dataset"
    )

    print(f"Loading dataset from Huggingface: {huggingface_repo}")
    # Load the raw dataset from Huggingface
    dataset = load_dataset(huggingface_repo, split="train")

    # Convert to pandas DataFrame for easier manipulation
    hf_df = pd.DataFrame(dataset)

    # Initialize empty list to store our prepared data
    prepared_data = []

    print("Processing datasets and matching entries...")

    for _, row in tqdm(hf_df.iterrows(), total=len(hf_df)):
        # Skip rows with index 0
        index = row["idx"]
        if index == 0:
            print(f"Skipping index 0")
            continue

        # Get instruction text
        instruction_text = row["instruction"]

        instruction_id = index

        # Create a directory for this datapoint
        datapoint_dir = os.path.join(audio_dir, str(index))
        os.makedirs(datapoint_dir, exist_ok=True)

        # Prepare a dictionary to store paths to all saved audio files
        data_point = {
            "index": index,
            "instruction_text": instruction_text,
        }

        # Copy instruction audio from speakbench508 if instruction_id matches
        if instruction_id and instruction_id in instruction_id_to_path:
            source_instruction_path = instruction_id_to_path[instruction_id]
            if os.path.exists(source_instruction_path):
                instruction_dest_path = os.path.join(datapoint_dir, "instruction.wav")
                try:
                    shutil.copy(source_instruction_path, instruction_dest_path)
                    data_point["instruction_path"] = instruction_dest_path
                    # print(f"Copied instruction audio for index {index}, instruction_id {instruction_id}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to copy instruction audio for index {index}, instruction_id {instruction_id}: {str(e)}"
                    )
            else:
                raise FileNotFoundError(
                    f"Instruction audio file not found for index {index}, instruction_id {instruction_id}: {source_instruction_path}"
                )

        # Process all columns except 'idx' and 'instruction'
        for column_name, column_value in row.items():
            # Skip the excluded columns
            if column_name in ["idx", "instruction"]:
                continue

            # File path for this column
            file_path = os.path.join(datapoint_dir, f"{column_name}.wav")

            try:
                # Save the audio data based on its format
                if (
                    isinstance(column_value, dict)
                    and "array" in column_value
                    and "sampling_rate" in column_value
                ):
                    # HuggingFace audio format with array and sampling rate
                    sf.write(
                        file_path, column_value["array"], column_value["sampling_rate"]
                    )
                elif isinstance(column_value, str) and os.path.exists(column_value):
                    # Path to an existing file
                    shutil.copy(column_value, file_path)
                elif isinstance(column_value, bytes):
                    # Raw bytes
                    with open(file_path, "wb") as f:
                        f.write(column_value)
                else:
                    # Skip non-audio columns
                    print(
                        f"Skipping column '{column_name}' for index {index} (not an audio format)"
                    )
                    continue

                # Add the path to the data point
                data_point[column_name] = file_path

            except Exception as e:
                print(
                    f"Error saving audio for column '{column_name}', index {index}: {str(e)}"
                )
                continue

        # Only add the data point if it has at least one audio file
        if len(data_point) > 2:  # More than just index and instruction_text
            prepared_data.append(data_point)

    print(f"Created {len(prepared_data)} data points. Saving to {output_path}")
    # Save the prepared data as JSON
    with open(output_path, "w") as f:
        json.dump(prepared_data, f, indent=2)

    print("Dataset preparation complete!")
    return prepared_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset for audio evaluation")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="name of the dataset to load"
    )
    args = parser.parse_args()
    audio_root = "audio_data"
    os.makedirs(audio_root, exist_ok=True)
    dataset_dir = "datasets"
    os.makedirs(dataset_dir, exist_ok=True)
    audio_dir = os.path.join(audio_root, f"{args.dataset_name}_audio")
    os.makedirs(audio_dir, exist_ok=True)
    output_path = os.path.join(dataset_dir, f"{args.dataset_name}_dataset.json")

    if args.dataset_name == "speakbench":
        huggingface_repo = "potsawee/speakbench-v1-all-outputs"
        prepared_data = prepare_speakbench_dataset(
            huggingface_repo,
            output_path,
            speakbench508_json_path="datasets/speakbench508_dataset.json",
            audio_dir=audio_dir,
        )
        print(f"Saved {len(prepared_data)} data points to {output_path}")
        print(f"Audio files saved in {audio_dir} directory")
