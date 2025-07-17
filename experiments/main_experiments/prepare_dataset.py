import pandas as pd
import json
import os
import shutil
from datasets import load_dataset
from tqdm import tqdm

def prepare_pronunciation_dataset(huggingface_repo, csv_path, output_path, 
                                  few_shot_examples_path=None, audio_dir="pronunciation_audio"):
    """
    Prepare a JSON dataset for the pronunciation evaluation by combining:
    1. A Huggingface dataset with audio features
    2. A CSV file with human labels
    
    Optionally filters out examples that are being used as few-shot examples.
    
    Args:
        huggingface_repo (str): The repository name on Huggingface
        csv_path (str): Path to the CSV file with human labels
        output_path (str): Path where the output JSON will be saved
        few_shot_examples_path (str): Path to JSON file containing few-shot examples to exclude
        audio_dir (str): Directory to save the audio files
    """
    # Create audio directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Created audio directory: {audio_dir}")
    print(f"Loading dataset from Huggingface: {huggingface_repo}")
    
    # Load the raw dataset from Huggingface
    dataset = load_dataset(huggingface_repo, split="train")
    
    # Convert to pandas DataFrame for easier manipulation
    hf_df = pd.DataFrame(dataset)
    
    print(f"Loading human labels from: {csv_path}")
    # Load the human labels CSV
    labels_df = pd.read_csv(csv_path)
    
    # Load few-shot examples if provided to exclude them from the dataset
    few_shot_examples = set()
    if few_shot_examples_path and os.path.exists(few_shot_examples_path):
        print(f"Loading few-shot examples from: {few_shot_examples_path}")
        with open(few_shot_examples_path, 'r') as f:
            few_shot_data = json.load(f)
        examples = few_shot_data.get("pronunciation", [])
        for example in examples:
            if isinstance(example, dict) and 'word' in example and 'region' in example:
                # Use OED if available, otherwise None
                oed = example.get('OED', None)
                few_shot_examples.add((example['word'], example['region'], oed))
        
        print(f"Found {len(few_shot_examples)} few-shot examples to exclude")
    
    # Initialize empty list to store our prepared data
    prepared_data = []
    
    print("Processing datasets and matching entries...")
    filtered_idx = 1
        
    # Loop through labels and find matching entries in the Huggingface dataset
    for idx, label_row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        word = label_row['word']
        region = label_row['region']
        oed = label_row['OED']
        label = label_row['label']
        
        # Skip entries with "Bad" label
        if label == "Bad":
            continue
        
        # Skip if this is a few-shot example
        if (word, region, oed) in few_shot_examples or (word, region, None) in few_shot_examples:
            print(f"Skipping few-shot example: {word}, {region}")
            continue
        
        # Find matching entry in Huggingface dataset
        matching_entries = hf_df[(hf_df['word'] == word) & (hf_df['region'] == region)]
        
        if len(matching_entries) == 0:
            print(f"Warning: No matching entry found for {word}, {region}")
            continue
        
        # Use the first matching entry
        match_entry = matching_entries.iloc[0]
        
        # Get audio data from the dataset
        audio1_data = match_entry.get('audio', None)
        audio2_data = match_entry.get('GPT4o_pronunciation', None)
        
        # Skip if either audio is missing
        if audio1_data is None or audio2_data is None:
            print(f"Warning: Missing audio for {word}, {region}")
            continue
            
        datapoint_dir = os.path.join(audio_dir, str(filtered_idx))
        os.makedirs(datapoint_dir, exist_ok=True)
        
        # Create unique filenames for the audio files based on index
        audio1_filename = f"original_{word}_{region}.wav"
        audio2_filename = f"gpt4o_{word}_{region}.wav"
        
        # Full paths to save the audio files
        audio1_path = os.path.join(datapoint_dir, audio1_filename)
        audio2_path = os.path.join(datapoint_dir, audio2_filename)
        
        # Save audio files
        try:
            # Save audio1
            # For HuggingFace datasets, audio might be returned in different formats
            # If it's a dictionary with 'array' and 'sampling_rate'
            if isinstance(audio1_data, dict) and 'array' in audio1_data and 'sampling_rate' in audio1_data:
                import soundfile as sf
                sf.write(audio1_path, audio1_data['array'], audio1_data['sampling_rate'])
            # If it's a path
            elif isinstance(audio1_data, str) and os.path.exists(audio1_data):
                shutil.copy(audio1_data, audio1_path)
            # If it's bytes
            elif isinstance(audio1_data, bytes):
                with open(audio1_path, 'wb') as f:
                    f.write(audio1_data)
            else:
                print(f"Warning: Unsupported audio format for {word}, {region}")
                continue
                
            # Save audio2 (same approach)
            if isinstance(audio2_data, dict) and 'array' in audio2_data and 'sampling_rate' in audio2_data:
                import soundfile as sf
                sf.write(audio2_path, audio2_data['array'], audio2_data['sampling_rate'])
            elif isinstance(audio2_data, str) and os.path.exists(audio2_data):
                shutil.copy(audio2_data, audio2_path)
            elif isinstance(audio2_data, bytes):
                with open(audio2_path, 'wb') as f:
                    f.write(audio2_data)
            else:
                print(f"Warning: Unsupported audio format for {word}, {region}")
                continue
                
        except Exception as e:
            print(f"Error saving audio for {word}, {region}: {str(e)}")
            continue
        
        # Convert label to boolean match value
        match = (label == "Same")
        
        # Create data point
        data_point = {
            "index": filtered_idx,
            "word": word,
            "region": region,
            "IPAs": match_entry.get('IPAs'),  # Use get() to handle missing columns
            "OED": oed,
            "gpt4o_reasoning": match_entry.get('gpt4o_reasoning'),
            "gpt4o_correct": str(match_entry.get('gpt4o_correct')).lower(),
            "audio1_path": audio1_path,  # Path to saved audio1
            "audio2_path": audio2_path,  # Path to saved audio2
            "match": str(match).lower(),  # Ground truth from human labels
            "transcript1": word,  # Add transcript (the word itself)
            "transcript2": word   # Both pronunciations are of the same word
        }
        
        prepared_data.append(data_point)
        filtered_idx += 1 
    
    print(f"Created {len(prepared_data)} data points. Saving to {output_path}")
    # Save the prepared data as JSON
    with open(output_path, 'w') as f:
        json.dump(prepared_data, f, indent=2)
    
    print("Dataset preparation complete!")
    return prepared_data

def prepare_speed_dataset(huggingface_repo, output_path, 
                                  few_shot_examples_path=None, audio_dir="speed_audio"):
    """
    Prepare a JSON dataset for the pronunciation evaluation by combining:
    1. A Huggingface dataset with audio features
    2. A CSV file with human labels
    
    Optionally filters out examples that are being used as few-shot examples.
    
    Args:
        huggingface_repo (str): The repository name on Huggingface
        csv_path (str): Path to the CSV file with human labels
        output_path (str): Path where the output JSON will be saved
        few_shot_examples_path (str): Path to JSON file containing few-shot examples to exclude
        audio_dir (str): Directory to save the audio files
    """
    # Create audio directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Created audio directory: {audio_dir}")
    print(f"Loading dataset from Huggingface: {huggingface_repo}")
    
    # Load the raw dataset from Huggingface
    dataset = load_dataset(huggingface_repo, split="train")
    
    # Convert to pandas DataFrame for easier manipulation
    hf_df = pd.DataFrame(dataset)
    
    
    # Load few-shot examples if provided to exclude them from the dataset
    few_shot_examples = set()
    if few_shot_examples_path and os.path.exists(few_shot_examples_path):
        print(f"Loading few-shot examples from: {few_shot_examples_path}")
        with open(few_shot_examples_path, 'r') as f:
            few_shot_data = json.load(f)
        examples = few_shot_data.get("speed", [])
        for example in examples:
            if isinstance(example, dict) and 'id_a' in example and 'id_b' in example:
                few_shot_examples.add((example['id_a'], example['id_b']))
        
        print(f"Found {len(few_shot_examples)} few-shot examples to exclude")
    
    # Initialize empty list to store our prepared data
    prepared_data = []
    
    print("Processing datasets and matching entries...")
        
    for idx, label_row in tqdm(hf_df.iterrows(), total=len(hf_df)):
        id_a = label_row['id_a']
        id_b = label_row['id_b']
        label = label_row['label']
        if label == "a":
            label = "1"
        elif label == "b":
            label = "2"
        transcript1 = label_row['text_normalized_a']
        transcript2 = label_row['text_normalized_b']
        
        # Skip if this is a few-shot example
        if (id_a, id_b) in few_shot_examples:
            print(f"Skipping few-shot example: {id_a}, {id_b}")
            continue
        
        
        # Get audio data from the dataset
        audio1_data = label_row['audio_a']
        audio2_data = label_row['audio_b']
        datapoint_dir = os.path.join(audio_dir, str(idx))
        os.makedirs(datapoint_dir, exist_ok=True)
        
        # Create unique filenames for the audio files based on index
        audio1_filename = f"audio_a.wav"
        audio2_filename = f"audio_b.wav"
        
        # Full paths to save the audio files
        audio1_path = os.path.join(datapoint_dir, audio1_filename)
        audio2_path = os.path.join(datapoint_dir, audio2_filename)
        
        # Save audio files
        try:
            # Save audio1
            # For HuggingFace datasets, audio might be returned in different formats
            # If it's a dictionary with 'array' and 'sampling_rate'
            if isinstance(audio1_data, dict) and 'array' in audio1_data and 'sampling_rate' in audio1_data:
                import soundfile as sf
                sf.write(audio1_path, audio1_data['array'], audio1_data['sampling_rate'])
            # If it's a path
            elif isinstance(audio1_data, str) and os.path.exists(audio1_data):
                shutil.copy(audio1_data, audio1_path)
            # If it's bytes
            elif isinstance(audio1_data, bytes):
                with open(audio1_path, 'wb') as f:
                    f.write(audio1_data)
            else:
                print(f"Warning: Unsupported audio format for {id_a}, {id_b}")
                continue
                
            # Save audio2 (same approach)
            if isinstance(audio2_data, dict) and 'array' in audio2_data and 'sampling_rate' in audio2_data:
                import soundfile as sf
                sf.write(audio2_path, audio2_data['array'], audio2_data['sampling_rate'])
            elif isinstance(audio2_data, str) and os.path.exists(audio2_data):
                shutil.copy(audio2_data, audio2_path)
            elif isinstance(audio2_data, bytes):
                with open(audio2_path, 'wb') as f:
                    f.write(audio2_data)
            else:
                print(f"Warning: Unsupported audio format for {id_a}, {id_b}")
                continue
                
        except Exception as e:
            print(f"Error saving audio for {id_a}, {id_b}: {str(e)}")
            continue
        
        # Create data point
        data_point = {
            "index": idx,
            "audio1_path": audio1_path,  # Path to saved audio1
            "audio2_path": audio2_path,  # Path to saved audio2
            "label": str(label).lower(), 
            "transcript1": transcript1,  
            "transcript2": transcript2,
            "meta_a": label_row['meta_a'],
            "meta_b": label_row['meta_b'],
            "speaker_a": label_row['speaker_a'],
            "speaker_b": label_row['speaker_b'],
            "id_a": id_a,
            "id_b": id_b,
        }
        
        prepared_data.append(data_point)
    
    print(f"Created {len(prepared_data)} data points. Saving to {output_path}")
    # Save the prepared data as JSON
    with open(output_path, 'w') as f:
        json.dump(prepared_data, f, indent=2)
    
    print("Dataset preparation complete!")
    return prepared_data
def prepare_speaker_dataset(huggingface_repo, output_path, 
                                  few_shot_examples_path=None, audio_dir="speaker_audio"):
    """
    Prepare a JSON dataset for the pronunciation evaluation by combining:
    1. A Huggingface dataset with audio features
    2. A CSV file with human labels
    
    Optionally filters out examples that are being used as few-shot examples.
    
    Args:
        huggingface_repo (str): The repository name on Huggingface
        csv_path (str): Path to the CSV file with human labels
        output_path (str): Path where the output JSON will be saved
        few_shot_examples_path (str): Path to JSON file containing few-shot examples to exclude
        audio_dir (str): Directory to save the audio files
    """
    # Create audio directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Created audio directory: {audio_dir}")
    print(f"Loading dataset from Huggingface: {huggingface_repo}")
    
    # Load the raw dataset from Huggingface
    dataset = load_dataset(huggingface_repo, split="train")
    
    # Convert to pandas DataFrame for easier manipulation
    hf_df = pd.DataFrame(dataset)
    
    
    # Load few-shot examples if provided to exclude them from the dataset
    few_shot_examples = set()
    if few_shot_examples_path and os.path.exists(few_shot_examples_path):
        print(f"Loading few-shot examples from: {few_shot_examples_path}")
        with open(few_shot_examples_path, 'r') as f:
            few_shot_data = json.load(f)
        examples = few_shot_data.get("speaker", [])
        for example in examples:
            if isinstance(example, dict) and 'id_a' in example and 'id_b' in example:
                few_shot_examples.add((example['id_a'], example['id_b']))
        
        print(f"Found {len(few_shot_examples)} few-shot examples to exclude")
    
    # Initialize empty list to store our prepared data
    prepared_data = []
    
    print("Processing datasets and matching entries...")
        
    for idx, label_row in tqdm(hf_df.iterrows(), total=len(hf_df)):
        id_a = label_row['id_a']
        id_b = label_row['id_b']
        match = label_row['label'] == "same"
        transcript1 = label_row['text_normalized_a']
        transcript2 = label_row['text_normalized_b']
        
        # Skip if this is a few-shot example
        if (id_a, id_b) in few_shot_examples:
            print(f"Skipping few-shot example: {id_a}, {id_b}")
            continue
        
        
        # Get audio data from the dataset
        audio1_data = label_row['audio_a']
        audio2_data = label_row['audio_b']
        datapoint_dir = os.path.join(audio_dir, str(idx))
        os.makedirs(datapoint_dir, exist_ok=True)
        
        # Create unique filenames for the audio files based on index
        audio1_filename = f"audio_a.wav"
        audio2_filename = f"audio_b.wav"
        
        # Full paths to save the audio files
        audio1_path = os.path.join(datapoint_dir, audio1_filename)
        audio2_path = os.path.join(datapoint_dir, audio2_filename)
        
        # Save audio files
        try:
            # Save audio1
            # For HuggingFace datasets, audio might be returned in different formats
            # If it's a dictionary with 'array' and 'sampling_rate'
            if isinstance(audio1_data, dict) and 'array' in audio1_data and 'sampling_rate' in audio1_data:
                import soundfile as sf
                sf.write(audio1_path, audio1_data['array'], audio1_data['sampling_rate'])
            # If it's a path
            elif isinstance(audio1_data, str) and os.path.exists(audio1_data):
                shutil.copy(audio1_data, audio1_path)
            # If it's bytes
            elif isinstance(audio1_data, bytes):
                with open(audio1_path, 'wb') as f:
                    f.write(audio1_data)
            else:
                print(f"Warning: Unsupported audio format for {id_a}, {id_b}")
                continue
                
            # Save audio2 (same approach)
            if isinstance(audio2_data, dict) and 'array' in audio2_data and 'sampling_rate' in audio2_data:
                import soundfile as sf
                sf.write(audio2_path, audio2_data['array'], audio2_data['sampling_rate'])
            elif isinstance(audio2_data, str) and os.path.exists(audio2_data):
                shutil.copy(audio2_data, audio2_path)
            elif isinstance(audio2_data, bytes):
                with open(audio2_path, 'wb') as f:
                    f.write(audio2_data)
            else:
                print(f"Warning: Unsupported audio format for {id_a}, {id_b}")
                continue
                
        except Exception as e:
            print(f"Error saving audio for {id_a}, {id_b}: {str(e)}")
            continue
        
        # Create data point
        data_point = {
            "index": idx,
            "audio1_path": audio1_path,  # Path to saved audio1
            "audio2_path": audio2_path,  # Path to saved audio2
            "match": str(match).lower(), 
            "transcript1": transcript1,  
            "transcript2": transcript2,
            "meta_a": label_row['meta_a'],
            "meta_b": label_row['meta_b'],
            "speaker_a": label_row['speaker_a'],
            "speaker_b": label_row['speaker_b'],
            "id_a": id_a,
            "id_b": id_b,
        }
        
        prepared_data.append(data_point)
    
    print(f"Created {len(prepared_data)} data points. Saving to {output_path}")
    # Save the prepared data as JSON
    with open(output_path, 'w') as f:
        json.dump(prepared_data, f, indent=2)
    
    print("Dataset preparation complete!")
    return prepared_data

def prepare_tmhintq_dataset(huggingface_repo, output_path, 
                                  few_shot_examples_path=None, audio_dir="tmhintq_audio"):
    """
    Prepare a JSON dataset for the pronunciation evaluation by combining:
    1. A Huggingface dataset with audio features
    2. A CSV file with human labels
    
    Optionally filters out examples that are being used as few-shot examples.
    
    Args:
        huggingface_repo (str): The repository name on Huggingface
        csv_path (str): Path to the CSV file with human labels
        output_path (str): Path where the output JSON will be saved
        few_shot_examples_path (str): Path to JSON file containing few-shot examples to exclude
        audio_dir (str): Directory to save the audio files
    """
    # Create audio directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Created audio directory: {audio_dir}")
    print(f"Loading dataset from Huggingface: {huggingface_repo}")
    
    # Load the raw dataset from Huggingface
    dataset = load_dataset(huggingface_repo, split="train")
    
    # Convert to pandas DataFrame for easier manipulation
    hf_df = pd.DataFrame(dataset)
    
    
    # Load few-shot examples if provided to exclude them from the dataset
    few_shot_examples = set()
    if few_shot_examples_path and os.path.exists(few_shot_examples_path):
        print(f"Loading few-shot examples from: {few_shot_examples_path}")
        with open(few_shot_examples_path, 'r') as f:
            few_shot_data = json.load(f)
        examples = few_shot_data.get("tmhintq", [])
        for example in examples:

            transcript1 = example.get('transcript1', '')
            transcript2 = example.get('transcript2', '')
            system_a = example.get('system_a', '')
            system_b = example.get('system_b', '')
            few_shot_examples.add((transcript1, transcript2, system_a, system_b))
    
        
        print(f"Found {len(few_shot_examples)} few-shot examples to exclude")
    
    # Initialize empty list to store our prepared data
    prepared_data = []
    
    print("Processing datasets and matching entries...")
        
    for idx, label_row in tqdm(hf_df.iterrows(), total=len(hf_df)):
        label = label_row['label']
        if label == "a":
            label = "1"
        elif label == "b":
            label = "2"
        transcript1 = label_row['text_a']
        transcript2 = label_row['text_b']
        human_quality_a = label_row['human_quality_a']
        human_quality_b = label_row['human_quality_b']
        system_a = label_row['system_a']
        system_b = label_row['system_b']
        
        # Skip if this is a few-shot example
        if (transcript1, transcript2, system_a, system_b) in few_shot_examples:
            print(f"Skipping few-shot example: {transcript1}, {transcript2}")
            continue
        
        
        # Get audio data from the dataset
        audio1_data = label_row['audio_a']
        audio2_data = label_row['audio_b']
        datapoint_dir = os.path.join(audio_dir, str(idx))
        os.makedirs(datapoint_dir, exist_ok=True)
        
        # Create unique filenames for the audio files based on index
        audio1_filename = f"audio_a.wav"
        audio2_filename = f"audio_b.wav"
        
        # Full paths to save the audio files
        audio1_path = os.path.join(datapoint_dir, audio1_filename)
        audio2_path = os.path.join(datapoint_dir, audio2_filename)
        
        # Save audio files
        try:
            # Save audio1
            # For HuggingFace datasets, audio might be returned in different formats
            # If it's a dictionary with 'array' and 'sampling_rate'
            if isinstance(audio1_data, dict) and 'array' in audio1_data and 'sampling_rate' in audio1_data:
                import soundfile as sf
                sf.write(audio1_path, audio1_data['array'], audio1_data['sampling_rate'])
            # If it's a path
            elif isinstance(audio1_data, str) and os.path.exists(audio1_data):
                shutil.copy(audio1_data, audio1_path)
            # If it's bytes
            elif isinstance(audio1_data, bytes):
                with open(audio1_path, 'wb') as f:
                    f.write(audio1_data)
            else:
                print(f"Warning: Unsupported audio format for {idx}")

                continue
                
            # Save audio2 (same approach)
            if isinstance(audio2_data, dict) and 'array' in audio2_data and 'sampling_rate' in audio2_data:
                import soundfile as sf
                sf.write(audio2_path, audio2_data['array'], audio2_data['sampling_rate'])
            elif isinstance(audio2_data, str) and os.path.exists(audio2_data):
                shutil.copy(audio2_data, audio2_path)
            elif isinstance(audio2_data, bytes):
                with open(audio2_path, 'wb') as f:
                    f.write(audio2_data)
            else:
                print(f"Warning: Unsupported audio format for {idx}")

                continue
                
        except Exception as e:
            print(f"Error saving audio for {idx}: {str(e)}")
            continue
        
        # Create data point
        data_point = {
            "index": idx,
            "audio1_path": audio1_path,  # Path to saved audio1
            "audio2_path": audio2_path,  # Path to saved audio2
            "label": str(label).lower(), 
            "transcript1": transcript1,  
            "transcript2": transcript2,
            "human_quality_a": human_quality_a,
            "human_quality_b": human_quality_b,
            "system_a": system_a,
            "system_b": system_b,
        }
        
        prepared_data.append(data_point)
    
    print(f"Created {len(prepared_data)} data points. Saving to {output_path}")
    # Save the prepared data as JSON
    with open(output_path, 'w') as f:
        json.dump(prepared_data, f, indent=2)
    
    print("Dataset preparation complete!")
    return prepared_data
def prepare_somos_dataset(huggingface_repo, output_path, 
                                  few_shot_examples_path=None, audio_dir="somos_audio"):
    """
    Prepare a JSON dataset for the pronunciation evaluation by combining:
    1. A Huggingface dataset with audio features
    2. A CSV file with human labels
    
    Optionally filters out examples that are being used as few-shot examples.
    
    Args:
        huggingface_repo (str): The repository name on Huggingface
        csv_path (str): Path to the CSV file with human labels
        output_path (str): Path where the output JSON will be saved
        few_shot_examples_path (str): Path to JSON file containing few-shot examples to exclude
        audio_dir (str): Directory to save the audio files
    """
    # Create audio directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Created audio directory: {audio_dir}")
    print(f"Loading dataset from Huggingface: {huggingface_repo}")
    
    # Load the raw dataset from Huggingface
    dataset = load_dataset(huggingface_repo, split="train")
    
    # Convert to pandas DataFrame for easier manipulation
    hf_df = pd.DataFrame(dataset)
    
    
    # Load few-shot examples if provided to exclude them from the dataset
    few_shot_examples = set()
    if few_shot_examples_path and os.path.exists(few_shot_examples_path):
        print(f"Loading few-shot examples from: {few_shot_examples_path}")
        with open(few_shot_examples_path, 'r') as f:
            few_shot_data = json.load(f)
        examples = few_shot_data.get("somos", [])
        for example in examples:

            uttId_a = example.get('uttId_a', '')
            uttId_b = example.get('uttId_b', '')
            few_shot_examples.add((uttId_a, uttId_b))
    
        
        print(f"Found {len(few_shot_examples)} few-shot examples to exclude")
    
    # Initialize empty list to store our prepared data
    prepared_data = []
    
    print("Processing datasets and matching entries...")
        
    for idx, label_row in tqdm(hf_df.iterrows(), total=len(hf_df)):
        label = label_row['label']
        if label == "a":
            label = "1"
        elif label == "b":
            label = "2"
        transcript1 = label_row['text_a']
        transcript2 = label_row['text_b']
        mos_a = label_row['mos_a']
        mos_b = label_row['mos_b']
        uttId_a = label_row['uttId_a']
        uttId_b = label_row['uttId_b']
        if (uttId_a, uttId_b) in few_shot_examples:
            print(f"Skipping few-shot example: {uttId_a}, {uttId_b}")
            continue
        
        
        # Get audio data from the dataset
        audio1_data = label_row['audio_a']
        audio2_data = label_row['audio_b']
        datapoint_dir = os.path.join(audio_dir, str(idx))
        os.makedirs(datapoint_dir, exist_ok=True)
        
        # Create unique filenames for the audio files based on index
        audio1_filename = f"audio_a.wav"
        audio2_filename = f"audio_b.wav"
        
        # Full paths to save the audio files
        audio1_path = os.path.join(datapoint_dir, audio1_filename)
        audio2_path = os.path.join(datapoint_dir, audio2_filename)
        
        # Save audio files
        try:
            # Save audio1
            # For HuggingFace datasets, audio might be returned in different formats
            # If it's a dictionary with 'array' and 'sampling_rate'
            if isinstance(audio1_data, dict) and 'array' in audio1_data and 'sampling_rate' in audio1_data:
                import soundfile as sf
                sf.write(audio1_path, audio1_data['array'], audio1_data['sampling_rate'])
            # If it's a path
            elif isinstance(audio1_data, str) and os.path.exists(audio1_data):
                shutil.copy(audio1_data, audio1_path)
            # If it's bytes
            elif isinstance(audio1_data, bytes):
                with open(audio1_path, 'wb') as f:
                    f.write(audio1_data)
            else:
                print(f"Warning: Unsupported audio format for {idx}")

                continue
                
            # Save audio2 (same approach)
            if isinstance(audio2_data, dict) and 'array' in audio2_data and 'sampling_rate' in audio2_data:
                import soundfile as sf
                sf.write(audio2_path, audio2_data['array'], audio2_data['sampling_rate'])
            elif isinstance(audio2_data, str) and os.path.exists(audio2_data):
                shutil.copy(audio2_data, audio2_path)
            elif isinstance(audio2_data, bytes):
                with open(audio2_path, 'wb') as f:
                    f.write(audio2_data)
            else:
                print(f"Warning: Unsupported audio format for {idx}")

                continue
                
        except Exception as e:
            print(f"Error saving audio for {idx}: {str(e)}")
            continue
        
        # Create data point
        data_point = {
            "index": idx,
            "audio1_path": audio1_path,  # Path to saved audio1
            "audio2_path": audio2_path,  # Path to saved audio2
            "label": str(label).lower(), 
            "transcript1": transcript1,  
            "transcript2": transcript2,
            "mos_a": mos_a,
            "mos_b": mos_b,
            "uttId_a": uttId_a,
            "uttId_b": uttId_b,
        }
        
        prepared_data.append(data_point)
    
    print(f"Created {len(prepared_data)} data points. Saving to {output_path}")
    # Save the prepared data as JSON
    with open(output_path, 'w') as f:
        json.dump(prepared_data, f, indent=2)
    
    print("Dataset preparation complete!")
    return prepared_data
def prepare_thaimos_dataset(huggingface_repo, output_path, 
                                  few_shot_examples_path=None, audio_dir="thaimos_audio"):
    """
    Prepare a JSON dataset for the pronunciation evaluation by combining:
    1. A Huggingface dataset with audio features
    2. A CSV file with human labels
    
    Optionally filters out examples that are being used as few-shot examples.
    
    Args:
        huggingface_repo (str): The repository name on Huggingface
        csv_path (str): Path to the CSV file with human labels
        output_path (str): Path where the output JSON will be saved
        few_shot_examples_path (str): Path to JSON file containing few-shot examples to exclude
        audio_dir (str): Directory to save the audio files
    """
    # Create audio directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Created audio directory: {audio_dir}")
    print(f"Loading dataset from Huggingface: {huggingface_repo}")
    
    # Load the raw dataset from Huggingface
    dataset = load_dataset(huggingface_repo, split="train")
    
    # Convert to pandas DataFrame for easier manipulation
    hf_df = pd.DataFrame(dataset)
    
    
    # Load few-shot examples if provided to exclude them from the dataset
    few_shot_examples = set()
    if few_shot_examples_path and os.path.exists(few_shot_examples_path):
        print(f"Loading few-shot examples from: {few_shot_examples_path}")
        with open(few_shot_examples_path, 'r') as f:
            few_shot_data = json.load(f)
        examples = few_shot_data.get("thaimos", [])
        for example in examples:

            datawow_id_a = example.get('datawow_id_a', '')
            datawow_id_b = example.get('datawow_id_b', '')
            few_shot_examples.add((datawow_id_a, datawow_id_b))
    
        
        print(f"Found {len(few_shot_examples)} few-shot examples to exclude")
    
    # Initialize empty list to store our prepared data
    prepared_data = []
    
    print("Processing datasets and matching entries...")
        
    for idx, label_row in tqdm(hf_df.iterrows(), total=len(hf_df)):
        label = label_row['label']
        if label == "a":
            label = "1"
        elif label == "b":
            label = "2"
        transcript1 = label_row['text_a']
        transcript2 = label_row['text_b']
        datawow_id_a = label_row['datawow_id_a']
        datawow_id_b = label_row['datawow_id_b']
        transcript1 = label_row['text_a']
        transcript2 = label_row['text_b']
        pronunciation_a = label_row['pronunciation_a']
        pronunciation_b = label_row['pronunciation_b']
        sound_a = label_row['sound_a']
        sound_b = label_row['sound_b']
        rhythm_a = label_row['rhythm_a']
        rhythm_b = label_row['rhythm_b']
        system_a = label_row['system_a']
        system_b = label_row['system_b']

        if (datawow_id_a, datawow_id_b) in few_shot_examples:
            print(f"Skipping few-shot example: {datawow_id_a}, {datawow_id_b}")
            continue
        
        
        # Get audio data from the dataset
        audio1_data = label_row['audio_a']
        audio2_data = label_row['audio_b']
        datapoint_dir = os.path.join(audio_dir, str(idx))
        os.makedirs(datapoint_dir, exist_ok=True)
        
        # Create unique filenames for the audio files based on index
        audio1_filename = f"audio_a.wav"
        audio2_filename = f"audio_b.wav"
        
        # Full paths to save the audio files
        audio1_path = os.path.join(datapoint_dir, audio1_filename)
        audio2_path = os.path.join(datapoint_dir, audio2_filename)
        
        # Save audio files
        try:
            # Save audio1
            # For HuggingFace datasets, audio might be returned in different formats
            # If it's a dictionary with 'array' and 'sampling_rate'
            if isinstance(audio1_data, dict) and 'array' in audio1_data and 'sampling_rate' in audio1_data:
                import soundfile as sf
                sf.write(audio1_path, audio1_data['array'], audio1_data['sampling_rate'])
            # If it's a path
            elif isinstance(audio1_data, str) and os.path.exists(audio1_data):
                shutil.copy(audio1_data, audio1_path)
            # If it's bytes
            elif isinstance(audio1_data, bytes):
                with open(audio1_path, 'wb') as f:
                    f.write(audio1_data)
            else:
                print(f"Warning: Unsupported audio format for {idx}")

                continue
                
            # Save audio2 (same approach)
            if isinstance(audio2_data, dict) and 'array' in audio2_data and 'sampling_rate' in audio2_data:
                import soundfile as sf
                sf.write(audio2_path, audio2_data['array'], audio2_data['sampling_rate'])
            elif isinstance(audio2_data, str) and os.path.exists(audio2_data):
                shutil.copy(audio2_data, audio2_path)
            elif isinstance(audio2_data, bytes):
                with open(audio2_path, 'wb') as f:
                    f.write(audio2_data)
            else:
                print(f"Warning: Unsupported audio format for {idx}")

                continue
                
        except Exception as e:
            print(f"Error saving audio for {idx}: {str(e)}")
            continue
        
        # Create data point
        data_point = {
            "index": idx,
            "audio1_path": audio1_path,  # Path to saved audio1
            "audio2_path": audio2_path,  # Path to saved audio2
            "label": str(label).lower(), 
            "transcript1": transcript1,  
            "transcript2": transcript2,
            "pronunciation_a": pronunciation_a,
            "pronunciation_b": pronunciation_b,
            "sound_a": sound_a,
            "sound_b": sound_b,
            "rhythm_a": rhythm_a,
            "rhythm_b": rhythm_b,
            "system_a": system_a,
            "system_b": system_b,
            "datawow_id_a": datawow_id_a,
            "datawow_id_b": datawow_id_b,
        }
        
        prepared_data.append(data_point)
    
    print(f"Created {len(prepared_data)} data points. Saving to {output_path}")
    # Save the prepared data as JSON
    with open(output_path, 'w') as f:
        json.dump(prepared_data, f, indent=2)
    
    print("Dataset preparation complete!")
    return prepared_data

def prepare_speakbench_dataset(huggingface_repo, output_path, 
                                  few_shot_examples_path=None, audio_dir="speakbench_audio"):
    """
    Prepare a JSON dataset for the pronunciation evaluation by combining:
    1. A Huggingface dataset with audio features
    2. A CSV file with human labels
    
    Optionally filters out examples that are being used as few-shot examples.
    
    Args:
        huggingface_repo (str): The repository name on Huggingface
        csv_path (str): Path to the CSV file with human labels
        output_path (str): Path where the output JSON will be saved
        few_shot_examples_path (str): Path to JSON file containing few-shot examples to exclude
        audio_dir (str): Directory to save the audio files
    """
    # Create audio directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Created audio directory: {audio_dir}")
    print(f"Loading dataset from Huggingface: {huggingface_repo}")
    
    # Load the raw dataset from Huggingface
    dataset = load_dataset(huggingface_repo, split="train")
    
    # Convert to pandas DataFrame for easier manipulation
    hf_df = pd.DataFrame(dataset)
    
    
    # Load few-shot examples if provided to exclude them from the dataset
    few_shot_examples = set()
    if few_shot_examples_path and os.path.exists(few_shot_examples_path):
        print(f"Loading few-shot examples from: {few_shot_examples_path}")
        with open(few_shot_examples_path, 'r') as f:
            few_shot_data = json.load(f)
        examples = few_shot_data.get("speakbench", [])
        for example in examples:

            index = example.get('index')
            few_shot_examples.add(index)
    
        
        print(f"Found {len(few_shot_examples)} few-shot examples to exclude")
    
    # Initialize empty list to store our prepared data
    prepared_data = []
    
    print("Processing datasets and matching entries...")
        
    for _, label_row in tqdm(hf_df.iterrows(), total=len(hf_df)):

        index = label_row['idx']
        if index in few_shot_examples:
            print(f"Skipping few-shot example: {index}")
            continue
        label = label_row['label']
        if label == "a":
            label = "1"
        elif label == "b":
            label = "2"
        else:
            label = "tie"
        model_a = label_row['model_a']
        model_b = label_row['model_b']
        # Get audio data from the dataset
        instruction = label_row['instruction']
        instruction_text = label_row['instruction_text']
        audio1_data = label_row['audio_a']
        audio2_data = label_row['audio_b']
        datapoint_dir = os.path.join(audio_dir, str(index))
        os.makedirs(datapoint_dir, exist_ok=True)
        
        # Create unique filenames for the audio files based on index
        instruction_filename = f"instruction.wav"
        audio1_filename = f"audio_a.wav"
        audio2_filename = f"audio_b.wav"
        
        # Full paths to save the audio files
        instruction_path = os.path.join(datapoint_dir, instruction_filename)
        audio1_path = os.path.join(datapoint_dir, audio1_filename)
        audio2_path = os.path.join(datapoint_dir, audio2_filename)

        
        # Save audio files
        try:
            # Save audio1
            # For HuggingFace datasets, audio might be returned in different formats
            # If it's a dictionary with 'array' and 'sampling_rate'
            if isinstance(instruction, dict) and 'array' in instruction and 'sampling_rate' in instruction:
                import soundfile as sf
                sf.write(instruction_path, instruction['array'], instruction['sampling_rate'])
            # If it's a path
            elif isinstance(instruction, str) and os.path.exists(instruction):
                shutil.copy(instruction, instruction_path)
            # If it's bytes
            elif isinstance(instruction, bytes):
                with open(instruction_path, 'wb') as f:
                    f.write(instruction)
            else:
                print(f"Warning: Unsupported audio format for {idx}")

                continue
            if isinstance(audio1_data, dict) and 'array' in audio1_data and 'sampling_rate' in audio1_data:
                import soundfile as sf
                sf.write(audio1_path, audio1_data['array'], audio1_data['sampling_rate'])
            # If it's a path
            elif isinstance(audio1_data, str) and os.path.exists(audio1_data):
                shutil.copy(audio1_data, audio1_path)
            # If it's bytes
            elif isinstance(audio1_data, bytes):
                with open(audio1_path, 'wb') as f:
                    f.write(audio1_data)
            else:
                print(f"Warning: Unsupported audio format for {idx}")

                continue
                
            # Save audio2 (same approach)
            if isinstance(audio2_data, dict) and 'array' in audio2_data and 'sampling_rate' in audio2_data:
                import soundfile as sf
                sf.write(audio2_path, audio2_data['array'], audio2_data['sampling_rate'])
            elif isinstance(audio2_data, str) and os.path.exists(audio2_data):
                shutil.copy(audio2_data, audio2_path)
            elif isinstance(audio2_data, bytes):
                with open(audio2_path, 'wb') as f:
                    f.write(audio2_data)
            else:
                print(f"Warning: Unsupported audio format for {idx}")

                continue
                
        except Exception as e:
            print(f"Error saving audio for {idx}: {str(e)}")
            continue
        data_point = {
            "index": index,
            "instruction_path": instruction_path,  # Path to saved instruction
            "instruction_text": instruction_text,
            "model_a": model_a,
            "model_b": model_b,
            "audio1_path": audio1_path,  # Path to saved audio1
            "audio2_path": audio2_path,  # Path to saved audio2
            "label": str(label).lower(), 
        }
        
        prepared_data.append(data_point)
    
    print(f"Created {len(prepared_data)} data points. Saving to {output_path}")
    # Save the prepared data as JSON
    with open(output_path, 'w') as f:
        json.dump(prepared_data, f, indent=2)
    
    print("Dataset preparation complete!")
    return prepared_data

def prepare_chatbotarena_dataset(huggingface_repo, output_path, 
                                  few_shot_examples_path=None, audio_dir="chatbotarena_audio"):
    """
    Prepare a JSON dataset for the pronunciation evaluation by combining:
    1. A Huggingface dataset with audio features (loaded fully and shuffled)
    2. Same column structure as speakbench
    
    Optionally filters out examples that are being used as few-shot examples.
    
    Args:
        huggingface_repo (str): The repository name on Huggingface
        output_path (str): Path where the output JSON will be saved
        few_shot_examples_path (str): Path to JSON file containing few-shot examples to exclude
        audio_dir (str): Directory to save the audio files
    """
    import os
    import json
    import shutil
    import random
    from datasets import load_dataset
    from tqdm import tqdm
    
    # Create audio directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Created audio directory: {audio_dir}")
    print(f"Loading dataset from Huggingface: {huggingface_repo}")
    
    # Load few-shot examples if provided to exclude them from the dataset
    few_shot_examples = set()
    if few_shot_examples_path and os.path.exists(few_shot_examples_path):
        print(f"Loading few-shot examples from: {few_shot_examples_path}")
        with open(few_shot_examples_path, 'r') as f:
            few_shot_data = json.load(f)
        examples = few_shot_data.get("chatbotarena", [])  # Changed from speakbench
        for example in examples:
            index = example.get('index')
            few_shot_examples.add(index)
        print(f"Found {len(few_shot_examples)} few-shot examples to exclude")
    
    # Load the full dataset (not streaming)
    print("Loading full dataset...")
    dataset = load_dataset(huggingface_repo, split="train")
    print(f"Loaded {len(dataset)} examples")
    
    # Create shuffled indices instead of shuffling the actual data
    print("Creating shuffled indices...")
    random.seed(42)  # Set seed for reproducibility
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    print("Indices shuffled")
    
    # Initialize empty list to store our prepared data
    prepared_data = []
    prepared_data_BA = []
    
    print("Processing datasets and matching entries...")
    
    # Process the dataset using shuffled indices
    processed_count = 0
    for shuffled_idx in tqdm(indices, desc="Processing examples"):
        # Get the example using the shuffled index
        example = dataset[shuffled_idx]
        # Use shuffled_idx as the original_index for matching with few-shot examples
        original_index = shuffled_idx
        
        if original_index in few_shot_examples:
            print(f"Skipping few-shot example: {original_index}")
            continue
            
        # Extract label and convert format (same as speakbench)
        metadata = example.get('metadata')
        label = metadata.get('winner')  # Based on metadata showing 'winner': 'model_a'
        if label == "model_a":
            label = "1"
            BA_label = "2"
        elif label == "model_b":
            label = "2"
            BA_label = "1"
        else:
            label = "tie"
            
        # Extract models
        model_a = metadata.get('model_a')
        model_b = metadata.get('model_b')
        
        # Extract conversation content for instruction
        conversation_a = metadata.get('conversation_a')
        conversation_b = metadata.get('conversation_b')
        
        # Get the user question (first message in conversation_a)
        instruction_text = conversation_a[0].get('content')
        transcript_a = conversation_a[1].get('content')
        transcript_b = conversation_b[1].get('content')
        
        # Get audio data (assuming voice_a, voice_b, voice_user contain audio)
        voice_user = example.get('question')  # This should be the instruction audio
        voice_a = example.get('assistant_a')       # This should be audio_a
        voice_b = example.get('assistant_b')       # This should be audio_b
        
        # Use original index for directory naming to maintain consistency with few-shot matching
        datapoint_dir = os.path.join(audio_dir, str(original_index))
        os.makedirs(datapoint_dir, exist_ok=True)
        
        # Create filenames
        instruction_filename = "instruction.wav"
        audio1_filename = "audio_a.wav"
        audio2_filename = "audio_b.wav"
        
        # Full paths
        instruction_path = os.path.join(datapoint_dir, instruction_filename)
        audio1_path = os.path.join(datapoint_dir, audio1_filename)
        audio2_path = os.path.join(datapoint_dir, audio2_filename)
        
        # Save audio files
        try:
            # Save instruction audio (voice_user)
            if voice_user is not None:
                save_audio_file(voice_user, instruction_path, original_index)
            else:
                print(f"Warning: No instruction audio for {original_index}")
                continue
                
            # Save audio_a (voice_a)
            if voice_a is not None:
                save_audio_file(voice_a, audio1_path, original_index)
            else:
                print(f"Warning: No audio_a for {original_index}")
                continue
                
            # Save audio_b (voice_b)
            if voice_b is not None:
                save_audio_file(voice_b, audio2_path, original_index)
            else:
                print(f"Warning: No audio_b for {original_index}")
                continue
                
        except Exception as e:
            print(f"Error saving audio for {original_index}: {str(e)}")
            continue
        
        # Create data point with same structure as speakbench
        # Use original_index to maintain consistency with few-shot examples
        data_point = {
            "index": original_index,
            "instruction_path": instruction_path,
            "instruction_text": instruction_text,
            "model_a": model_a,
            "model_b": model_b,
            "audio1_path": audio1_path,
            "audio2_path": audio2_path,
            "label": str(label).lower(),
            "transcript_a": transcript_a,
            "transcript_b": transcript_b,
        }

        data_point_BA = {
            "index": original_index,
            "instruction_path": instruction_path,
            "instruction_text": instruction_text,
            "model_a": model_b,
            "model_b": model_a,
            "audio1_path": audio2_path,
            "audio2_path": audio1_path,
            "label": str(BA_label).lower(),
            "transcript_a": transcript_b,
            "transcript_b": transcript_a,
        }

        prepared_data.append(data_point)
        prepared_data_BA.append(data_point_BA)
        processed_count += 1
        
        if processed_count >= 1000:
            break 
    
    print(f"Created {len(prepared_data)} data points. Saving to {output_path}")
    
    # Save the prepared data as JSON
    with open(output_path, 'w') as f:
        json.dump(prepared_data, f, indent=2)
    with open(output_path.replace('chatbotarena', 'chatbotarena_BA'), 'w') as f_BA:
        json.dump(prepared_data_BA, f_BA, indent=2)
    
    print("Dataset preparation complete!")
    return prepared_data


def save_audio_file(audio_data, file_path, index):
    """
    Save audio data to file, handling different formats.
    
    Args:
        audio_data: Audio data in various formats
        file_path: Path where to save the audio
        index: Index for error reporting
    """
    try:
        # If it's a dictionary with 'array' and 'sampling_rate' (HuggingFace format)
        if isinstance(audio_data, dict) and 'array' in audio_data and 'sampling_rate' in audio_data:
            import soundfile as sf
            sf.write(file_path, audio_data['array'], audio_data['sampling_rate'])
        # If it's a path to existing file
        elif isinstance(audio_data, str) and os.path.exists(audio_data):
            shutil.copy(audio_data, file_path)
        # If it's bytes
        elif isinstance(audio_data, bytes):
            with open(file_path, 'wb') as f:
                f.write(audio_data)
        # If it's a list (might be [format, filename] based on metadata)
        elif isinstance(audio_data, list) and len(audio_data) >= 2:
            # Handle format like ['b', 'bm_george'] - might need different logic
            # For now, assume second element is audio identifier
            print(f"Audio data is list format for {index}: {audio_data}")
            # You might need to implement specific logic based on your dataset structure
            return False
        else:
            print(f"Warning: Unsupported audio format for {index}: {type(audio_data)}")
            return False
        return True
    except Exception as e:
        print(f"Error saving audio for {index}: {str(e)}")
        return False

def prepare_speakbench508_dataset(huggingface_repo, output_path, 
                                  few_shot_examples_path=None, audio_dir="speakbench508_audio"):
    """
    Prepare a JSON dataset for the pronunciation evaluation by combining:
    1. A Huggingface dataset with audio features
    2. A CSV file with human labels
    
    Optionally filters out examples that are being used as few-shot examples.
    
    Args:
        huggingface_repo (str): The repository name on Huggingface
        csv_path (str): Path to the CSV file with human labels
        output_path (str): Path where the output JSON will be saved
        few_shot_examples_path (str): Path to JSON file containing few-shot examples to exclude
        audio_dir (str): Directory to save the audio files
    """
    # Create audio directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    print(f"Created audio directory: {audio_dir}")
    print(f"Loading dataset from Huggingface: {huggingface_repo}")
    
    # Load the raw dataset from Huggingface
    dataset = load_dataset(huggingface_repo, split="train")
    
    # Convert to pandas DataFrame for easier manipulation
    hf_df = pd.DataFrame(dataset)
    
    
    # Load few-shot examples if provided to exclude them from the dataset
    few_shot_examples = set()
    if few_shot_examples_path and os.path.exists(few_shot_examples_path):
        print(f"Loading few-shot examples from: {few_shot_examples_path}")
        with open(few_shot_examples_path, 'r') as f:
            few_shot_data = json.load(f)
        examples = few_shot_data.get("speakbench508", [])
        for example in examples:
            index = example.get('index')
            few_shot_examples.add(index)
    
        
        print(f"Found {len(few_shot_examples)} few-shot examples to exclude")
    
    # Initialize empty list to store our prepared data
    prepared_data = []
    
    print("Processing datasets and matching entries...")
        
    for _, label_row in tqdm(hf_df.iterrows(), total=len(hf_df)):

        index = label_row['i']
        if index in few_shot_examples:
            print(f"Skipping few-shot example: {index}")
            continue
        instruction_id = label_row['instruction_ID']
        if instruction_id == 0:
            print(f"Skipping instruction ID 0 for {index}")
            continue
        label = label_row['label']
        if label == "a":
            label = "1"
        elif label == "b":
            label = "2"
        else:
            label = "tie"
        model_a = label_row['model_a']
        model_b = label_row['model_b']
        # Get audio data from the dataset
        instruction = label_row['instruction']
        instruction_text = label_row['instruction_text']
        audio1_data = label_row['audio_a']
        audio2_data = label_row['audio_b']
        datapoint_dir = os.path.join(audio_dir, str(index))
        os.makedirs(datapoint_dir, exist_ok=True)
        
        # Create unique filenames for the audio files based on index
        instruction_filename = f"instruction.wav"
        audio1_filename = f"audio_a.wav"
        audio2_filename = f"audio_b.wav"
        
        # Full paths to save the audio files
        instruction_path = os.path.join(datapoint_dir, instruction_filename)
        audio1_path = os.path.join(datapoint_dir, audio1_filename)
        audio2_path = os.path.join(datapoint_dir, audio2_filename)

        
        # Save audio files
        try:
            # Save audio1
            # For HuggingFace datasets, audio might be returned in different formats
            # If it's a dictionary with 'array' and 'sampling_rate'
            if isinstance(instruction, dict) and 'array' in instruction and 'sampling_rate' in instruction:
                import soundfile as sf
                sf.write(instruction_path, instruction['array'], instruction['sampling_rate'])
            # If it's a path
            elif isinstance(instruction, str) and os.path.exists(instruction):
                shutil.copy(instruction, instruction_path)
            # If it's bytes
            elif isinstance(instruction, bytes):
                with open(instruction_path, 'wb') as f:
                    f.write(instruction)
            else:
                print(f"Warning: Unsupported audio format for {index}")

                continue
            if isinstance(audio1_data, dict) and 'array' in audio1_data and 'sampling_rate' in audio1_data:
                import soundfile as sf
                sf.write(audio1_path, audio1_data['array'], audio1_data['sampling_rate'])
            # If it's a path
            elif isinstance(audio1_data, str) and os.path.exists(audio1_data):
                shutil.copy(audio1_data, audio1_path)
            # If it's bytes
            elif isinstance(audio1_data, bytes):
                with open(audio1_path, 'wb') as f:
                    f.write(audio1_data)
            else:
                print(f"Warning: Unsupported audio format for {index}")

                continue
                
            # Save audio2 (same approach)
            if isinstance(audio2_data, dict) and 'array' in audio2_data and 'sampling_rate' in audio2_data:
                import soundfile as sf
                sf.write(audio2_path, audio2_data['array'], audio2_data['sampling_rate'])
            elif isinstance(audio2_data, str) and os.path.exists(audio2_data):
                shutil.copy(audio2_data, audio2_path)
            elif isinstance(audio2_data, bytes):
                with open(audio2_path, 'wb') as f:
                    f.write(audio2_data)
            else:
                print(f"Warning: Unsupported audio format for {index}")

                continue
                
        except Exception as e:
            print(f"Error saving audio for {index}: {str(e)}")
            continue
        data_point = {
            "index": index,
            "instruction_path": instruction_path,  # Path to saved instruction
            "instruction_text": instruction_text,
            "model_a": model_a,
            "model_b": model_b,
            "audio1_path": audio1_path,  # Path to saved audio1
            "audio2_path": audio2_path,  # Path to saved audio2
            "label": str(label).lower(), 
            "instruction_id": instruction_id,
        }
        
        prepared_data.append(data_point)
    
    print(f"Created {len(prepared_data)} data points. Saving to {output_path}")
    # Save the prepared data as JSON
    with open(output_path, 'w') as f:
        json.dump(prepared_data, f, indent=2)
    
    print("Dataset preparation complete!")
    return prepared_data
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare dataset for audio evaluation")
    parser.add_argument("--dataset_name", type=str, required=True, help="name of the dataset to load")
    args = parser.parse_args()
    audio_root = "audio_data"
    os.makedirs(audio_root, exist_ok=True)
    dataset_dir = "datasets"
    os.makedirs(dataset_dir, exist_ok=True)
    audio_dir = os.path.join(audio_root, f"{args.dataset_name}_audio")
    os.makedirs(audio_dir, exist_ok=True)
    output_path = os.path.join(dataset_dir, f"{args.dataset_name}_dataset.json")
    if args.dataset_name == "pronunciation":
        huggingface_repo = "MichaelR207/wiktionary_pronunciations-final"
        csv_path = "human_labels.csv"
        few_shot_examples_path = "few_shots_examples.json"
    
        prepared_data = prepare_pronunciation_dataset(
            huggingface_repo, 
            csv_path, 
            output_path, 
            few_shot_examples_path, 
            audio_dir
        )
        
        print(f"Saved {len(prepared_data)} data points to {output_path}")
        print(f"Audio files saved in {audio_dir} directory")
    elif args.dataset_name == "speed":
        huggingface_repo = "potsawee/paralinguistic-judge-speed"
        csv_path = "human_labels.csv"
        few_shot_examples_path = "few_shots_examples.json"
    
        prepared_data = prepare_speed_dataset(
            huggingface_repo, 
            output_path, 
            few_shot_examples_path, 
            audio_dir
        )
        
        print(f"Saved {len(prepared_data)} data points to {output_path}")
        print(f"Audio files saved in {audio_dir} directory")
    elif args.dataset_name == "speaker":
        huggingface_repo = "potsawee/paralinguistic-judge-speaker"
        few_shot_examples_path = "few_shots_examples.json"
    
        prepared_data = prepare_speaker_dataset(
            huggingface_repo, 
            output_path, 
            few_shot_examples_path, 
            audio_dir
        )
        print(f"Saved {len(prepared_data)} data points to {output_path}")
        print(f"Audio files saved in {audio_dir} directory")
    elif args.dataset_name == "tmhintq":
        huggingface_repo = "potsawee/speech-quality-tmhintq-pairwise"
        few_shot_examples_path = "few_shots_examples.json"
        prepared_data = prepare_tmhintq_dataset(
            huggingface_repo, 
            output_path, 
            few_shot_examples_path, 
            audio_dir
        )

        
        print(f"Saved {len(prepared_data)} data points to {output_path}")
        print(f"Audio files saved in {audio_dir} directory")
    elif args.dataset_name == "somos":
        huggingface_repo = "potsawee/speech-quality-somos-pairwise-diff1.0"
        few_shot_examples_path = "few_shots_examples.json"
        prepared_data = prepare_somos_dataset(
            huggingface_repo, 
            output_path, 
            few_shot_examples_path, 
            audio_dir
        )

        
        print(f"Saved {len(prepared_data)} data points to {output_path}")
        print(f"Audio files saved in {audio_dir} directory")

    elif args.dataset_name == "thaimos":
        huggingface_repo = "potsawee/speech-quality-thaimos-pairwise"
        few_shot_examples_path = "few_shots_examples.json"
        prepared_data = prepare_thaimos_dataset(
            huggingface_repo, 
            output_path, 
            few_shot_examples_path, 
            audio_dir
        )
        print(f"Saved {len(prepared_data)} data points to {output_path}")
        print(f"Audio files saved in {audio_dir} directory")

    elif args.dataset_name == "chatbotarena":
        huggingface_repo = "potsawee/chatbotarena-spoken-all-7824"
        few_shot_examples_path = "few_shots_examples.json"
        prepared_data = prepare_chatbotarena_dataset(
            huggingface_repo, 
            output_path, 
            few_shot_examples_path, 
            audio_dir
        )
        print(f"Saved {len(prepared_data)} data points to {output_path}")
        print(f"Audio files saved in {audio_dir} directory")
    elif args.dataset_name == "speakbench508":
        huggingface_repo = "potsawee/speakbench-v1-labelled-508"
        few_shot_examples_path = "few_shots_examples.json"
        prepared_data = prepare_speakbench508_dataset(
            huggingface_repo, 
            output_path, 
            few_shot_examples_path, 
            audio_dir
        )
        print(f"Saved {len(prepared_data)} data points to {output_path}")
        print(f"Audio files saved in {audio_dir} directory")