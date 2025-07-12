# AudioJudge Main Experiment

The codebase for reproducing our results in AudioJudge main paper.

## Setup

Create a `.env` file with your API keys:

```bash
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Configurations

The experiments use the following configurations by default:

```python
PROMPT_TYPES = ["standard_cot"] # Different types of prompts to use

# Transcript options
TRANSCRIPT_TYPES = ["none"]  # Can also include "groundtruth", "asr"

# Few-shot example configurations
FEWSHOT_CONFIGS = {
    "aggregate": {"aggregate_fewshot": True, "concat_fewshot": False},
    # "concat": {"aggregate_fewshot": False, "concat_fewshot": True},
    # "separate": {"aggregate_fewshot": False, "concat_fewshot": False}
}

# Test audio configurations
TEST_CONFIGS = [False, True]  # False = separate, True = concatenated

# Turn configuration
TWO_TURNS_CONFIGS = [False]  # Can also include True for two-turn mode
```

## Prompting Methods

The five main prompting strategies correspond to specific configuration combinations:

1. **No Concatenation**: 
   - `FEWSHOT_CONFIGS["separate"]` (`aggregate_fewshot=False, concat_fewshot=False`) 
   - `TEST_CONFIGS=False` (separate test files)

2. **Pair Example Concatenation**: 
   - `FEWSHOT_CONFIGS["concat"]` (`aggregate_fewshot=False, concat_fewshot=True`) 
   - `TEST_CONFIGS=False` (separate test files)

3. **Examples Concatenation**: 
   - `FEWSHOT_CONFIGS["aggregate"]` (`aggregate_fewshot=True, concat_fewshot=False`) 
   - `TEST_CONFIGS=False` (separate test files)

4. **Test Concatenation**: 
   - `FEWSHOT_CONFIGS["separate"]` (`aggregate_fewshot=False, concat_fewshot=False`) 
   - `TEST_CONFIGS=True` (concatenated test files)

5. **Test&Examples Concatenation**: 
   - `FEWSHOT_CONFIGS["aggregate"]` (`aggregate_fewshot=True, concat_fewshot=False`) 
   - `TEST_CONFIGS=True` (concatenated test files)

## Running Experiments 

### Standard Datasets (except SpeakBench)

1. **Prepare dataset** from Hugging Face:
   ```bash
   python prepare_dataset.py --dataset_name <DATASET_NAME>
   ```
   
   Where `<DATASET_NAME>` is one of: `pronunciation`, `speed`, `speaker`, `somos`, `thaimos`, `tmhintq`, `chatbotarena`

2. **Run experiment**:
   ```bash
   bash main.sh
   ```

   **Note**: For chatbotarena, you need to run both datasetname as `chatbotarena` and `chatbotarena_BA` in main to mitigate the effect of positional bias in the winrate calculation later.

3. **Get results**: You will get the accuracy score in main.

4. **Calculate correlation** for chatbotarena:
   ```bash
   python correlation_chatbotarena.py
   ```

### SpeakBench Dataset

1. **Prepare the SpeakBench dataset**:
   
   First, download the dataset with human annotation and instruction audios:
   ```bash
   python prepare_dataset.py --dataset_name speakbench508
   ```
   
   Then download the dataset:
   ```bash
   python prepare_dataset_speakbench.py --dataset_name speakbench
   ```
   
2. **Run evaluation**:
   ```bash
   bash main_speakbench.sh
   ```

3. **Calculate correlation**:
   ```bash
   python correlation_speakbench.py
   ```

### Ensemble Method (SpeakBench)

If you want to try the ensemble method:

1. **Change prompt types**:
   ```bash
   --prompt_types paralinguistic_cot lexical_cot speech_quality_cot
   ```

2. **Run ensemble**:
   ```bash
   python ensemble_speakbench.py
   ```
   
   This ensembles the results of different judges. The metrics will be saved to `ensemble_results/detailed_ensemble_time/model_metrics.csv`.

3. **Calculate correlation**: Use the ensemble results filepath with `correlation_speakbench.py`.

### Pointwise Evaluation

For pointwise evaluation (independent audio rating):

```bash
bash main_pointwise.sh
```

## Additional Analysis

- **Human Baseline**: See `human_baseline/` folder for annotator performance evaluation
- **Positional Bias**: See `positional_bias/` folder for bias analysis  
- **Significance Test**: See `significance_test/` folder for statistical significance testing