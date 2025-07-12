# AudioJudge Experiments 🔬

This folder contains the results and complete experimental codebase for reproducing the results from our AudioJudge research paper. The experiments validate AudioJudge's performance across multiple audio evaluation datasets and comparison methodologies.

## Structure

```
experiments/
├── main_experiments/           # Core paper experiments
│   ├── prepare_dataset.py      # Data preparation scripts
│   ├── main.sh                 # Main experiment runner
│   ├── results/                # results of experiments
│   ├── correlation_*.py        # Correlation analysis
│   └── README.md                   # Detailed setup and usage
├── human_baseline/                 # Human annotator performance
├── positional_bias/                # Bias analysis experiments
├── significance_test/              # Statistical significance testing
├── lexical_context_chatbotarena/   # Cross-Modality experiments for ChatbotArena
└── lam_speakbench_inference/       # Curating SpeakBench Dataset
```