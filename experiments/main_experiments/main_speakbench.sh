for model_name in "gpt-4o-audio-preview"; do
    python main_speakbench.py --model $model_name --n_shots 4 --prompt_types standard_cot
done