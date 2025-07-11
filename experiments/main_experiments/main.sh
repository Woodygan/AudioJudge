for dataset in "somos"; do
    for model in "gpt-4o-audio-preview"; do
        python main.py --dataset_name $dataset --model $model
    done
done