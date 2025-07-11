for dataset in "tmhintq" "somos" "thaimos"; do
    python main_pointwise.py --dataset_name $dataset
done