import json

if __name__ == "__main__":
    answer_file = "../main_experiments/datasets/tmhintq_dataset.json"
    human_eval_file = "human_eval/tmhintq.json"
    if "speaker" in answer_file or "pronunciation" in answer_file:
        ground_truth = "match"
    else:
        ground_truth = "label"
    with open(answer_file, "r", encoding="utf-8") as f:
        answer_data = json.load(f)
    with open(human_eval_file, "r", encoding="utf-8") as f:
        human_eval_data = json.load(f)
    length = len(human_eval_data)
    score = 0
    for i in range(length):
        if ground_truth == "match":
            answer = answer_data[i]["match"]
            human_answer = human_eval_data[i]
            if answer == "true" and human_answer == 1:
                score += 1
            elif answer == "false" and human_answer == 0:
                score += 1
        else:
            answer = answer_data[i]["label"]
            human_answer = human_eval_data[i]
            if answer == str(human_answer):
                score += 1
    print(f"Score: {score}/{length} = {score / length:.3f}")
    print(f"Ground Truth: {ground_truth}")
