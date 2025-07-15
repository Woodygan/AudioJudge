# %%
import json
import os
import re

# %%
with open("./chatbot-arena-spoken-1turn-english-difference-voices.json") as f:
    raw_data = json.load(f)
gts = []
gt_mapping = {"model_a": "A", "model_b": "B", "tie": "C", "tie (bothbad)": "C"}
for x in raw_data:
    gts.append(gt_mapping[x["winner"]])
print("len:", len(gts))


# %%
def read_jsonl(file_path):
    data = []
    # Open and read the file line by line
    with open(file_path, "r") as file:
        for line in file:
            # Parse each line as a JSON object
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    print("len:", len(data))
    return data


# %%
def process_output(data):
    labels = []
    for x in data:
        response = x["response"]
        response = response[-20:]
        labels += [extract_abc(response)]
    calculate_percentage(labels)
    return labels


def calculate_correciness(labels, gts, reverse=False):
    # assert len(labels) == len(gts)
    labels = labels[: len(gts)]
    correct, incorrect = 0, 0
    correct_AB, incorrect_AB = 0, 0
    for label, gt in zip(labels, gts):
        if reverse == True:
            if gt == "A":
                gt = "B"
            elif gt == "B":
                gt = "A"
        if label == gt:
            correct += 1
        else:
            incorrect += 1

        if label == "A" or label == "B":
            if label == gt:
                correct_AB += 1
            else:
                incorrect_AB += 1
    print("correct:   {:.2f}%".format(correct / (correct + incorrect) * 100))
    print("incorrect: {:.2f}%".format(incorrect / (correct + incorrect) * 100))
    # print("correct_AB:   {:.2f}%".format(correct_AB/(correct_AB+incorrect_AB)*100))
    # print("incorrect_AB: {:.2f}%".format(incorrect_AB/(correct_AB+incorrect_AB)*100))


# %%
def extract_abc(text):
    pattern = r"\[\[(A|B|C)\]\]"

    # Search for the match
    match = re.search(pattern, text)

    if match:
        result = match.group(1)
        # print(f"Extracted value: {result}")
    else:
        result = "D"
    return result


# %%
def calculate_percentage(arr):
    # Get the total number of items
    total_count = len(arr)

    # Create a dictionary to store counts of each unique item
    item_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

    for item in arr:
        item_counts[item] = item_counts.get(item, 0) + 1

    # Calculate percentages and store them in a dictionary
    percentages = {
        item: (count / total_count) * 100 for item, count in item_counts.items()
    }

    # Display the result
    print("---------------")
    for item, percentage in percentages.items():
        print(f"{item}: {percentage:.2f}%")
    print("---------------")


# %%
def measure_bias(preds_ab, preds_ba):
    n = min(len(preds_ab), len(preds_ba))
    preds_ab = preds_ab[:n]
    preds_ba = preds_ba[:n]
    biasA, biasB, consistentAB, consistentCC, other = 0, 0, 0, 0, 0
    for pred_ab, pred_ba in zip(preds_ab, preds_ba):
        if pred_ab == "A" and pred_ba == "A":
            biasA += 1
        elif pred_ab == "B" and pred_ba == "B":
            biasB += 1
        elif pred_ab == "A" and pred_ba == "B":
            consistentAB += 1
        elif pred_ab == "B" and pred_ba == "A":
            consistentAB += 1
        elif pred_ab == "C" and pred_ba == "C":
            consistentCC += 1
        else:
            other += 1
    total = biasA + biasB + consistentAB  # + consistentCC + other
    print("consistentAB: {:.2f}".format(consistentAB / total * 100))
    # print("consistentCC: {:.2f}".format(consistentCC/total*100))
    print("biasA:      {:.2f}".format(biasA / total * 100))
    print("biasB:      {:.2f}".format(biasB / total * 100))
    # print("other:      {:.2f}".format(other/total*100))


# %%
text_text = process_output(
    read_jsonl("./experiments/chatbot-arena-7824/text-text-gemini1.5flash.jsonl")
)
calculate_correciness(text_text, gts)
print("############################")
text_text_BA = process_output(
    read_jsonl("./experiments/chatbot-arena-7824/text-text-gemini1.5flash_BA.jsonl")
)
calculate_correciness(text_text_BA, gts, reverse=True)
print("############################")
measure_bias(text_text, text_text_BA)

# %%
0.5 * (52.15 + 52.35)

# %%
audio_text = process_output(
    read_jsonl("./experiments/chatbot-arena-7824/audio-text-gemini1.5flash.jsonl")
)
calculate_correciness(audio_text, gts)
print("############################")
audio_text_BA = process_output(
    read_jsonl("./experiments/chatbot-arena-7824/audio-text-gemini1.5flash_BA.jsonl")
)
calculate_correciness(audio_text_BA, gts, reverse=True)
print("############################")
measure_bias(audio_text, audio_text_BA)

# %%
0.5 * (52.44 + 51.71)

# %%
audio_audio = process_output(
    read_jsonl("./experiments/chatbot-arena-7824/audio-audio-gemini1.5flash.jsonl")
)
calculate_correciness(audio_audio, gts)
print("############################")
audio_audio_BA = process_output(
    read_jsonl("./experiments/chatbot-arena-7824/audio-audio-gemini1.5flash_BA.jsonl")
)
calculate_correciness(audio_audio_BA, gts, reverse=True)
print("############################")
measure_bias(audio_audio, audio_audio_BA)
