# %%
import json
import os
import re

# %%
with open("../data/chatbot-arena-spoken-1turn-english-difference-voices.json") as f:
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
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")


# %%
def count_tokens(text):
    return len(tokenizer.tokenize(text))


# %%
def verbosity_bias(preds, gts):
    assert len(preds) == len(gts)
    assert len(raw_data) == len(gts)
    i = 0
    tie, longer, shorter, total = 0, 0, 0, 0
    for pred, gt in zip(preds, gts):
        if gt != "C":
            i += 1
            continue
        tokens_a = count_tokens(raw_data[i]["conversation_a"][1]["content"])
        tokens_b = count_tokens(raw_data[i]["conversation_b"][1]["content"])
        print("tokens_a:", tokens_a, "tokens_b:", tokens_b, "pred:", pred)
        if abs(tokens_a - tokens_b) < 5:
            i += 1
            continue
        if pred in ["C", "D"]:
            tie += 1
        else:
            if tokens_a > tokens_b:
                if pred == "A":
                    longer += 1
                else:
                    shorter += 1
            else:
                if pred == "A":
                    shorter += 1
                else:
                    longer += 1
        total += 1
    print("tie     = {:.2f}%".format(tie / total * 100))
    print("longer  = {:.2f}%".format(longer / total * 100))
    print("shorter = {:.2f}%".format(shorter / total * 100))


text_text = process_output(
    read_jsonl("../experiments/chatbot-arena-7824/audio-text-gemini2.5flash.jsonl")
)
verbosity_bias(text_text, gts)
