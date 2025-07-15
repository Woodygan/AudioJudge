import json
import re
from datasets import load_dataset


def filter_yes_results(input_filepath, output_filepath):
    with open(input_filepath, "r") as infile:
        filtered_data = []
        for line in infile:
            item = json.loads(line.strip())
            if item.get("result") == "Yes":
                item["conversation_a"] = filter_conversation(item["conversation_a"])
                item["conversation_b"] = filter_conversation(item["conversation_b"])
                filtered_data.append(item)

    with open(output_filepath, "w") as outfile:
        for item in filtered_data:
            outfile.write(json.dumps(item) + "\n")


def filter_conversation(conversation):
    return [
        {
            "content": re.sub(
                r"\\u[0-9A-Fa-f]{4}|[^\x00-\x7F+=\-<>]", "", message["content"]
            ),
            "role": message["role"],
        }
        for message in conversation
        if not re.search(r"\\u[0-9A-Fa-f]{4}", message["content"])
    ]


def get_question_ids_with_yes_results(input_filepath):
    question_ids = []
    with open(input_filepath, "r") as infile:
        for line in infile:
            item = json.loads(line.strip())
            if item["result"] == "Yes":
                question_ids.append(item["question_id"])
    return question_ids


def filter_dataset_conversations(dataset):
    for item in dataset:
        item["conversation_a"] = filter_conversation(item["conversation_a"])
        item["conversation_b"] = filter_conversation(item["conversation_b"])
    return dataset


# Example usage
ds = load_dataset("lmsys/chatbot_arena_conversations")["train"]

# Get question IDs where result is 'Yes'
input_filepath = "conversations_ChatBotArena.jsonl"
question_ids = get_question_ids_with_yes_results(input_filepath)

# Filter conversations in the dataset
filtered_ds = filter_dataset_conversations(ds)

# Dump the filtered dataset as JSONL file
output_filepath = "filtered_dataset_ChatBotArena.jsonl"
with open(output_filepath, "w") as outfile:
    for item in filtered_ds:
        outfile.write(json.dumps(item) + "\n")
