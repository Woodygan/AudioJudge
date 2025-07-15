import json
import re


def iterate_jsonl(filepath, output_filepath):
    valid_items = []
    with open(filepath, "r") as infile:
        for line in infile:
            item = json.loads(line.strip())
            if is_valid_conversation(item["conversation_a"]) and is_valid_conversation(
                item["conversation_b"]
            ):
                valid_items.append(item)

    with open(output_filepath, "w") as outfile:
        for item in valid_items:
            outfile.write(json.dumps(item) + "\n")


def is_valid_conversation(conversation):
    for message in conversation:
        if re.search(r"[^\x00-\x7F+\-<>]|<div>|#|\||:|=", message["content"]):
            return False
    return True


# Example usage
input_filepath = "filtered_dataset_ChatBotArena.jsonl"
output_filepath = "valid_conversations_ChatBotArena.jsonl"
iterate_jsonl(input_filepath, output_filepath)
