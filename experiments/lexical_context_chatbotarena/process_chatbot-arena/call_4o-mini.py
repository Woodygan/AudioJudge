import re  # Import regex module
from datasets import load_dataset
import openai
import pdb
import json  # Import JSON module for serialization
from openai import OpenAI


def call_4omini(prompt):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message


def parse_response(response):
    """
    Parse the response to extract [[Yes]] or [[No]].
    """
    match = re.search(r"\[\[Yes\]\]|\[\[No\]\]", response["content"])
    if match:
        return match.group(0)  # Return the matched [[Yes]] or [[No]]
    else:
        return "No valid answer found"


prompt_template = """
You are an expert in natural language understanding. 
Your task is to determine whether the provided JSON input represents a conversational structure or not, and also evaluate the quality of the conversation in terms of code switching or language switching.
A conversational structure involves interactions resembling dialogue, where there is an exchange of information, questions, or responses between participants.

If there is any code switching (e.g., switching between programming code and natural language) or language switching (e.g., switching between different spoken languages) that makes the semantics or meaning of the conversation nonsensical or incoherent, classify the conversation as not a good conversation.

Output '[[Yes]]' if:
1. The JSON represents a conversational structure.
2. There is no code switching or language switching that causes the semantics to be nonsensical.

Output '[[No]]' if:
1. The JSON does not represent a conversational structure.
2. There is any code switching or language switching that makes the semantics nonsensical or incoherent.

Provide a brief explanation for your decision.

Here is the JSON input:

###INPUT_JSON###
"""

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("lmsys/chatbot_arena_conversations")

# Open JSONL file for appending
with open("conversations_results.jsonl", "a", encoding="utf-8") as jsonl_file:
    for data in ds["train"]:
        question_id = data["question_id"]
        conversation_a = data["conversation_a"]
        conversation_b = data["conversation_b"]

        # Serialize conversations into JSON strings
        conversation_a_json = json.dumps(conversation_a, ensure_ascii=False, indent=2)
        conversation_b_json = json.dumps(conversation_b, ensure_ascii=False, indent=2)

        # Prepare prompts
        prompt_a = prompt_template.replace("###INPUT_JSON###", conversation_a_json)
        prompt_b = prompt_template.replace("###INPUT_JSON###", conversation_b_json)

        # Call the GPT-4o-mini model with the prompt
        cls_a = call_4omini(prompt_a)
        cls_b = call_4omini(prompt_b)

        # Parse the responses
        result_a = parse_response(cls_a)
        result_b = parse_response(cls_b)

        # Create JSON objects
        result_a_json = {
            "question_id": question_id,
            "conversation": conversation_a,
            "result": result_a,
        }
        result_b_json = {
            "question_id": question_id,
            "conversation": conversation_b,
            "result": result_b,
        }

        # Write JSON objects as lines in the JSONL file
        jsonl_file.write(json.dumps(result_a_json, ensure_ascii=False) + "\n")
        jsonl_file.write(json.dumps(result_b_json, ensure_ascii=False) + "\n")

        # Print confirmation for debugging
        print(f"Appended results for question_id {question_id}")
