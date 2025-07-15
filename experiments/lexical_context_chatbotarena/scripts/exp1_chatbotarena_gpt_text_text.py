import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie (which can be equally good and equally bad)."""

prompt_template = """[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"""


def experiment(
    data_path,
    output_path,
    order="ab",
):
    print("-----------------------------")
    print("data_path:", data_path)
    print("output_path:", output_path)
    print("order:", order)
    print("-----------------------------")

    with open(data_path) as f:
        data = json.load(f)
    print("len(data):", len(data))

    outputs = []
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                x = json.loads(line)
                outputs.append(x)
        num_done = len(outputs)
    else:
        num_done = 0
    print("num_done = {}".format(num_done))

    for i in tqdm(range(num_done, len(data))):
        if order == "ab":
            conversation_a, conversation_b = (
                data[i]["conversation_a"],
                data[i]["conversation_b"],
            )
        elif order == "ba":
            conversation_a, conversation_b = (
                data[i]["conversation_b"],
                data[i]["conversation_a"],
            )
        else:
            raise ValueError("order must be 'ab' or 'ba'")
        assert conversation_a[0]["content"] == conversation_b[0]["content"]

        prompt = prompt_template.format(
            question=conversation_a[0]["content"],
            answer_a=conversation_a[1]["content"],
            answer_b=conversation_b[1]["content"],
        )
        message = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        # Send request to GPT-4o for A/B testing
        completion = client.chat.completions.create(
            model="gpt-4o-2024-11-20", messages=message
        )
        # Extract and return the A/B testing decision from the response
        # text_output = completion.choices[0].message.audio.transcript.strip()
        response = completion.choices[0].message.content
        item = {"data_path": data_path, "i": i, "response": response}
        print(i, response)
        with open(output_path, "a") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run a specific model via gradio_client."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Specify the model name to run."
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output Path")
    parser.add_argument(
        "--order", type=str, default="ab", help="Order of the audio files"
    )
    args = parser.parse_args()
    experiment(args.data_path, args.output_path, args.order)
    # usage: python -m scripts.exp1_chatbotarena_gpt_text_text.py --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/text-text-gpt4o.jsonl --order ab

    # usage: python -m scripts.exp1_chatbotarena_gpt_text_text.py --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/text-text-gpt4o_BA.jsonl --order ba


if __name__ == "__main__":
    main()
