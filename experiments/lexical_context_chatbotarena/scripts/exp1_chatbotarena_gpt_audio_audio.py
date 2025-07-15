import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from gpt4o_audio_api import get_encoded_audio_from_path

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie (which can be equally good and equally bad). Note that the user question and the responses of the assistants will be provided to you in the audio format. You should evaluate the responses based on the user question and not on the responses of the other assistant. You should also not consider the quality of the audio or the voice of the assistants. You should only consider the content of the responses."""


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
        # question
        question_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/kokoroTTS/wav/{data[i]['question_id']}-user.wav"
        encoded_audio_question = get_encoded_audio_from_path(question_wav_path)

        # assistant a
        assistant_a_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/kokoroTTS/wav/{data[i]['question_id']}-assistant-a.wav"
        # assistant b
        assistant_b_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/kokoroTTS/wav/{data[i]['question_id']}-assistant-b.wav"

        if order == "ab":
            encoded_audio_responseA = get_encoded_audio_from_path(assistant_a_wav_path)
            encoded_audio_responseB = get_encoded_audio_from_path(assistant_b_wav_path)
        elif order == "ba":
            encoded_audio_responseA = get_encoded_audio_from_path(assistant_b_wav_path)
            encoded_audio_responseB = get_encoded_audio_from_path(assistant_a_wav_path)
        else:
            raise ValueError("order must be 'ab' or 'ba'")

        message = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is the user question in the audio format.",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio_question,
                            "format": "wav",
                        },
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is the assistant A's response in the audio format.",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio_responseA,
                            "format": "wav",
                        },
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is the assistant B's response in the audio format.",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio_responseB,
                            "format": "wav",
                        },
                    },
                ],
            },
        ]
        # Send request to GPT-4o for A/B testing
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview-2024-12-17",
            modalities=["text"],
            messages=message,
        )
        # Extract and return the A/B testing decision from the response
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

    # usage: python -m scripts.exp1_chatbotarena_gpt_audio_audio.py --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-audio-gpt4o.jsonl --order ab
    # usage: python -m scripts.exp1_chatbotarena_gpt_audio_audio.py --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-audio-gpt4o_BA.jsonl --order ba


if __name__ == "__main__":
    main()
