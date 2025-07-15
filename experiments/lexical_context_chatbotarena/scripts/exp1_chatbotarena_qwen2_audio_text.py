import os
import io
import json
import argparse
from tqdm import tqdm
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa

system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie (which can be equally good and equally bad). Note that the user question will be provided to you in the audio format, while the responses of the two assistants will be provided to you in the text format. You should evaluate the responses based on the user question and not on the responses of the other assistant. You should also not consider the quality of the audio or the voice of the assistants. You should only consider the content of the responses."""

prompt_template_question = """This is the user question in the audio format."""

prompt_template_response = """[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto"
)


@torch.no_grad()
def run_inference(conversation):
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(
                            ele["audio_url"],
                            sr=processor.feature_extractor.sampling_rate,
                        )[0]
                    )

    inputs = processor(
        text=text,
        audios=audios,
        return_tensors="pt",
        padding=True,
        sampling_rate=processor.feature_extractor.sampling_rate,
    ).to("cuda")

    generate_ids = model.generate(
        **inputs, max_length=4000, do_sample=True, temperature=0.2
    )
    generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


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

        question_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/kokoroTTS/wav/{data[i]['question_id']}-user.wav"

        prompt_response = prompt_template_response.format(
            answer_a=conversation_a[1]["content"], answer_b=conversation_b[1]["content"]
        )

        # Generate the response
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_template_question},
                    {
                        "type": "audio",
                        "audio_url": question_wav_path,
                    },
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt_response}]},
        ]
        response = run_inference(conversation)
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

    # usage: python -m scripts.exp1_chatbotarena_qwen2_audio_text.py --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-text-qwen2.jsonl --order ab
    # usage: python -m scripts.exp1_chatbotarena_qwen2_audio_text.py --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-text-qwen2_BA.jsonl --order ba


if __name__ == "__main__":
    main()
