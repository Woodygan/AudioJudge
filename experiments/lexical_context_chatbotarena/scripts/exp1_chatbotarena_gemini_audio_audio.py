import os
import io
import json
import argparse
from tqdm import tqdm
import google.generativeai as genai
import google
from pydub import AudioSegment

model_name = "gemini-2.5-flash-preview-04-17"
model = genai.GenerativeModel(f"models/{model_name}")

system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie (which can be equally good and equally bad). Note that the user question and the responses of the assistants will be provided to you in the audio format. You should evaluate the responses based on the user question and not on the responses of the other assistant. You should also not consider the quality of the audio or the voice of the assistants. You should only consider the content of the responses."""


def convert_to_16kHz_bytes(audio_path):
    # Load the audio file
    audio = AudioSegment.from_file(audio_path)

    # Check if resampling is necessary
    if audio.frame_rate != 16000:
        print(f"Resampling from {audio.frame_rate} Hz to 16 kHz...")
        audio = audio.set_frame_rate(16000)
    else:
        # print("Audio is already 16 kHz.")
        pass

    # Export the audio to an in-memory buffer in WAV format
    output_buffer = io.BytesIO()
    audio.export(output_buffer, format="wav")
    output_buffer.seek(0)  # Reset the buffer to the beginning

    # Return the binary data equivalent to pathlib.Path().read_bytes()
    return output_buffer.read()


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
        question_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge-new/kokoroTTS/wav/{data[i]['question_id']}-user.wav"

        if order == "ab":
            assistant_a_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge-new/kokoroTTS/wav/{data[i]['question_id']}-assistant-a.wav"
            assistant_b_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge-new/kokoroTTS/wav/{data[i]['question_id']}-assistant-b.wav"
        elif order == "ba":
            assistant_a_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge-new/kokoroTTS/wav/{data[i]['question_id']}-assistant-b.wav"
            assistant_b_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge-new/kokoroTTS/wav/{data[i]['question_id']}-assistant-a.wav"
        else:
            raise ValueError("Invalid order")

        # Generate the response
        response = model.generate_content(
            [
                system_prompt,
                "This is the user question in the audio format.",
                {
                    "mime_type": "audio/wav",
                    "data": convert_to_16kHz_bytes(question_wav_path),
                },
                "This is the assistant A's response in the audio format.",
                {
                    "mime_type": "audio/wav",
                    "data": convert_to_16kHz_bytes(assistant_a_wav_path),
                },
                "This is the assistant B's response in the audio format.",
                {
                    "mime_type": "audio/wav",
                    "data": convert_to_16kHz_bytes(assistant_b_wav_path),
                },
            ]
        )

        try:
            response = response.text
        except ValueError as e:
            response = "ValueError: " + str(e) + "\n" + "[[D]]"

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

    import time

    for _ in range(20):
        try:
            experiment(args.data_path, args.output_path, args.order)
        except google.api_core.exceptions.InternalServerError:
            print("InternalServerError, retrying in 10 seconds...")
            time.sleep(10)

    # usage: python -m scripts.exp1_chatbotarena_gemini_audio_audio.py --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-audio-gemini1.5flash.jsonl --order ab
    # usage: python -m scripts.exp1_chatbotarena_gemini_audio_audio.py --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-audio-gemini1.5flash_BA.jsonl --order ba

    # usage: python -m scripts.exp1_chatbotarena_gemini_audio_audio --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-audio-gemini2.0flash.jsonl --order ab
    # usage: python -m scripts.exp1_chatbotarena_gemini_audio_audio --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-audio-gemini2.0flash_BA.jsonl --order ba

    # usage: python -m scripts.exp1_chatbotarena_gemini_audio_audio --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-audio-gemini2.5flash.jsonl --order ab
    # usage: python -m scripts.exp1_chatbotarena_gemini_audio_audio --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-audio-gemini2.5flash_BA.jsonl --order ba


if __name__ == "__main__":
    main()
