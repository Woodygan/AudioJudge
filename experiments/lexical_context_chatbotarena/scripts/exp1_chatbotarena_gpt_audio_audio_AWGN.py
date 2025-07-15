import os
import json
import argparse
import base64
import io
import numpy as np
import soundfile as sf
import hashlib
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie (which can be equally good and equally bad). Note that the user question and the responses of the assistants will be provided to you in the audio format. You should evaluate the responses based on the user question and not on the responses of the other assistant. You should also not consider the quality of the audio or the voice of the assistants. You should only consider the content of the responses."""


def get_encoded_audio_from_path(filepath):
    """
    Reads a WAV file and returns the audio as a base64-encoded string.
    Args:
        filepath (str): Path to the WAV file.

    Returns:
        str: Base64-encoded string of the noisy WAV data.
    """
    # 1. Read audio data with soundfile
    audio, sr = sf.read(filepath)  # audio is a NumPy array, sr is the sample rate

    # 2. Write the audio to an in-memory buffer as WAV
    mem_buffer = io.BytesIO()
    sf.write(mem_buffer, audio, sr, format="WAV")
    mem_buffer.seek(0)  # reset buffer position to the beginning

    # 3. Encode the in-memory WAV data as base64
    encoded = base64.b64encode(mem_buffer.read()).decode("utf-8")

    return encoded


def get_encoded_audio_from_path_with_additive_gaussian_noise(filepath, snr_db):
    """
    Reads a WAV file, adds additive white Gaussian noise at the specified SNR (in dB),
    and returns the noisy audio as a base64-encoded string.

    Args:
        filepath (str): Path to the WAV file.
        snr_db (float): Desired signal-to-noise ratio in decibels.

    Returns:
        str: Base64-encoded string of the noisy WAV data.
    """
    # 1. Read audio data with soundfile
    audio, sr = sf.read(filepath)  # audio is a NumPy array, sr is the sample rate

    # 2. Compute signal power
    signal_power = np.mean(audio**2)

    # 3. Convert SNR from dB to linear
    snr_linear = 10 ** (snr_db / 10.0)

    # 4. Determine required noise power for the desired SNR
    noise_power = signal_power / snr_linear

    # 5. Generate white Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)

    # 6. Add noise to the original audio
    noisy_audio = audio + noise

    # 7. Write the noisy audio to an in-memory buffer as WAV
    mem_buffer = io.BytesIO()
    sf.write(mem_buffer, noisy_audio, sr, format="WAV")
    mem_buffer.seek(0)  # reset buffer position to the beginning

    # 8. Encode the in-memory WAV data as base64
    encoded_noisy = base64.b64encode(mem_buffer.read()).decode("utf-8")

    return encoded_noisy


def get_encoder_audio(filepath, snr_db):
    """
    Reads a WAV file and returns the audio as a base64-encoded string.
    """
    if snr_db > 99:
        return get_encoded_audio_from_path(filepath)
    else:
        return get_encoded_audio_from_path_with_additive_gaussian_noise(
            filepath, snr_db
        )


# For sanity check
# def save_wav(encoded_audio_str, output_wav_path):
#     """
#     Saves a Base64-encoded WAV string to an actual WAV file.

#     Args:
#         encoded_audio_str (str): The Base64-encoded audio data (WAV format).
#         output_wav_path (str): Path to the WAV file to be created.
#     """
#     # Decode the base64 string to get raw WAV bytes
#     wav_bytes = base64.b64decode(encoded_audio_str)

#     # Write the raw bytes to a .wav file
#     with open(output_wav_path, "wb") as f:
#         f.write(wav_bytes)


def experiment(
    data_path: str,
    snr: int,
    output_path: str,
):
    print("-----------------------------")
    print("data_path:", data_path)
    print("snr:", snr)
    print("output_path:", output_path)
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

    HIGH_SNR = 100000000

    for i in tqdm(range(num_done, len(data))):
        # it's a tie
        hashed = hashlib.sha256(data[i]["question_id"].encode()).digest()
        bit = hashed[0] & 1
        if bit == 0:
            # noise added to model_a
            snr_model_a = snr
            snr_model_b = HIGH_SNR
        else:
            # noise added to model_b
            snr_model_a = HIGH_SNR
            snr_model_b = snr

        # question
        question_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/kokoroTTS/wav_af_bella/{data[i]['question_id']}-user.wav"
        encoded_audio_question = get_encoder_audio(question_wav_path, snr_db=HIGH_SNR)

        # assistant a
        assistant_a_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/kokoroTTS/wav_af_bella/{data[i]['question_id']}-assistant-a.wav"
        encoded_audio_responseA = get_encoder_audio(
            assistant_a_wav_path, snr_db=snr_model_a
        )

        # assistant b
        assistant_b_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/kokoroTTS/wav_af_bella/{data[i]['question_id']}-assistant-b.wav"
        encoded_audio_responseB = get_encoder_audio(
            assistant_b_wav_path, snr_db=snr_model_b
        )

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
            temperature=0.000000000001,
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
    parser.add_argument("--snr", type=int, default=0, help="Signal to Noise Ratio")
    parser.add_argument("--output_path", type=str, required=True, help="Output Path")
    args = parser.parse_args()
    experiment(args.data_path, args.snr, args.output_path)

    # No tie subset
    # python -m scripts.exp1_chatbotarena_gpt_audio_audio_AWGN.py --data_path data/chatbot-arena-spoken-1turn-english-subset1000.json  --snr 100000 --output_path experiments/chatbot-arena-subset1000-temp0.0000/audio-audio-gpt4o-awgn-snr100000.jsonl

    # python -m scripts.exp1_chatbotarena_gpt_audio_audio_AWGN.py --data_path data/chatbot-arena-spoken-1turn-english-subset1000.json  --snr 30 --output_path experiments/chatbot-arena-subset1000-temp0.0000/audio-audio-gpt4o-awgn-snr30.jsonl

    # python -m scripts.exp1_chatbotarena_gpt_audio_audio_AWGN.py --data_path data/chatbot-arena-spoken-1turn-english-subset1000.json  --snr 20 --output_path experiments/chatbot-arena-subset1000-temp0.0000/audio-audio-gpt4o-awgn-snr20.jsonl

    # python -m scripts.exp1_chatbotarena_gpt_audio_audio_AWGN.py --data_path data/chatbot-arena-spoken-1turn-english-subset1000.json  --snr 10 --output_path experiments/chatbot-arena-subset1000-temp0.0000/audio-audio-gpt4o-awgn-snr10.jsonl

    # python -m scripts.exp1_chatbotarena_gpt_audio_audio_AWGN.py --data_path data/chatbot-arena-spoken-1turn-english-subset1000.json  --snr 5 --output_path experiments/chatbot-arena-subset1000-temp0.0000/audio-audio-gpt4o-awgn-snr5.jsonl

    # python -m scripts.exp1_chatbotarena_gpt_audio_audio_AWGN.py --data_path data/chatbot-arena-spoken-1turn-english-subset1000.json  --snr 1 --output_path experiments/chatbot-arena-subset1000-temp0.0000/audio-audio-gpt4o-awgn-snr1.jsonl

    # python -m scripts.exp1_chatbotarena_gpt_audio_audio_AWGN.py --data_path data/chatbot-arena-spoken-1turn-english-subset1000.json  --snr 40 --output_path experiments/chatbot-arena-subset1000-temp0.0000/audio-audio-gpt4o-awgn-snr40.jsonl

    # python -m scripts.exp1_chatbotarena_gpt_audio_audio_AWGN.py --data_path data/chatbot-arena-spoken-1turn-english-subset1000.json  --snr 500 --output_path experiments/chatbot-arena-subset1000-temp0.0000/audio-audio-gpt4o-awgn-snr500.jsonl


if __name__ == "__main__":
    main()
