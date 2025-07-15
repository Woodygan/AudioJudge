import argparse
import soundfile as sf
import json
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm


def transcribe_audio(model, processor, audio_path):
    audio_array, sr = sf.read(
        audio_path
    )  # audio is a NumPy array, sr is the sample rate
    input_features = processor(
        audio_array, sampling_rate=sr, return_tensors="pt"
    ).input_features
    input_features = input_features.to(model.device)

    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    # [' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']
    return transcription[0].strip()


def experiment(data_path, output_path):
    # Load the data
    with open(data_path) as f:
        data = json.load(f)
    print("len(data):", len(data))

    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model.config.forced_decoder_ids = None
    model.cuda()

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
        question_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/kokoroTTS/wav/{data[i]['question_id']}-user.wav"
        assistant_a_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/kokoroTTS/wav/{data[i]['question_id']}-assistant-a.wav"
        assistant_b_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/kokoroTTS/wav/{data[i]['question_id']}-assistant-b.wav"

        question_transcription = transcribe_audio(model, processor, question_wav_path)
        assistant_a_transcription = transcribe_audio(
            model, processor, assistant_a_wav_path
        )
        assistant_b_transcription = transcribe_audio(
            model, processor, assistant_b_wav_path
        )
        item = {
            "i": i,
            "question_transcription": question_transcription,
            "assistant_a_transcription": assistant_a_transcription,
            "assistant_b_transcription": assistant_b_transcription,
        }
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
    args = parser.parse_args()
    experiment(args.data_path, args.output_path)

    # python whisper_asr.py --data_path ../chatbot-arena-spoken-1turn-english-difference-voices.json --output_path chatbot-arena-spoken-1turn-english-whisper-transcript.jsonl


if __name__ == "__main__":
    main()
