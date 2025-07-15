import os
import argparse
from glob import glob
from pydub import AudioSegment
from google.cloud import speech


def transcribe_file(
    client, audio_file: str, sample_rate: int
) -> speech.RecognizeResponse:
    """Transcribe the given audio file.
    Args:
        audio_file (str): Path to the local audio file to be transcribed.
            Example: "resources/audio.wav"
    Returns:
        cloud_speech.RecognizeResponse: The response containing the transcription results
    """

    with open(audio_file, "rb") as f:
        audio_content = f.read()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,  # 16000
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(f"Transcript: {result.alternatives[0].transcript}")

    return response


def experiment(
    input_dir: str,
    text_dir: str,
):
    print("-------------------------------------------")
    print("input_dir:", input_dir)
    print("text_dir:", text_dir)
    print("-------------------------------------------")

    client = speech.SpeechClient()

    wav_paths = sorted(glob(f"{input_dir}/*.wav"))
    for wav_path in wav_paths:
        id = int(wav_path.replace(input_dir, "").split(".")[0].strip("/"))
        txt_path = f"{text_dir}/{id}.txt"
        # check if the transcript file already exists
        if os.path.exists(txt_path):
            print(f"Skipping {id}")
            continue

        assert os.path.exists(wav_path)
        sample_rate = AudioSegment.from_wav(wav_path).frame_rate
        response = transcribe_file(client, wav_path, sample_rate)
        text = response.results[0].alternatives[0].transcript.strip()
        with open(txt_path, "w") as f:
            f.write(text)
    print("Done.")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the input dir containing files to transcribe. (*.wav files)",
    )
    parser.add_argument(
        "--text_dir",
        type=str,
        help="Path to the output file to save the ASR results. (*.txt files)",
    )
    args = parser.parse_args()
    experiment(args.input_dir, args.text_dir)

    # usage: python asr_google_generation.py --input_dir data/questions1_kokoro_wav --text_dir  experiments/advvoiceq1/asr_google/transcript
