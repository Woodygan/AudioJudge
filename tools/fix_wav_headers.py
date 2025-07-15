import os
import wave
import struct
from pathlib import Path


def fix_wav_header(file_path):
    """
    Fix WAV file headers by correcting the nframes value.

    Args:
        file_path: Path to the WAV file

    Returns:
        True if file was fixed, False if no fixing was needed
    """
    # First, check if the file has incorrect nframes
    with wave.open(str(file_path), "rb") as wav:
        params = wav.getparams()
        print(f"File: {file_path}")
        print(f"  Original params: {params}")

        # If nframes is incorrect (INT_MAX or larger than file size would allow)
        if params.nframes == 2147483647:
            # Calculate correct nframes from file size
            file_size = os.path.getsize(file_path)
            data_size = file_size - 44  # 44 bytes for standard WAV header
            frame_size = params.nchannels * params.sampwidth
            if frame_size > 0:
                correct_nframes = data_size // frame_size

                # Create temporary file path
                temp_path = str(file_path) + ".temp"

                # Read audio data
                wav.setpos(0)
                audio_data = wav.readframes(wav.getnframes())

                # Write corrected WAV file
                with wave.open(temp_path, "wb") as fixed_wav:
                    fixed_wav.setparams(params._replace(nframes=correct_nframes))
                    fixed_wav.writeframes(audio_data)

                # Replace original with fixed file
                os.replace(temp_path, file_path)

                print(f"  Fixed! New nframes: {correct_nframes}")
                return True
            else:
                print(f"  Error: Frame size is zero!")
                return False
        else:
            print(f"  No fix needed, nframes: {params.nframes}")
            return False


def fix_wav_directory(directory_path):
    """
    Fix all WAV files in a directory.

    Args:
        directory_path: Path to directory containing WAV files

    Returns:
        Number of files fixed
    """
    fixed_count = 0
    dir_path = Path(directory_path)

    for file_path in dir_path.glob("*.wav"):
        try:
            if fix_wav_header(file_path):
                fixed_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return fixed_count


if __name__ == "__main__":
    # Fix all WAV files in the signal_audios directory
    script_dir = Path(__file__).parent.absolute()  # Get directory of this script
    project_root = script_dir.parent  # Go up one level to project root
    directory = project_root / "src" / "audiojudge" / "signal_audios"
    fixed_count = fix_wav_directory(directory)
    print(f"\nFixed {fixed_count} files in {directory}")
