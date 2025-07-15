import os
import wave
from pathlib import Path


def check_wav_header(file_path):
    """
    Check if the WAV file has correct header information
    """
    try:
        with wave.open(str(file_path), "rb") as wav:
            params = wav.getparams()
            frames = wav.getnframes()

            # Calculate expected frames based on file size
            file_size = os.path.getsize(file_path)
            data_size = file_size - 44  # Standard WAV header size
            frame_size = params.nchannels * params.sampwidth
            expected_frames = data_size // frame_size if frame_size > 0 else 0

            # Check if frames match expected value (allow small difference)
            if abs(frames - expected_frames) > 5:
                print(
                    f"{file_path}: ISSUE - nframes: {frames}, expected: {expected_frames}"
                )
                return False
            else:
                print(
                    f"{file_path}: OK - {params.nchannels} channels, {params.framerate}Hz, {frames} frames"
                )
                return True
    except Exception as e:
        print(f"{file_path}: ERROR - {e}")
        return False


def verify_directory(directory_path):
    """
    Verify all WAV files in a directory
    """
    dir_path = Path(directory_path)
    ok_count = 0
    issue_count = 0
    error_count = 0

    for file_path in sorted(dir_path.glob("*.wav")):
        try:
            if check_wav_header(file_path):
                ok_count += 1
            else:
                issue_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            error_count += 1

    return ok_count, issue_count, error_count


if __name__ == "__main__":
    # Get paths relative to script location
    script_dir = Path(__file__).parent.absolute()  # Get directory of this script
    project_root = script_dir.parent  # Go up one level to project root
    directory = project_root / "src" / "audiojudge" / "signal_audios"

    ok_count, issue_count, error_count = verify_directory(directory)
    total = ok_count + issue_count + error_count

    print(f"\nVerification Summary for {directory}:")
    print(f"  Total files:  {total}")
    print(f"  OK files:     {ok_count} ({ok_count / total * 100:.1f}%)")
    print(f"  Issue files:  {issue_count} ({issue_count / total * 100:.1f}%)")
    print(f"  Error files:  {error_count} ({error_count / total * 100:.1f}%)")
