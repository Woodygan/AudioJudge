import os
from pathlib import Path
from pydub import AudioSegment

# Get paths relative to script location
script_dir = Path(__file__).parent.absolute()  # Get directory of this script
project_root = script_dir.parent  # Go up one level to project root
signal_audios_dir = project_root / "src" / "audiojudge" / "signal_audios"

# List of files that had "file does not start with RIFF id" errors
problematic_files = [
    signal_audios_dir / "example_0.wav",
    signal_audios_dir / "first_audio.wav",
    signal_audios_dir / "second_audio.wav",
    signal_audios_dir / "test_audio.wav",
]


def fix_non_riff_wav(file_path):
    """
    Try to fix a WAV file that doesn't have a RIFF header by using pydub
    to read and re-export it.
    """
    try:
        print(f"Processing {file_path}...")

        # Create a backup of the original file
        backup_path = str(file_path) + ".backup"
        os.rename(file_path, backup_path)
        print(f"  Created backup at {backup_path}")

        # Try to load the file with pydub (which is more forgiving with file formats)
        try:
            audio = AudioSegment.from_file(backup_path)
            print(
                f"  Successfully loaded with pydub: {audio.channels} channels, {audio.frame_rate}Hz, {audio.duration_seconds:.2f}s"
            )

            # Export as a proper WAV file
            audio.export(file_path, format="wav")
            print(f"  Fixed and saved to {file_path}")
            return True
        except Exception as e:
            # If pydub fails, restore from backup
            print(f"  Error: {e}")
            os.rename(backup_path, file_path)
            print(f"  Restored original file from backup")
            return False
    except Exception as e:
        print(f"  Error processing file: {e}")
        return False


if __name__ == "__main__":
    fixed_count = 0
    for file_path in problematic_files:
        if os.path.exists(file_path):
            if fix_non_riff_wav(file_path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")

    print(f"\nFixed {fixed_count} out of {len(problematic_files)} problematic files")
