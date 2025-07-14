# AudioJudge Utility Tools

This directory contains utility scripts for maintenance and repair of audio files used by AudioJudge.

## Overview

AudioJudge uses WAV files for signal generation and audio testing. Sometimes these files can have corrupted metadata in their headers, which causes issues with audio concatenation and processing. The tools in this directory can help fix these issues.

## Tools

### 1. fix_wav_headers.py

Repairs WAV files with incorrect `nframes` values in their headers, which can lead to errors like `'L' format requires 0 <= number <= 4294967295`.

**Usage:**
```bash
python tools/fix_wav_headers.py
```

This script:
- Scans all WAV files in `src/audiojudge/signal_audios/`
- Detects files with corrupted headers (usually with `nframes=2147483647`)
- Calculates the correct number of frames based on file size and audio parameters
- Rewrites the files with correct headers

### 2. fix_non_riff_wavs.py

Repairs WAV files that don't have proper RIFF headers, which can cause "file does not start with RIFF id" errors.

**Usage:**
```bash
python tools/fix_non_riff_wavs.py
```

This script:
- Processes a predefined list of problematic WAV files
- Creates backups of the original files (with .backup extension)
- Uses the pydub library to read and re-export the files with proper headers
- Restores from backup if repair fails

### 3. verify_wav_headers.py

Verifies that all WAV files have correct headers after repairs.

**Usage:**
```bash
python tools/verify_wav_headers.py
```

This script:
- Scans all WAV files in `src/audiojudge/signal_audios/`
- Checks the header information against expected values based on file size
- Reports on the status of each file (OK, ISSUE, or ERROR)
- Provides a summary of the verification results

## Troubleshooting

If you encounter audio processing errors in AudioJudge, especially with errors mentioning:
- `'L' format requires 0 <= number <= 4294967295`
- `file does not start with RIFF id`

Run these tools in the recommended order:

1. `fix_wav_headers.py`
2. `fix_non_riff_wavs.py`
3. `verify_wav_headers.py`

After running these tools, your AudioJudge should work properly with the fixed audio files. 