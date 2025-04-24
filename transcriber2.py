import os
import torch
import pandas as pd
from faster_whisper import WhisperModel # Use correct class name
from pyannote.audio import Pipeline
import gc
import time # To measure execution time

# --- Configuration ---
# Paths inside the container
audio_path = "/app/audio_input/salemapril15.mp3" # Point to your specific audio file
hf_token_file = "/app/hf-token.txt"
output_dir = "/app/transcripts_output"

os.makedirs(output_dir, exist_ok=True)

# --- Device Settings ---
# Use CUDA for Pyannote if available, but force CPU for faster-whisper
pyannote_device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_device = "cpu" # Force CPU for faster-whisper
# Choose a compute type compatible with CPU, int8 is often efficient on CPU
whisper_compute_type = "int8"
whisper_model_name = "large-v3" # or "large-v2" etc.
print(f"Using device for Whisper: {whisper_device}, compute_type: {whisper_compute_type}")
print(f"Using device for Pyannote: {pyannote_device}")

# --- Load Hugging Face Token ---
try:
    with open(hf_token_file, 'r') as f:
        hf_token = f.read().strip()
    if not hf_token:
        raise ValueError("Token file is empty.")
    print("Hugging Face token loaded successfully.")
except FileNotFoundError:
    print(f"Error: Hugging Face token file not found at {hf_token_file}")
    exit()
except Exception as e:
    print(f"Error reading Hugging Face token file: {e}")
    exit()

# --- 1. Transcription with faster-whisper on CPU ---
start_time_transcribe = time.time()
print(f"Loading faster-whisper model: {whisper_model_name}...")
try:
    # Load the faster-whisper model ON CPU
    transcription_model = WhisperModel(
        whisper_model_name,
        device=whisper_device, # Set to "cpu"
        compute_type=whisper_compute_type # e.g., "int8"
    )
except Exception as e:
    print(f"Error loading faster-whisper model: {e}")
    exit()

print(f"Transcribing audio on CPU: {audio_path} (this will be slow)...")
transcript_results = []
try:
    # Get word timestamps directly from faster-whisper if needed
    segments, info = transcription_model.transcribe(audio_path, beam_size=5, word_timestamps=True)

    print(f"Detected language '{info.language}' with probability {info.language_probability}")
    print(f"Transcription duration: {info.duration}s")

    # Process segments (generator) - Convert to list for easier handling
    for segment in segments:
        segment_dict = {"start": segment.start, "end": segment.end, "text": segment.text, "words": []}
        if segment.words:
             for word in segment.words:
                  segment_dict["words"].append({
                       "word": word.word,
                       "start": word.start,
                       "end": word.end,
                       "probability": word.probability
                  })
        transcript_results.append(segment_dict)

except Exception as e:
    print(f"Error during transcription: {e}")
    exit()

end_time_transcribe = time.time()
print(f"Transcription complete. Time taken: {end_time_transcribe - start_time_transcribe:.2f}s")

# Save transcription results
transcript_df = pd.DataFrame(transcript_results)
transcript_df.to_json(os.path.join(output_dir, "transcript_word_timestamps.json"), orient="records", indent=2)
print(f"Transcription with word timestamps saved to {output_dir}/transcript_word_timestamps.json")


# --- Cleanup faster-whisper model ---
print("Unloading faster-whisper model...")
del transcription_model
gc.collect()
# No torch.cuda.empty_cache() needed as model was on CPU


# --- 2. Diarization with pyannote.audio (attempting GPU) ---
start_time_diarize = time.time()
print("Loading pyannote.audio diarization pipeline...")
diarization_pipeline = None
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    # Try sending pyannote to CUDA
    diarization_pipeline.to(torch.device(pyannote_device))
    print(f"Pyannote pipeline loaded on {pyannote_device}")
except Exception as e:
    print(f"Error loading diarization pipeline: {e}")
    # If this fails specifically with CUDA/cuDNN errors,
    # you might need to change pyannote_device to "cpu" above.
    exit()

print(f"Diarizing audio: {audio_path} (this may take a while)...")
diarization_results = []
try:
    diarization = diarization_pipeline(audio_path, min_speakers=None, max_speakers=None)

    # Process diarization results into a list of dicts
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_results.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

except Exception as e:
    print(f"Error during diarization: {e}")
    exit()

end_time_diarize = time.time()
print(f"Diarization complete. Time taken: {end_time_diarize - start_time_diarize:.2f}s")

# Save diarization results
if diarization_results:
    diarization_df = pd.DataFrame(diarization_results)
    diarization_df.to_csv(os.path.join(output_dir, "diarization_turns.csv"), index=False)
    print(f"Diarization turns saved to {output_dir}/diarization_turns.csv")
else:
    print("No speaker turns detected by diarization pipeline.")


# --- Cleanup pyannote model ---
print("Unloading diarization pipeline...")
del diarization_pipeline
gc.collect()
if pyannote_device == "cuda":
    torch.cuda.empty_cache()


# --- 3. Combine Results (Basic Example - Assign Speaker to Words) ---
print("Combining transcription and diarization results (assigning speaker to words)...")

# Ensure we have both results to combine
if transcript_results and diarization_results:
    # Create a structure mapping time to speaker
    speaker_map = []
    for turn in diarization_results:
        speaker_map.append((turn['start'], turn['end'], turn['speaker']))

    # Sort by start time just in case
    speaker_map.sort()

    # Assign speaker to each word based on word midpoint time
    for segment in transcript_results:
        if 'words' in segment and segment['words']:
            for word_info in segment['words']:
                word_midpoint = word_info['start'] + (word_info['end'] - word_info['start']) / 2
                assigned_speaker = "UNKNOWN"
                for turn_start, turn_end, speaker in speaker_map:
                    if turn_start <= word_midpoint < turn_end:
                        assigned_speaker = speaker
                        break
                word_info['speaker'] = assigned_speaker
    print("Speaker assignment to words complete (basic overlap method).")

    # Save the final combined data structure
    final_combined_df = pd.DataFrame(transcript_results)
    final_combined_df.to_json(os.path.join(output_dir, "final_combined_word_level.json"), orient="records", indent=2)
    print(f"Combined results saved to {output_dir}/final_combined_word_level.json")

else:
    print("Skipping combination step as transcription or diarization results are missing.")

print("Script finished.")
