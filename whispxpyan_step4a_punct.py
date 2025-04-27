# whispxpyan_step4a_punct.py
import os
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import gc
import time
import json
import faiss
import numpy as np
from speechbrain.inference.classifiers import EncoderClassifier
import torchaudio
from collections import defaultdict
import argparse
from transformers import pipeline # Added for punctuation

# ==============================================================
# Configuration (Defaults - Can be overridden by CLI args)
# ==============================================================
DEFAULT_OUTPUT_DIR = "/app/transcripts_output"
DEFAULT_HF_TOKEN_FILE = "/app/hf-token.txt"
DEFAULT_FAISS_INDEX_FILENAME = "speaker_embeddings.index"
DEFAULT_SPEAKER_MAP_FILENAME = "speaker_names.json"
DEFAULT_IDENTIFIED_JSON_FILENAME = "intermediate_identified_transcript.json" # Renamed intermediate
DEFAULT_PUNCTUATED_TXT_FILENAME = "final_punctuated_transcript.txt" # New final output

DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_WHISPER_DEVICE = "cpu"
DEFAULT_WHISPER_COMPUTE = "int8"

DEFAULT_PROCESSING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_PUNCTUATION_MODEL = "felflare/bert-restore-punctuation"
DEFAULT_SIMILARITY_THRESHOLD = 0.80
DEFAULT_MIN_SEGMENT_DURATION = 2.0
DEFAULT_PUNCTUATION_CHUNK_SIZE = 256 # Max words per chunk for punctuation model (limits memory)

# ==============================================================
# Helper Functions (Unchanged: FAISS/Map Load/Save, Audio Extraction, Embeddings, Identification, Enrollment)
# ==============================================================
# Assume functions load/save_faiss_index, load/save_speaker_map,
# extract_speaker_audio_segments, get_speaker_embeddings, identify_speakers,
# update_transcript_speakers, enroll_new_speakers_cli are here and correct
# from the previous step (whispxpyan_step3_cli.py)
# ... (Include the definitions for these helper functions here) ...
# --- [PASTE HELPER FUNCTIONS FROM PREVIOUS SCRIPT HERE] ---
def load_or_create_faiss_index(index_path, dimension):
    """Loads a FAISS index from disk or creates a new one if not found."""
    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from {index_path}")
        try:
            index = faiss.read_index(index_path)
            if index.d != dimension:
                 print(f"Warning: Index dimension ({index.d}) differs from model dimension ({dimension}). Creating new index.")
                 index = faiss.IndexFlatIP(dimension) # Inner Product for cosine similarity
            else:
                print(f"FAISS index loaded with {index.ntotal} embeddings.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Creating a new one.")
            index = faiss.IndexFlatIP(dimension)
    else:
        print(f"FAISS index not found at {index_path}. Creating a new one.")
        index = faiss.IndexFlatIP(dimension) # Use Inner Product (IP) for cosine similarity
    return index

def save_faiss_index(index, index_path):
    """Saves the FAISS index to disk."""
    print(f"Saving FAISS index to {index_path} with {index.ntotal} embeddings...")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        print("FAISS index saved successfully.")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

def load_or_create_speaker_map(map_path):
    """Loads the speaker name map (FAISS ID -> Name) from JSON or creates an empty one."""
    if os.path.exists(map_path):
        print(f"Loading speaker map from {map_path}")
        try:
            with open(map_path, 'r') as f:
                speaker_map = json.load(f)
            # Ensure keys are integers if loaded from JSON
            speaker_map = {int(k): v for k, v in speaker_map.items()}
            print(f"Speaker map loaded with {len(speaker_map)} entries.")
        except Exception as e:
            print(f"Error loading speaker map: {e}. Creating a new one.")
            speaker_map = {}
    else:
        print(f"Speaker map not found at {map_path}. Creating a new one.")
        speaker_map = {} # Maps FAISS index ID (int) to speaker name (str)
    return speaker_map

def save_speaker_map(speaker_map, map_path):
    """Saves the speaker name map to JSON."""
    print(f"Saving speaker map to {map_path} with {len(speaker_map)} entries...")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        # Ensure keys are strings for JSON compatibility
        save_map = {str(k): v for k, v in speaker_map.items()}
        with open(map_path, 'w') as f:
            json.dump(save_map, f, indent=2)
        print("Speaker map saved successfully.")
    except Exception as e:
        print(f"Error saving speaker map: {e}")

def extract_speaker_audio_segments(transcript_data, audio_waveform, sample_rate, min_segment_duration, target_sr=16000):
    """Extracts audio segments for each generic speaker."""
    speaker_segments = defaultdict(list)
    total_duration = defaultdict(float)

    if sample_rate != target_sr:
        print(f"Resampling audio from {sample_rate} Hz to {target_sr} Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        audio_waveform = resampler(audio_waveform)
        sample_rate = target_sr
        print(f"Resampling complete.")
    else:
        print(f"Audio already at target sample rate {target_sr} Hz.")

    print("Extracting audio segments for each speaker...")
    processed_words = 0
    skipped_words = 0
    for segment in transcript_data:
        if 'words' in segment and segment['words']:
            for word_info in segment['words']:
                speaker = word_info.get('speaker')
                if speaker and speaker.startswith("SPEAKER_"):
                    start_time = word_info.get('start')
                    end_time = word_info.get('end')
                    if start_time is not None and end_time is not None and end_time > start_time:
                        start_sample = int(start_time * sample_rate)
                        end_sample = int(end_time * sample_rate)
                        if end_sample <= audio_waveform.shape[1]:
                           if audio_waveform.shape[0] > 1:
                               audio_chunk = torch.mean(audio_waveform[:, start_sample:end_sample], dim=0, keepdim=True)
                           else:
                               audio_chunk = audio_waveform[:, start_sample:end_sample]
                           speaker_segments[speaker].append(audio_chunk)
                           total_duration[speaker] += (end_time - start_time)
                           processed_words +=1
                        else:
                            skipped_words += 1
                    else:
                       skipped_words += 1

    print(f"Audio segment extraction complete (Processed: {processed_words}, Skipped Invalid: {skipped_words}).")
    filtered_segments = {}
    for speaker, duration in total_duration.items():
        if duration >= min_segment_duration:
            try:
                concatenated_audio = torch.cat(speaker_segments[speaker], dim=1)
                filtered_segments[speaker] = concatenated_audio
                print(f"  - Speaker {speaker}: {duration:.2f}s audio (sufficient).")
            except Exception as e:
                 print(f"  - Speaker {speaker}: Error concatenating audio segments: {e}. Skipping.")
        else:
            print(f"  - Speaker {speaker}: {duration:.2f}s audio (insufficient, skipping embedding).")
    return filtered_segments, sample_rate

def get_speaker_embeddings(speaker_audio_segments, embedding_model, sample_rate, device):
    """Generates speaker embeddings using the SpeechBrain model."""
    embeddings = {}
    if not speaker_audio_segments:
        print("No sufficient audio segments found to generate embeddings.")
        return embeddings
    print("Generating speaker embeddings...")
    with torch.no_grad():
        for speaker, audio_tensor in speaker_audio_segments.items():
            print(f"  - Processing {speaker}...")
            try:
                audio_tensor = audio_tensor.to(device)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                elif audio_tensor.shape[0] > 1:
                    print(f"    Warning: Audio tensor for {speaker} still has multiple channels ({audio_tensor.shape[0]}).")
                    audio_tensor = audio_tensor[0, :].unsqueeze(0)
                embedding = embedding_model.encode_batch(audio_tensor)
                embedding = embedding.squeeze()
                norm = torch.linalg.norm(embedding)
                if norm > 1e-6:
                   embedding = embedding / norm
                else:
                   print(f"    Warning: Embedding norm for {speaker} is close to zero. Skipping normalization.")
                embeddings[speaker] = embedding.cpu().numpy()
                print(f"    Generated embedding for {speaker} (shape: {embeddings[speaker].shape})")
            except Exception as e:
                print(f"    Error generating embedding for {speaker}: {e}")
                import traceback; traceback.print_exc()
    print("Embedding generation complete.")
    return embeddings

def identify_speakers(embeddings, faiss_index, speaker_map, similarity_threshold):
    """Identifies speakers by comparing embeddings against the FAISS index."""
    speaker_assignments = {}
    new_speaker_embeddings_info = []
    unknown_speaker_count = 0
    print(f"Starting identification. Index size: {faiss_index.ntotal}, Known speakers in map: {len(speaker_map)}")
    if faiss_index.ntotal == 0:
        print("FAISS index is empty. All detected speakers will be marked as unknown.")
        for speaker, embedding in embeddings.items():
             unknown_speaker_count += 1
             temp_id = f"UNKNOWN_SPEAKER_{unknown_speaker_count}"
             speaker_assignments[speaker] = temp_id
             new_speaker_embeddings_info.append({'temp_id': temp_id, 'embedding': embedding, 'original_label': speaker})
    else:
        print("Identifying speakers using FAISS index...")
        for speaker, embedding in embeddings.items():
            try:
                query_embedding = np.expand_dims(embedding.astype(np.float32), axis=0)
                distances, indices = faiss_index.search(query_embedding, k=1)
                faiss_id = indices[0][0]
                similarity = distances[0][0]
                print(f"  - Speaker {speaker}: Closest match FAISS ID {faiss_id}, Similarity: {similarity:.4f}")
                if similarity >= similarity_threshold and faiss_id >= 0 :
                    if faiss_id in speaker_map:
                        identified_name = speaker_map[faiss_id]
                        speaker_assignments[speaker] = identified_name
                        print(f"    Identified as: '{identified_name}' (Similarity: {similarity:.4f} >= {similarity_threshold})")
                    else:
                        print(f"    Warning: FAISS ID {faiss_id} found but missing in speaker map. Treating as unknown.")
                        unknown_speaker_count += 1
                        temp_id = f"UNKNOWN_SPEAKER_{unknown_speaker_count}"
                        speaker_assignments[speaker] = temp_id
                        new_speaker_embeddings_info.append({'temp_id': temp_id, 'embedding': embedding, 'original_label': speaker})
                else:
                    print(f"    Similarity {similarity:.4f} < {similarity_threshold} (or no match). Marking as unknown.")
                    unknown_speaker_count += 1
                    temp_id = f"UNKNOWN_SPEAKER_{unknown_speaker_count}"
                    speaker_assignments[speaker] = temp_id
                    new_speaker_embeddings_info.append({'temp_id': temp_id, 'embedding': embedding, 'original_label': speaker})
            except Exception as e:
                print(f"    Error identifying speaker {speaker}: {e}")
                unknown_speaker_count += 1
                speaker_assignments[speaker] = f"UNKNOWN_SPEAKER_{unknown_speaker_count}" # Fallback
    print("Speaker identification complete.")
    return speaker_assignments, new_speaker_embeddings_info

def update_transcript_speakers(transcript_data, speaker_assignments):
    """Updates the transcript data with identified speaker names or temp IDs."""
    print("Updating transcript with identified speaker names...")
    updated_data = []
    for segment in transcript_data:
        updated_segment = segment.copy()
        if 'words' in segment and segment['words']:
            updated_words = []
            for word_info in segment['words']:
                updated_word_info = word_info.copy()
                generic_speaker = word_info.get('speaker')
                if generic_speaker and generic_speaker in speaker_assignments:
                    updated_word_info['speaker'] = speaker_assignments[generic_speaker]
                elif generic_speaker and (not generic_speaker.startswith("SPEAKER_")):
                     pass
                elif generic_speaker and generic_speaker.startswith("SPEAKER_"):
                    updated_word_info['speaker'] = f"UNPROCESSED_{generic_speaker}"
                updated_words.append(updated_word_info)
            updated_segment['words'] = updated_words
        updated_data.append(updated_segment)
    print("Transcript update complete.")
    return updated_data

def enroll_new_speakers_cli(new_speaker_info, faiss_index, speaker_map, faiss_index_path, speaker_map_path):
    """Handles interactive enrollment of new speakers via the CLI."""
    if not new_speaker_info:
        print("\nNo new speakers detected requiring enrollment in this run.")
        return False # Indicate no changes made
    print("\n--- Speaker Enrollment ---")
    changes_made = False
    new_speaker_info.sort(key=lambda x: int(x['temp_id'].split('_')[-1]))
    for speaker_data in new_speaker_info:
        temp_id = speaker_data['temp_id']
        embedding = speaker_data['embedding']
        original_label = speaker_data.get('original_label', 'N/A')
        prompt = f"Enroll speaker {temp_id} (originally {original_label})? Enter name (or leave blank to skip): "
        try:
            entered_name = input(prompt).strip()
        except EOFError:
            print("\nEOF detected, skipping enrollment.")
            break
        if entered_name:
            try:
                faiss_index.add(np.expand_dims(embedding.astype(np.float32), axis=0))
                new_faiss_id = faiss_index.ntotal - 1
                speaker_map[new_faiss_id] = entered_name
                print(f"  -> Enrolled '{entered_name}' with FAISS ID {new_faiss_id}.")
                changes_made = True
            except Exception as e:
                 print(f"  -> Error enrolling {temp_id} as '{entered_name}': {e}")
        else:
            print(f"  -> Skipping enrollment for {temp_id}.")
    if changes_made:
        print("\nSaving updated speaker database...")
        save_faiss_index(faiss_index, faiss_index_path)
        save_speaker_map(speaker_map, speaker_map_path)
        return True # Indicate changes made
    else:
        print("\nNo changes made to speaker database.")
        return False
# --- [END OF PASTED HELPER FUNCTIONS] ---

# ==============================================================
# NEW: Punctuation Restoration Function
# ==============================================================
def format_punctuated_output(results):
    """Reconstructs text from the punctuation pipeline output."""
    text = ''
    for i, word_info in enumerate(results):
        word = word_info['word']
        label = word_info['entity'] # The label from the model

        # Handle beginning of the text
        if i == 0:
            # If the model predicts capitalization or it's the start, capitalize
            if label == 'Upper' or label == 'O': # Heuristic: capitalize first word if unsure
                 word = word.capitalize()
            text += word
            continue # Skip punctuation check for first word

        # Check previous word's info if needed (useful for space handling)
        prev_word_info = results[i-1]

        # Handle punctuation based on label
        if label.startswith('PUNC'):
            punc = label.split('_')[-1]
            # Handle cases where the token itself might be the punctuation (depends on model tokenizer)
            if word == punc:
                text += punc # Append only punctuation
            else:
                # Apply punctuation to the *previous* word if applicable (e.g. '.' or ',')
                 if punc in ['.', ',', '?', '!', ':', ';']:
                    # Avoid double punctuation if previous word already ended with one
                    if text and not text[-1] in ['.', ',', '?', '!', ':', ';']:
                       text += punc + ' ' + word
                    else:
                       text += ' ' + word # Just add space and word
                 elif punc == '-': # Handle hyphens - attach without space? Context dependent.
                     text += '-' + word
                 else: # Default: space then word (might need refinement)
                     text += ' ' + word

        elif label == 'Upper':
            # Capitalize the current word and add preceding space
             text += ' ' + word.capitalize()
        elif label == 'O': # 'O' usually means no specific punctuation/capitalization
             # Simple space separation
             text += ' ' + word
        else: # Unknown label, fallback to space separation
             print(f"Warning: Unknown punctuation label '{label}' for word '{word}'. Applying default spacing.")
             text += ' ' + word

    return text.strip()


def apply_punctuation(transcript_data, punctuation_pipeline, chunk_size=256):
    """Applies punctuation to the transcript, processing speaker turns."""
    print("\n--- Applying Punctuation ---")
    punctuated_turns = [] # List to store (speaker, punctuated_text) tuples
    current_speaker = None
    current_words = []

    total_words_processed = 0

    # Iterate through segments and words to group by speaker turns
    for segment in transcript_data:
        if 'words' not in segment or not segment['words']:
            continue

        for word_info in segment['words']:
            speaker = word_info.get('speaker', 'UNKNOWN') # Use final assigned speaker
            word = word_info.get('word', '').strip()
            if not word: # Skip empty words
                continue

            # If speaker changes, process the previous turn
            if speaker != current_speaker and current_words:
                print(f"Processing turn for {current_speaker} ({len(current_words)} words)")
                full_turn_text = ""
                # Process in chunks to avoid exceeding model limits
                for i in range(0, len(current_words), chunk_size):
                    chunk_words = current_words[i:i+chunk_size]
                    raw_text_chunk = " ".join(chunk_words)
                    try:
                        # Pipeline might handle tokenization details internally
                        # We pass the raw string chunk
                        # Adjust aggregation strategy if needed (e.g., add_special_tokens=False)
                        processed_results = punctuation_pipeline(raw_text_chunk)
                        # Helper function to reconstruct string from pipeline output
                        punctuated_chunk = format_punctuated_output(processed_results)
                        full_turn_text += punctuated_chunk + " " # Add space between chunks
                        total_words_processed += len(chunk_words)
                    except Exception as e:
                        print(f"  Error processing chunk for {current_speaker}: {e}. Skipping chunk.")
                        # Optionally add the raw chunk back?
                        # full_turn_text += raw_text_chunk + " "

                punctuated_turns.append((current_speaker, full_turn_text.strip()))
                current_words = [] # Reset for new speaker

            # Start new turn or continue current one
            current_speaker = speaker
            current_words.append(word)

    # Process the very last turn
    if current_words:
        print(f"Processing final turn for {current_speaker} ({len(current_words)} words)")
        full_turn_text = ""
        for i in range(0, len(current_words), chunk_size):
            chunk_words = current_words[i:i+chunk_size]
            raw_text_chunk = " ".join(chunk_words)
            try:
                processed_results = punctuation_pipeline(raw_text_chunk)
                punctuated_chunk = format_punctuated_output(processed_results)
                full_turn_text += punctuated_chunk + " "
                total_words_processed += len(chunk_words)
            except Exception as e:
                print(f"  Error processing chunk for {current_speaker}: {e}. Skipping chunk.")

        punctuated_turns.append((current_speaker, full_turn_text.strip()))

    print(f"Punctuation application complete. Processed approx {total_words_processed} words.")
    return punctuated_turns


# ==============================================================
# Main Processing Function (Modified)
# ==============================================================
def main(args):
    """Main processing pipeline."""
    start_total_time = time.time()
    # --- Construct full paths ---
    output_dir = args.output_dir
    audio_path = args.input_audio
    hf_token_file = args.hf_token
    faiss_index_path = os.path.join(output_dir, args.faiss_file)
    speaker_map_path = os.path.join(output_dir, args.map_file)
    identified_json_path = os.path.join(output_dir, args.output_json_file) # Changed name
    punctuated_txt_path = os.path.join(output_dir, args.output_punct_file) # New output

    os.makedirs(output_dir, exist_ok=True)

    # --- Device Settings ---
    # Determine the actual device to use, handling 'cuda' preference
    processing_device = args.processing_device if torch.cuda.is_available() else "cpu"
    if args.processing_device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU for processing.")
    whisper_device = args.whisper_device
    whisper_compute_type = args.whisper_compute

    print(f"Settings:")
    print(f"  Input Audio: {audio_path}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Output JSON File: {identified_json_path}")
    print(f"  Output Punctuated TXT File: {punctuated_txt_path}")
    print(f"  FAISS Index: {faiss_index_path}")
    print(f"  Speaker Map: {speaker_map_path}")
    print(f"  Whisper Device: {whisper_device}, Compute: {whisper_compute_type}, Model: {args.whisper_model}")
    print(f"  Processing Device (Pyannote/SpeechBrain/Punct): {processing_device}")
    print(f"  Embedding Model: {args.embedding_model}")
    print(f"  Punctuation Model: {args.punctuation_model}")
    print(f"  Similarity Threshold: {args.similarity_threshold}")
    print(f"  Min Segment Duration: {args.min_segment_duration}")
    print("-" * 20)

    # --- Load Hugging Face Token ---
    # ... (Token loading code as before) ...
    hf_token = None
    if hf_token_file and os.path.exists(hf_token_file):
        try:
            with open(hf_token_file, 'r') as f: hf_token = f.read().strip()
            if not hf_token: print(f"Warning: HF token file '{hf_token_file}' is empty."); hf_token = None
            else: print("Hugging Face token loaded successfully.")
        except Exception as e: print(f"Error loading HF token from {hf_token_file}: {e}"); hf_token = None
    else: print("Warning: HF token file not found or not specified.")


    # --- Initialize Models ---
    print("\nLoading models...")
    start_load_time = time.time()
    whisper_model = None
    diarization_pipeline = None
    embedding_model = None
    punctuation_pipeline_model = None # Renamed variable
    emb_dim = 192

    try: # Whisper
        whisper_model = WhisperModel(args.whisper_model, device=whisper_device, compute_type=whisper_compute_type, local_files_only=False) # Keep local_files_only=False for robustness?
        print(f"Faster Whisper model '{args.whisper_model}' loaded.")
    except Exception as e: print(f"Error loading Faster Whisper model: {e}"); return

    if hf_token: # Pyannote
        try:
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
            diarization_pipeline.to(torch.device(processing_device))
            print(f"Pyannote diarization pipeline loaded.")
        except Exception as e: print(f"Warning: Error loading Pyannote pipeline: {e}"); diarization_pipeline = None
    else: print("Skipping Pyannote loading (no token).")

    try: # SpeechBrain
        embedding_cache_dir = os.path.join(output_dir, 'embedding_model_cache')
        embedding_model = EncoderClassifier.from_hparams(source=args.embedding_model, run_opts={"device": processing_device}, savedir=embedding_cache_dir)
        embedding_model.eval()
        emb_dim = embedding_model.encode_batch(torch.rand(1, 16000).to(processing_device)).shape[-1]
        print(f"SpeechBrain embedding model '{args.embedding_model}' loaded (Dim: {emb_dim}).")
    except Exception as e: print(f"Warning: Error loading SpeechBrain model: {e}"); embedding_model = None

    try: # Punctuation
        # Map 'cuda'/'cpu' to device index expected by transformers pipeline
        device_index = 0 if processing_device == "cuda" else -1
        punctuation_pipeline_model = pipeline(
            "token-classification",
            model=args.punctuation_model,
            device=device_index # Use 0 for first GPU, -1 for CPU
        )
        print(f"Punctuation model '{args.punctuation_model}' loaded.")
    except Exception as e:
        print(f"Warning: Error loading punctuation model: {e}")
        punctuation_pipeline_model = None

    end_load_time = time.time()
    print(f"Model loading took {end_load_time - start_load_time:.2f} seconds.")
    # --- ASR/Diarization/Combination Steps ---
    # ... (ASR, Diarization, Combination logic as before) ...
    # ... (Ensure `combined_transcript` variable holds the result) ...
    print(f"\n--- Processing Audio File: {audio_path} ---")
    if not os.path.exists(audio_path): print(f"Error: Audio file not found at {audio_path}"); return
    print("Starting transcription..."); start_asr_time = time.time()
    try:
        segments, info = whisper_model.transcribe(audio_path, beam_size=5, word_timestamps=True)
        print(f"Detected language '{info.language}' with probability {info.language_probability}")
        transcript_results = []
        for segment in segments:
            segment_dict = {"start": segment.start, "end": segment.end, "text": segment.text, "words": [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in segment.words] if segment.words else []}
            transcript_results.append(segment_dict)
        end_asr_time = time.time(); print(f"Transcription complete. Took {end_asr_time - start_asr_time:.2f} seconds.")
    except Exception as e: print(f"Error during transcription: {e}"); return

    diarization_results_structured = None
    if diarization_pipeline:
        print("Starting diarization..."); start_dia_time = time.time()
        try:
            diarization = diarization_pipeline(audio_path)
            diarization_results_structured = [{"start": turn.start, "end": turn.end, "speaker": speaker} for turn, _, speaker in diarization.itertracks(yield_label=True)]
            diarization_results_structured.sort(key=lambda x: x['start'])
            end_dia_time = time.time(); num_speakers = len(set(d['speaker'] for d in diarization_results_structured)); print(f"Diarization complete. Found {num_speakers} speakers. Took {end_dia_time - start_dia_time:.2f} seconds.")
        except Exception as e: print(f"Warning: Error during diarization: {e}"); diarization_results_structured = None
    else: print("Skipping diarization.")

    print("Combining transcription and diarization results...")
    combined_transcript = []
    word_count_total = 0
    if diarization_results_structured:
        speaker_map_timeline = [(turn['start'], turn['end'], turn['speaker']) for turn in diarization_results_structured]
        assigned_count = 0
        for segment in transcript_results:
            segment_copy = segment.copy(); segment_copy['words'] = []
            if 'words' in segment and segment['words']:
                 word_count_total += len(segment['words'])
                 for word_info in segment['words']:
                    word_copy = word_info.copy()
                    word_midpoint = word_info.get('start', 0) + (word_info.get('end', 0) - word_info.get('start', 0)) / 2
                    assigned_speaker = "UNKNOWN"
                    for turn_start, turn_end, speaker in speaker_map_timeline:
                        if turn_start <= word_midpoint < turn_end: assigned_speaker = speaker; assigned_count += 1; break
                    word_copy['speaker'] = assigned_speaker; segment_copy['words'].append(word_copy)
            combined_transcript.append(segment_copy)
        print(f"Speaker assignment to words complete. Assigned speakers to {assigned_count}/{word_count_total} words.")
    else:
        print("Diarization results missing. Assigning 'UNKNOWN' speaker.")
        for segment in transcript_results:
             segment_copy = segment.copy(); segment_copy['words'] = []
             if 'words' in segment and segment['words']:
                  word_count_total += len(segment['words'])
                  for word_info in segment['words']:
                       word_copy = word_info.copy(); word_copy['speaker'] = "UNKNOWN"; segment_copy['words'].append(word_copy)
             combined_transcript.append(segment_copy)
    # --- Speaker Identification Step ---
    identified_transcript = combined_transcript # Start with combined data
    new_speaker_info_for_enrollment = []
    faiss_index = None # Initialize
    speaker_map = None # Initialize

    if embedding_model:
        print("\n--- Starting Speaker Identification ---")
        start_id_time = time.time()
        waveform, sample_rate = None, None
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            print(f"Audio loaded for embeddings. Shape: {waveform.shape}, SR: {sample_rate}")
        except Exception as e: print(f"Error loading audio for embeddings: {e}")

        if waveform is not None:
            faiss_index = load_or_create_faiss_index(faiss_index_path, emb_dim)
            speaker_map = load_or_create_speaker_map(speaker_map_path)
            speaker_audio_segments, effective_sample_rate = extract_speaker_audio_segments(combined_transcript, waveform, sample_rate, args.min_segment_duration)
            speaker_embeddings = get_speaker_embeddings(speaker_audio_segments, embedding_model, effective_sample_rate, processing_device)
            speaker_assignments, new_speaker_info_for_enrollment = identify_speakers(speaker_embeddings, faiss_index, speaker_map, args.similarity_threshold)
            identified_transcript = update_transcript_speakers(combined_transcript, speaker_assignments)
            # Save intermediate identified JSON (before punctuation)
            try:
                with open(identified_json_path, 'w') as f: json.dump(identified_transcript, f, indent=2)
                print(f"Intermediate identified transcript saved to {identified_json_path}")
            except Exception as e: print(f"Error saving intermediate identified JSON: {e}")
            end_id_time = time.time()
            print(f"Speaker Identification process took {end_id_time - start_id_time:.2f} seconds.")
        else: print("Skipping speaker identification due to audio loading error.")
    else: print("\nSkipping Speaker Identification (no embedding model).")

    # --- Punctuation Restoration Step ---
    punctuated_output = None
    if punctuation_pipeline_model and identified_transcript: # Check if model loaded and we have transcript
        start_punct_time = time.time()
        punctuated_output = apply_punctuation(
            identified_transcript,
            punctuation_pipeline_model,
            args.punctuation_chunk_size
            )
        # Save punctuated text file
        try:
            with open(punctuated_txt_path, 'w') as f:
                for speaker, text in punctuated_output:
                     # Format: Speaker Name: Text \n\n (or choose another format)
                     f.write(f"{speaker}:\n{text}\n\n")
            print(f"Final punctuated transcript saved to {punctuated_txt_path}")
        except Exception as e:
            print(f"Error saving punctuated text file: {e}")
        end_punct_time = time.time()
        print(f"Punctuation restoration took {end_punct_time - start_punct_time:.2f} seconds.")

    elif not punctuation_pipeline_model:
        print("\nSkipping Punctuation Restoration (no punctuation model loaded).")
    else:
         print("\nSkipping Punctuation Restoration (no transcript data).")


    # --- Final Enrollment Step (if needed) ---
    # Run enrollment only if identification happened and faiss/map were loaded
    if faiss_index is not None and speaker_map is not None:
         enroll_new_speakers_cli(
             new_speaker_info_for_enrollment, # Use the list generated during identification
             faiss_index,
             speaker_map,
             faiss_index_path,
             speaker_map_path
         )
    else:
        print("\nSkipping enrollment step as identification did not run or DB failed to load.")

    # --- Cleanup ---
    del whisper_model, diarization_pipeline, embedding_model, punctuation_pipeline_model
    del transcript_results, combined_transcript, identified_transcript, punctuated_output
    gc.collect()
    if processing_device == "cuda": torch.cuda.empty_cache()

    end_total_time = time.time()
    print(f"\nScript finished. Total execution time: {end_total_time - start_total_time:.2f} seconds.")


# ==============================================================
# Script Execution Start
# ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio, identify speakers, and restore punctuation.")

    # --- Input/Output Arguments ---
    parser.add_argument("input_audio", type=str, help="Path to the input audio file.")
    parser.add_argument("-o", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Directory to save output files (Default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--output-json-file", type=str, default=DEFAULT_IDENTIFIED_JSON_FILENAME, help=f"Filename for the intermediate identified transcript JSON (Default: {DEFAULT_IDENTIFIED_JSON_FILENAME})")
    parser.add_argument("--output-punct-file", type=str, default=DEFAULT_PUNCTUATED_TXT_FILENAME, help=f"Filename for the final punctuated transcript TXT (Default: {DEFAULT_PUNCTUATED_TXT_FILENAME})")
    parser.add_argument("--hf-token", type=str, default=DEFAULT_HF_TOKEN_FILE, help=f"Path to Hugging Face token file (Default: {DEFAULT_HF_TOKEN_FILE})")
    parser.add_argument("--faiss-file", type=str, default=DEFAULT_FAISS_INDEX_FILENAME, help=f"Filename for the FAISS index (Default: {DEFAULT_FAISS_INDEX_FILENAME})")
    parser.add_argument("--map-file", type=str, default=DEFAULT_SPEAKER_MAP_FILENAME, help=f"Filename for the speaker name map JSON (Default: {DEFAULT_SPEAKER_MAP_FILENAME})")

    # --- Model/Processing Arguments ---
    parser.add_argument("--whisper-model", type=str, default=DEFAULT_WHISPER_MODEL, help=f"Faster Whisper model name (Default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--whisper-device", type=str, default=DEFAULT_WHISPER_DEVICE, choices=["cpu", "cuda"], help=f"Device for Faster Whisper (Default: {DEFAULT_WHISPER_DEVICE})")
    parser.add_argument("--whisper-compute", type=str, default=DEFAULT_WHISPER_COMPUTE, choices=["int8", "float16", "float32"], help=f"Compute type for Faster Whisper (Default: {DEFAULT_WHISPER_COMPUTE})")
    parser.add_argument("--processing-device", type=str, default=DEFAULT_PROCESSING_DEVICE, choices=["cpu", "cuda"], help=f"Device for Pyannote, SpeechBrain, Punctuation (Default: {DEFAULT_PROCESSING_DEVICE}, uses CPU if CUDA unavailable)")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL, help=f"SpeechBrain embedding model name (Default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--punctuation-model", type=str, default=DEFAULT_PUNCTUATION_MODEL, help=f"Hugging Face punctuation model name (Default: {DEFAULT_PUNCTUATION_MODEL})")
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help=f"Cosine similarity threshold for speaker ID (Default: {DEFAULT_SIMILARITY_THRESHOLD})")
    parser.add_argument("--min-segment-duration", type=float, default=DEFAULT_MIN_SEGMENT_DURATION, help=f"Min audio duration (s) per speaker for embedding (Default: {DEFAULT_MIN_SEGMENT_DURATION})")
    parser.add_argument("--punctuation-chunk-size", type=int, default=DEFAULT_PUNCTUATION_CHUNK_SIZE, help=f"Max words per chunk for punctuation model (Default: {DEFAULT_PUNCTUATION_CHUNK_SIZE})")


    cli_args = parser.parse_args()
    main(cli_args)
