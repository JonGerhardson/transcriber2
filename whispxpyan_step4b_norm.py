# whispxpyan_step4b_norm.py
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
from transformers import pipeline
import re # Added for custom regex
from cucco import Cucco # Added for normalization
import inflect # Added for number to words

# ==============================================================
# Configuration (Defaults)
# ==============================================================
DEFAULT_OUTPUT_DIR = "/app/transcripts_output"
DEFAULT_HF_TOKEN_FILE = "/app/hf-token.txt"
DEFAULT_FAISS_INDEX_FILENAME = "speaker_embeddings.index"
DEFAULT_SPEAKER_MAP_FILENAME = "speaker_names.json"
DEFAULT_IDENTIFIED_JSON_FILENAME = "intermediate_identified_transcript.json"
DEFAULT_FINAL_OUTPUT_FILENAME = "final_normalized_transcript.txt" # Final output name changed

DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_WHISPER_DEVICE = "cpu"
DEFAULT_WHISPER_COMPUTE = "int8"

DEFAULT_PROCESSING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_PUNCTUATION_MODEL = "felflare/bert-restore-punctuation"
DEFAULT_SIMILARITY_THRESHOLD = 0.80
DEFAULT_MIN_SEGMENT_DURATION = 2.0
DEFAULT_PUNCTUATION_CHUNK_SIZE = 256

# Default Normalization Options
DEFAULT_NORMALIZE_NUMBERS = False
DEFAULT_REMOVE_FILLERS = False
DEFAULT_FILLER_WORDS = ['um', 'uh', 'hmm', 'mhm', 'uh huh', 'like', 'you know'] # Basic list

# Initialize Cucco and Inflect globally (or pass as needed)
cucco_instance = Cucco()
inflect_engine = inflect.engine()

# ==============================================================
# Helper Functions (Including Punctuation - Needs Pasting)
# ==============================================================
# Assume functions load/save_faiss_index, load/save_speaker_map,
# extract_speaker_audio_segments, get_speaker_embeddings, identify_speakers,
# update_transcript_speakers, enroll_new_speakers_cli,
# format_punctuated_output, apply_punctuation are here and correct
# --- [PASTE HELPER FUNCTIONS FROM PREVIOUS SCRIPT HERE] ---
# (Ensure apply_punctuation and its helper format_punctuated_output are included)
# ...
# --- [HELPER FUNCTIONS WOULD GO HERE - START] ---
# --- Example Placeholder for format_punctuated_output ---
def format_punctuated_output(results):
    """Reconstructs text from the punctuation pipeline output."""
    # (Implementation from step4a should be pasted here)
    text = ''
    # Example logic - replace with actual implementation
    for item in results:
         word = item['word']
         # Simplified logic - needs proper handling of labels and spacing
         text += word + ' '
    return text.strip() + "." # Placeholder punctuation

# --- Example Placeholder for apply_punctuation ---
def apply_punctuation(transcript_data, punctuation_pipeline, chunk_size=256):
    """Applies punctuation to the transcript, processing speaker turns."""
    # (Implementation from step4a should be pasted here)
    print("\n--- Applying Punctuation (Placeholder Implementation) ---")
    punctuated_turns = []
    current_speaker = None
    current_text = ""
    # Example logic - replace with actual implementation
    for segment in transcript_data:
        if 'words' in segment and segment['words']:
            for word_info in segment['words']:
                 speaker = word_info.get('speaker', 'UNKNOWN')
                 word = word_info.get('word', '').strip()
                 if not word: continue

                 if speaker != current_speaker and current_text:
                      # Placeholder processing - apply punctuation model here
                      # processed_text = punctuation_pipeline(...)
                      processed_text = format_punctuated_output([{'word': w} for w in current_text.split()]) # Very basic placeholder
                      punctuated_turns.append((current_speaker, processed_text))
                      current_text = ""

                 current_speaker = speaker
                 current_text += word + " "

    if current_text:
         # processed_text = punctuation_pipeline(...)
         processed_text = format_punctuated_output([{'word': w} for w in current_text.split()]) # Very basic placeholder
         punctuated_turns.append((current_speaker, processed_text))

    print("Punctuation application complete.")
    return punctuated_turns

# --- Other helper functions (FAISS, Embeddings, etc.) would also go here ---
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
            speaker_map = {int(k): v for k, v in speaker_map.items()}
            print(f"Speaker map loaded with {len(speaker_map)} entries.")
        except Exception as e:
            print(f"Error loading speaker map: {e}. Creating a new one.")
            speaker_map = {}
    else:
        print(f"Speaker map not found at {map_path}. Creating a new one.")
        speaker_map = {}
    return speaker_map

def save_speaker_map(speaker_map, map_path):
    """Saves the speaker name map to JSON."""
    print(f"Saving speaker map to {map_path} with {len(speaker_map)} entries...")
    try:
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        save_map = {str(k): v for k, v in speaker_map.items()}
        with open(map_path, 'w') as f:
            json.dump(save_map, f, indent=2)
        print("Speaker map saved successfully.")
    except Exception as e:
        print(f"Error saving speaker map: {e}")

def extract_speaker_audio_segments(transcript_data, audio_waveform, sample_rate, min_segment_duration, target_sr=16000):
    """Extracts audio segments for each generic speaker."""
    speaker_segments = defaultdict(list); total_duration = defaultdict(float)
    if sample_rate != target_sr:
        print(f"Resampling audio from {sample_rate} Hz to {target_sr} Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        audio_waveform = resampler(audio_waveform); sample_rate = target_sr
        print("Resampling complete.")
    print("Extracting audio segments...")
    for segment in transcript_data: # Process segments...
        if 'words' in segment and segment['words']:
             for word_info in segment['words']: # Process words...
                 # (Logic as in step4a to extract chunks based on speaker)
                 speaker = word_info.get('speaker'); start_time = word_info.get('start'); end_time = word_info.get('end')
                 if speaker and speaker.startswith("SPEAKER_") and start_time is not None and end_time is not None:
                     start_sample = int(start_time * sample_rate); end_sample = int(end_time * sample_rate)
                     if end_sample <= audio_waveform.shape[1]:
                         audio_chunk = torch.mean(audio_waveform[:, start_sample:end_sample], dim=0, keepdim=True) if audio_waveform.shape[0] > 1 else audio_waveform[:, start_sample:end_sample]
                         speaker_segments[speaker].append(audio_chunk)
                         total_duration[speaker] += (end_time - start_time)
    filtered_segments = {}
    for speaker, duration in total_duration.items(): # Filter by duration...
        if duration >= min_segment_duration:
            try: filtered_segments[speaker] = torch.cat(speaker_segments[speaker], dim=1); print(f"  - Speaker {speaker}: {duration:.2f}s (sufficient).")
            except Exception as e: print(f"  - Speaker {speaker}: Error concatenating: {e}. Skipping.")
        else: print(f"  - Speaker {speaker}: {duration:.2f}s (insufficient, skipping).")
    return filtered_segments, sample_rate

def get_speaker_embeddings(speaker_audio_segments, embedding_model, sample_rate, device):
    """Generates speaker embeddings using the SpeechBrain model."""
    embeddings = {}; print("Generating speaker embeddings...");
    if not speaker_audio_segments: print("No segments found."); return embeddings
    with torch.no_grad():
        for speaker, audio_tensor in speaker_audio_segments.items(): # Generate embeddings...
            print(f"  - Processing {speaker}...")
            try:
                 audio_tensor = audio_tensor.to(device)
                 if audio_tensor.dim() == 1: audio_tensor = audio_tensor.unsqueeze(0)
                 embedding = embedding_model.encode_batch(audio_tensor).squeeze()
                 norm = torch.linalg.norm(embedding); embedding = embedding / norm if norm > 1e-6 else embedding
                 embeddings[speaker] = embedding.cpu().numpy()
                 print(f"    Generated embedding for {speaker}.")
            except Exception as e: print(f"    Error generating embedding for {speaker}: {e}")
    print("Embedding generation complete."); return embeddings

def identify_speakers(embeddings, faiss_index, speaker_map, similarity_threshold):
    """Identifies speakers by comparing embeddings against the FAISS index."""
    speaker_assignments = {}; new_speaker_embeddings_info = []; unknown_speaker_count = 0
    print(f"Starting identification. Index size: {faiss_index.ntotal}")
    if faiss_index.ntotal == 0: # Handle empty index...
        print("FAISS index empty. Marking all as unknown.")
        for speaker, embedding in embeddings.items(): unknown_speaker_count += 1; temp_id = f"UNKNOWN_{unknown_speaker_count}"; speaker_assignments[speaker] = temp_id; new_speaker_embeddings_info.append({'temp_id': temp_id, 'embedding': embedding, 'original_label': speaker})
    else: # Search index...
        print("Identifying speakers using FAISS index...")
        for speaker, embedding in embeddings.items():
            try:
                query_embedding = np.expand_dims(embedding.astype(np.float32), axis=0)
                distances, indices = faiss_index.search(query_embedding, k=1)
                faiss_id = indices[0][0]; similarity = distances[0][0]
                print(f"  - Speaker {speaker}: Closest FAISS ID {faiss_id}, Similarity: {similarity:.4f}")
                if similarity >= similarity_threshold and faiss_id >= 0 and faiss_id in speaker_map:
                     identified_name = speaker_map[faiss_id]; speaker_assignments[speaker] = identified_name; print(f"    Identified as: '{identified_name}'")
                else: # Mark as unknown...
                    unknown_speaker_count += 1; temp_id = f"UNKNOWN_{unknown_speaker_count}"; speaker_assignments[speaker] = temp_id; new_speaker_embeddings_info.append({'temp_id': temp_id, 'embedding': embedding, 'original_label': speaker}); print(f"    Marking as unknown ({temp_id}). Reason: {'Similarity too low' if similarity < similarity_threshold else 'ID not in map' if faiss_id not in speaker_map else 'No match found'}).")
            except Exception as e: print(f"    Error identifying {speaker}: {e}"); unknown_speaker_count += 1; speaker_assignments[speaker] = f"UNKNOWN_{unknown_speaker_count}" # Fallback
    print("Speaker identification complete."); return speaker_assignments, new_speaker_embeddings_info

def update_transcript_speakers(transcript_data, speaker_assignments):
    """Updates the transcript data with identified speaker names or temp IDs."""
    print("Updating transcript with identified speaker names...")
    updated_data = []
    for segment in transcript_data: # Update segments...
        updated_segment = segment.copy(); updated_words = []
        if 'words' in segment and segment['words']:
            for word_info in segment['words']: # Update words...
                 updated_word_info = word_info.copy()
                 generic_speaker = word_info.get('speaker')
                 if generic_speaker and generic_speaker in speaker_assignments: updated_word_info['speaker'] = speaker_assignments[generic_speaker]
                 elif generic_speaker and not generic_speaker.startswith("SPEAKER_"): pass # Already identified or UNKNOWN
                 else: updated_word_info['speaker'] = f"UNPROCESSED_{generic_speaker}" if generic_speaker else "UNKNOWN_WORD_SPEAKER"
                 updated_words.append(updated_word_info)
            updated_segment['words'] = updated_words
        updated_data.append(updated_segment)
    print("Transcript update complete."); return updated_data

def enroll_new_speakers_cli(new_speaker_info, faiss_index, speaker_map, faiss_index_path, speaker_map_path):
    """Handles interactive enrollment of new speakers via the CLI."""
    if not new_speaker_info: print("\nNo new speakers detected for enrollment."); return False
    print("\n--- Speaker Enrollment ---"); changes_made = False
    new_speaker_info.sort(key=lambda x: int(x['temp_id'].split('_')[-1]))
    for speaker_data in new_speaker_info: # Enroll loop...
        temp_id = speaker_data['temp_id']; embedding = speaker_data['embedding']; original_label = speaker_data.get('original_label', 'N/A')
        prompt = f"Enroll speaker {temp_id} (orig: {original_label})? Enter name (or blank to skip): "
        try: entered_name = input(prompt).strip()
        except EOFError: print("\nEOF detected, skipping enrollment."); break
        if entered_name: # Add to DB...
            try:
                faiss_index.add(np.expand_dims(embedding.astype(np.float32), axis=0)); new_faiss_id = faiss_index.ntotal - 1
                speaker_map[new_faiss_id] = entered_name; print(f"  -> Enrolled '{entered_name}' with FAISS ID {new_faiss_id}."); changes_made = True
            except Exception as e: print(f"  -> Error enrolling {temp_id} as '{entered_name}': {e}")
        else: print(f"  -> Skipping enrollment for {temp_id}.")
    if changes_made: print("\nSaving updated speaker database..."); save_faiss_index(faiss_index, faiss_index_path); save_speaker_map(speaker_map, speaker_map_path); return True
    else: print("\nNo changes made to speaker database."); return False
# --- [HELPER FUNCTIONS WOULD GO HERE - END] ---

# ==============================================================
# NEW: Text Normalization Function
# ==============================================================
def normalize_text_custom(text, normalize_numbers=False, remove_fillers=False, filler_words=None):
    """Applies selected text normalization rules."""
    global cucco_instance, inflect_engine # Use global instances

    if filler_words is None:
        filler_words = []

    normalized_text = text

    # 1. Cucco basic cleanup (whitespace)
    try:
        # Apply only specific, safe normalizations from Cucco
        normalized_text = cucco_instance.normalize(normalized_text, ['remove_extra_white_spaces'])
    except Exception as e:
        print(f"Warning: Cucco normalization failed: {e}")
        # Continue with the original text if cucco fails

    # 2. Number to Words Conversion (using inflect)
    if normalize_numbers:
        # Use regex to find numbers (integers and potentially simple decimals)
        # This regex is basic, might need refinement for complex numbers, currency, etc.
        def replace_num(match):
            num_str = match.group()
            try:
                # Handle potential commas in numbers before conversion
                num_str_no_comma = num_str.replace(',', '')
                # Basic check if it looks like a number (int or float)
                if re.match(r'^-?\d+(?:,\d+)*(\.\d+)?$', num_str):
                     # Use inflect for conversion
                     num_word = inflect_engine.number_to_words(num_str_no_comma)
                     # Remove commas potentially added by inflect for large numbers if desired
                     # num_word = num_word.replace(',', '')
                     return num_word
                else:
                    return num_str # Not a standard number format, leave as is
            except Exception as e:
                 print(f"Warning: Failed to convert number '{num_str}' to words: {e}")
                 return num_str # Return original if conversion fails

        # Regex to find sequences of digits, possibly with commas or a decimal point
        # Be careful not to convert things like years or codes unintentionally
        # This pattern tries to capture standalone numbers
        normalized_text = re.sub(r'\b-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+\b', replace_num, normalized_text)


    # 3. Filler Word Removal (case-insensitive)
    if remove_fillers and filler_words:
        # Create a regex pattern: \b(um|uh|like|...)\b optionally followed by punctuation
        # Sort fillers by length descending to match longer phrases first (e.g., 'you know' before 'know')
        filler_words_sorted = sorted(filler_words, key=len, reverse=True) # Use a sorted copy
        # Ensure filler words are regex escaped
        escaped_fillers = [re.escape(word) for word in filler_words_sorted]
        # Pattern: word boundary, non-capturing group of fillers, word boundary, optional punctuation and space
        # We aim to remove the filler and potentially ONE trailing space IF it exists before punctuation or end of string
        pattern = r'\b(?:' + '|'.join(escaped_fillers) + r')\b[.,?!;:]?\s?'
        # Replace with empty string, use re.IGNORECASE
        normalized_text = re.sub(pattern, '', normalized_text, flags=re.IGNORECASE)
        # Clean up potential double spaces left after removal
        normalized_text = re.sub(r'\s{2,}', ' ', normalized_text).strip()
        # Handle cases where removal might leave punctuation hanging, e.g., " , text" -> ", text"
        # Or "text ." -> "text."
        normalized_text = re.sub(r'^\s*([.,?!;:])', r'\1', normalized_text) # At beginning
        normalized_text = re.sub(r'\s+([.,?!;:])', r'\1', normalized_text) # Elsewhere


    # 4. Final whitespace cleanup (redundant with cucco but safe)
    normalized_text = ' '.join(normalized_text.split())

    return normalized_text

# ==============================================================
# Main Processing Function (Modified for Normalization)
# ==============================================================
def main(args):
    """Main processing pipeline."""
    start_total_time = time.time()
    # --- Construct paths ---
    output_dir = args.output_dir
    audio_path = args.input_audio
    hf_token_file = args.hf_token
    faiss_index_path = os.path.join(output_dir, args.faiss_file)
    speaker_map_path = os.path.join(output_dir, args.map_file)
    identified_json_path = os.path.join(output_dir, args.output_json_file)
    final_txt_path = os.path.join(output_dir, args.output_final_file) # Use final filename arg


    os.makedirs(output_dir, exist_ok=True)
    # --- Device Settings ---
    processing_device = args.processing_device if torch.cuda.is_available() else "cpu"
    if args.processing_device == "cuda" and not torch.cuda.is_available(): print("Warning: CUDA requested but not available, falling back to CPU for processing.")
    whisper_device = args.whisper_device
    whisper_compute_type = args.whisper_compute

    # --- Print Settings ---
    print(f"Settings:")
    print(f"  Input Audio: {audio_path}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Output JSON File: {identified_json_path}")
    print(f"  Output Final TXT File: {final_txt_path}")
    print(f"  FAISS Index: {faiss_index_path}")
    print(f"  Speaker Map: {speaker_map_path}")
    print(f"  HF Token File: {hf_token_file}")
    print(f"  Whisper Device: {whisper_device}, Compute: {whisper_compute_type}, Model: {args.whisper_model}")
    print(f"  Processing Device (Pyannote/SpeechBrain/Punct): {processing_device}")
    print(f"  Embedding Model: {args.embedding_model}")
    print(f"  Punctuation Model: {args.punctuation_model}")
    print(f"  Similarity Threshold: {args.similarity_threshold}")
    print(f"  Min Segment Duration: {args.min_segment_duration}")
    print(f"  Punctuation Chunk Size: {args.punctuation_chunk_size}")
    print(f"  Normalize Numbers: {args.normalize_numbers}")
    print(f"  Remove Fillers: {args.remove_fillers}")
    print(f"  Filler Words List: {DEFAULT_FILLER_WORDS if args.remove_fillers else 'N/A'}")
    print("-" * 20)

    # --- Load HF Token ---
    hf_token = None # Define hf_token before try block
    if hf_token_file and os.path.exists(hf_token_file):
        try:
            with open(hf_token_file, 'r') as f: hf_token = f.read().strip()
            if not hf_token: print(f"Warning: HF token file '{hf_token_file}' is empty."); hf_token = None
            else: print("Hugging Face token loaded successfully.")
        except Exception as e: print(f"Error loading HF token from {hf_token_file}: {e}"); hf_token = None
    elif hf_token_file: print(f"Warning: HF token file specified but not found: {hf_token_file}")
    else: print("Hugging Face token file not specified. Accept gated access terms for pyannote/diarization3.1 and pyannote/segmentation3.0 then generate an access token at https://huggingface.co/settings/tokens with read access to gated repos, and save the token at hf-token.txt in this directory.")


    # --- Initialize Models ---
    print("\nLoading models...")
    start_load_time = time.time()
    whisper_model, diarization_pipeline, embedding_model, punctuation_pipeline_model = None, None, None, None
    emb_dim = 192 # Default, will be updated by embedding model if loaded
    try: # Whisper
        print(f"Loading Whisper model: {args.whisper_model}...")
        whisper_model = WhisperModel(args.whisper_model, device=whisper_device, compute_type=whisper_compute_type, local_files_only=False)
        print(f"Faster Whisper model '{args.whisper_model}' loaded.")
    except Exception as e: print(f"FATAL Error loading Faster Whisper model: {e}"); return

    if hf_token: # Pyannote
        try:
            print("Loading Pyannote pipeline...")
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
            diarization_pipeline.to(torch.device(processing_device))
            print(f"Pyannote diarization pipeline loaded to {processing_device}.")
        except Exception as e: print(f"Warning: Error loading Pyannote pipeline: {e}. Diarization will be skipped."); diarization_pipeline = None
    else: print("Skipping Pyannote loading (no valid token found/specified).")

    try: # SpeechBrain
        print(f"Loading SpeechBrain model: {args.embedding_model}...")
        # Define a cache directory relative to the output directory
        embedding_cache_dir = os.path.join(output_dir, 'embedding_model_cache')
        os.makedirs(embedding_cache_dir, exist_ok=True) # Ensure cache dir exists
        embedding_model = EncoderClassifier.from_hparams(source=args.embedding_model, run_opts={"device": processing_device}, savedir=embedding_cache_dir)
        embedding_model.eval()
        # Get embedding dimension dynamically
        try:
             dummy_input = torch.rand(1, 16000).to(processing_device) # Create on correct device
             emb_dim = embedding_model.encode_batch(dummy_input).shape[-1]
             print(f"SpeechBrain embedding model '{args.embedding_model}' loaded to {processing_device} (Dim: {emb_dim}).")
        except Exception as e_emb:
             print(f"Warning: Could not determine embedding dimension dynamically: {e_emb}. Using default {emb_dim}.")
    except Exception as e: print(f"Warning: Error loading SpeechBrain model: {e}. Speaker identification will be skipped."); embedding_model = None

    try: # Punctuation
        print(f"Loading Punctuation model: {args.punctuation_model}...")
        # Map 'cuda'/'cpu' to device index expected by transformers pipeline
        device_index = 0 if processing_device == "cuda" else -1
        punctuation_pipeline_model = pipeline(
            "token-classification",
            model=args.punctuation_model,
            aggregation_strategy="simple", # Try simple strategy
            device=device_index # Use 0 for first GPU, -1 for CPU
        )
        print(f"Punctuation model '{args.punctuation_model}' loaded to device index {device_index}.")
    except Exception as e:
        import traceback
        print(f"ERROR: Failed to load punctuation model '{args.punctuation_model}'. Punctuation/Normalization may fail. Details below:")
        traceback.print_exc()
        punctuation_pipeline_model = None # Set to None if loading fails


    end_load_time = time.time(); print(f"Model loading took {end_load_time - start_load_time:.2f} seconds.")


    # --- ASR/Diarization/Combination Steps ---
    print(f"\n--- Processing Audio File: {audio_path} ---")
    if not os.path.exists(audio_path): print(f"Error: Audio file not found at {audio_path}"); return

    print("Starting transcription..."); start_asr_time = time.time()
    transcript_results = []
    try:
        # Transcribe with word timestamps
        segments, info = whisper_model.transcribe(audio_path, beam_size=5, word_timestamps=True)
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

        # Structure the results
        for segment in segments:
            transcript_results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in segment.words] if segment.words else []
            })
        end_asr_time = time.time(); print(f"Transcription complete. Took {end_asr_time - start_asr_time:.2f} seconds.")
    except Exception as e: print(f"Error during transcription: {e}"); return # Exit if ASR fails

    # Diarization
    diarization_results_structured = None
    if diarization_pipeline:
        print("Starting diarization..."); start_dia_time = time.time()
        try:
            diarization = diarization_pipeline(audio_path)
            # Structure diarization results
            diarization_results_structured = [{"start": turn.start, "end": turn.end, "speaker": speaker}
                                              for turn, _, speaker in diarization.itertracks(yield_label=True)]
            diarization_results_structured.sort(key=lambda x: x['start']) # Sort by start time
            end_dia_time = time.time()
            num_speakers = len(set(d['speaker'] for d in diarization_results_structured))
            print(f"Diarization complete. Found {num_speakers} speakers. Took {end_dia_time - start_dia_time:.2f} seconds.")
        except Exception as e: print(f"Warning: Error during diarization: {e}"); diarization_results_structured = None
    else: print("Skipping diarization.")

    # Combination
    print("Combining transcription and diarization results...")
    combined_transcript = []
    word_count_total = 0
    assigned_count = 0

    if diarization_results_structured:
        speaker_map_timeline = [(turn['start'], turn['end'], turn['speaker']) for turn in diarization_results_structured]

        for segment in transcript_results:
            segment_copy = segment.copy(); segment_copy['words'] = []
            if 'words' in segment and segment['words']:
                 word_count_total += len(segment['words'])
                 for word_info in segment['words']:
                    word_copy = word_info.copy()
                    # Use word midpoint for speaker assignment
                    word_midpoint = word_info.get('start', 0) + (word_info.get('end', 0) - word_info.get('start', 0)) / 2
                    assigned_speaker = "UNKNOWN" # Default if no match
                    for turn_start, turn_end, speaker in speaker_map_timeline:
                        if turn_start <= word_midpoint < turn_end:
                            assigned_speaker = speaker
                            assigned_count += 1
                            break # Found speaker for this word
                    word_copy['speaker'] = assigned_speaker
                    segment_copy['words'].append(word_copy)
            combined_transcript.append(segment_copy)
        print(f"Speaker assignment to words complete. Assigned speakers to {assigned_count}/{word_count_total} words based on diarization.")
    else:
        # If no diarization, assign 'UNKNOWN' to all words
        print("Diarization results missing or failed. Assigning 'UNKNOWN' speaker to all words.")
        for segment in transcript_results:
             segment_copy = segment.copy(); segment_copy['words'] = []
             if 'words' in segment and segment['words']:
                  word_count_total += len(segment['words'])
                  for word_info in segment['words']:
                       word_copy = word_info.copy()
                       word_copy['speaker'] = "UNKNOWN" # Assign UNKNOWN
                       segment_copy['words'].append(word_copy)
             combined_transcript.append(segment_copy)


    # --- Speaker Identification Step ---
    identified_transcript = combined_transcript # Start with combined data (might be just UNKNOWN speakers)
    new_speaker_info_for_enrollment = []
    faiss_index = None # Initialize
    speaker_map = None # Initialize

    # Proceed only if embedding model loaded and combined_transcript exists
    if embedding_model and combined_transcript:
        print("\n--- Starting Speaker Identification ---")
        start_id_time = time.time()
        waveform, sample_rate = None, None
        try:
            print(f"Loading audio for embeddings: {audio_path}")
            waveform, sample_rate = torchaudio.load(audio_path)
            print(f"Audio loaded for embeddings. Shape: {waveform.shape}, SR: {sample_rate}")
        except Exception as e: print(f"Error loading audio for embeddings: {e}")

        if waveform is not None:
            # Load or create FAISS index and speaker map
            faiss_index = load_or_create_faiss_index(faiss_index_path, emb_dim)
            speaker_map = load_or_create_speaker_map(speaker_map_path)

            # Extract audio segments for speakers identified by diarization (e.g., SPEAKER_00, SPEAKER_01)
            speaker_audio_segments, effective_sample_rate = extract_speaker_audio_segments(
                combined_transcript, waveform, sample_rate, args.min_segment_duration
            )

            # Generate embeddings for these speakers
            speaker_embeddings = get_speaker_embeddings(
                speaker_audio_segments, embedding_model, effective_sample_rate, processing_device
            )

            # Identify speakers against the known database
            speaker_assignments, new_speaker_info_for_enrollment = identify_speakers(
                speaker_embeddings, faiss_index, speaker_map, args.similarity_threshold
            )

            # Update the transcript with identified names (or UNKNOWN_X if new)
            identified_transcript = update_transcript_speakers(
                combined_transcript, speaker_assignments
            )

            # Save intermediate identified JSON (before punctuation/normalization)
            try:
                print(f"Saving intermediate identified transcript to {identified_json_path}")
                with open(identified_json_path, 'w') as f:
                     json.dump(identified_transcript, f, indent=2)
                print(f"Intermediate identified transcript saved successfully.")
            except Exception as e: print(f"Error saving intermediate identified JSON: {e}")

            end_id_time = time.time()
            print(f"Speaker Identification process took {end_id_time - start_id_time:.2f} seconds.")
        else:
            print("Skipping speaker identification due to audio loading error.")
            identified_transcript = combined_transcript # Pass through if identification skipped
    elif not embedding_model:
        print("\nSkipping Speaker Identification (no embedding model loaded).")
        identified_transcript = combined_transcript # Pass through if no model
    else:
        print("\nSkipping Speaker Identification (no initial transcript data).")
        # identified_transcript is already likely [] or None in this case


    # --- Punctuation Restoration Step ---
    punctuated_output = None
    # Ensure punctuation model is loaded and we have some transcript data
    print(f"DEBUG: Checking identified_transcript before punctuation. Is None? {identified_transcript is None}. Is empty list? {isinstance(identified_transcript, list) and not identified_transcript}")
    if punctuation_pipeline_model and identified_transcript:
        print("\n--- Applying Punctuation ---")
        start_punct_time = time.time()
        # Use the apply_punctuation function (assuming it's pasted correctly above)
        punctuated_output = apply_punctuation(
            identified_transcript,
            punctuation_pipeline_model,
            args.punctuation_chunk_size
            )
        # Note: Punctuation output format is [(speaker, text), (speaker, text), ...]
        end_punct_time = time.time()
        print(f"Punctuation restoration took {end_punct_time - start_punct_time:.2f} seconds.")
    elif not punctuation_pipeline_model:
        print("\nSkipping Punctuation Restoration (punctuation model failed to load or was not specified).")
    else: # No transcript data
         print("\nSkipping Punctuation Restoration (no transcript data from previous steps).")


    # --- Text Normalization Step ---
    final_output_turns = []
    if punctuated_output: # Check if punctuation ran successfully and produced output
         print("\n--- Applying Text Normalization ---")
         start_norm_time = time.time()
         # Determine filler list based on args
         filler_list = DEFAULT_FILLER_WORDS if args.remove_fillers else []
         # Process each speaker's punctuated turn
         for speaker, punctuated_text in punctuated_output:
             normalized_text = normalize_text_custom(
                 punctuated_text,
                 normalize_numbers=args.normalize_numbers,
                 remove_fillers=args.remove_fillers,
                 filler_words=filler_list # Pass the actual list
             )
             final_output_turns.append((speaker, normalized_text))
         end_norm_time = time.time()
         print(f"Text normalization complete. Took {end_norm_time - start_norm_time:.2f} seconds.")

         # --- Save Final Normalized Output ---
         try:
            print(f"Saving final normalized transcript to {final_txt_path}")
            with open(final_txt_path, 'w', encoding='utf-8') as f: # Ensure utf-8 encoding
                for speaker, text in final_output_turns:
                     # Format: Speaker Name:\nText\n\n
                     f.write(f"{speaker}:\n{text}\n\n")
            print(f"Final normalized transcript saved successfully.")
         except Exception as e:
            print(f"Error saving final normalized text file: {e}")

    elif identified_transcript:
         # If punctuation failed/skipped, but we have identified transcript,
         # maybe save that directly? Or skip normalization.
         print("\nSkipping Text Normalization because punctuation step did not produce output.")
         # Optionally, save the unpunctuated, identified transcript here if desired.
         # For now, we only save if punctuation AND normalization runs.

    else:
        print("\nSkipping Text Normalization (no data from previous steps).")


    # --- Final Enrollment Step (if needed) ---
    # Run enrollment only if identification happened and faiss/map were loaded
    if faiss_index is not None and speaker_map is not None and new_speaker_info_for_enrollment:
         enroll_new_speakers_cli(
             new_speaker_info_for_enrollment, # Use the list generated during identification
             faiss_index,
             speaker_map,
             faiss_index_path,
             speaker_map_path
         )
    else:
        print("\nSkipping enrollment step (identification may have been skipped, DB failed to load, or no new speakers found).")


    # --- Cleanup ---
    print("\nCleaning up models...")
    del whisper_model, diarization_pipeline, embedding_model, punctuation_pipeline_model
    del transcript_results, combined_transcript, identified_transcript, punctuated_output, final_output_turns
    gc.collect()
    if processing_device == "cuda":
        try:
             torch.cuda.empty_cache()
             print("Cleared CUDA cache.")
        except Exception as e:
             print(f"Warning: Could not clear CUDA cache: {e}")


    end_total_time = time.time()
    print(f"\nScript finished. Total execution time: {end_total_time - start_total_time:.2f} seconds.")


# ==============================================================
# Script Execution Start (Modified Argparse)
# ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio, identify speakers, restore punctuation, and normalize text.")

    # --- Input/Output Arguments ---
    parser.add_argument("input_audio", type=str, help="Path to the input audio file.")
    parser.add_argument("-o", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Directory to save output files (Default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--output-json-file", type=str, default=DEFAULT_IDENTIFIED_JSON_FILENAME, help=f"Filename for the intermediate identified transcript JSON (Default: {DEFAULT_IDENTIFIED_JSON_FILENAME})")
    parser.add_argument("--output-final-file", type=str, default=DEFAULT_FINAL_OUTPUT_FILENAME, help=f"Filename for the final normalized/punctuated transcript TXT (Default: {DEFAULT_FINAL_OUTPUT_FILENAME})") # Changed name
    parser.add_argument("--hf-token", type=str, default=DEFAULT_HF_TOKEN_FILE, help=f"Path to Hugging Face token file (Optional, reads if exists. Default: {DEFAULT_HF_TOKEN_FILE})")
    parser.add_argument("--faiss-file", type=str, default=DEFAULT_FAISS_INDEX_FILENAME, help=f"Filename for the FAISS index within output dir (Default: {DEFAULT_FAISS_INDEX_FILENAME})")
    parser.add_argument("--map-file", type=str, default=DEFAULT_SPEAKER_MAP_FILENAME, help=f"Filename for the speaker name map JSON within output dir (Default: {DEFAULT_SPEAKER_MAP_FILENAME})")

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

    # --- NEW Normalization Arguments ---
    parser.add_argument("--normalize-numbers", action='store_true', default=DEFAULT_NORMALIZE_NUMBERS, help="Convert digits to words (e.g., 5 -> five).")
    parser.add_argument("--remove-fillers", action='store_true', default=DEFAULT_REMOVE_FILLERS, help="Remove common filler words (um, uh, like, etc.). See DEFAULT_FILLER_WORDS in script.")

    cli_args = parser.parse_args()

    # --- Run Main Pipeline ---
    main(cli_args)
