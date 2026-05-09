import os
import yt_dlp
import json
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
from transformers import WhisperProcessor, WhisperForConditionalGeneration, BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
import re

# distil-whisper/distil-small.en — English-only, ~5x faster than whisper-tiny
# at materially higher accuracy. Override via env var if you need
# multilingual or a different size.
WHISPER_MODEL_ID = os.environ.get("WHISPER_MODEL", "distil-whisper/distil-small.en")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID)

# English-only checkpoints (.en) bake in transcription-in-English; multilingual
# checkpoints will silently translate unless we force language + task. Modern
# transformers (4.40+) takes language= and task= directly on generate(); the
# older forced_decoder_ids API is deprecated in 4.50+.
IS_ENGLISH_ONLY = WHISPER_MODEL_ID.endswith(".en")

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def download_youtube_video(video_url, output_folder="download"):
    """Download a YouTube video to the specified output folder."""
    video_file_name = "VideoForTranscription.webm"  # Fixed file name
    video_path = os.path.join(output_folder, video_file_name)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        'outtmpl': video_path,
        'quiet': False,
    }
    ffmpeg_path = os.environ.get("FFMPEG_PATH")
    if ffmpeg_path:
        ydl_opts['ffmpeg_location'] = ffmpeg_path

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        video_title = info_dict.get('title', None)

    print(f"Download completed! Video saved to: {video_path}")
    return video_path, video_title


def extract_audio(video_path, output_audio_path):
    """Extract audio from the video file."""
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(output_audio_path)
    video_clip.close()


def split_audio(audio, sr, segment_duration=30):
    """Split audio into chunks of the specified duration (in seconds)."""
    total_duration = len(audio) / sr
    segments = []
    start = 0

    while start < total_duration:
        end = min(start + segment_duration, total_duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segments.append(audio[start_sample:end_sample])
        start = end

    return segments


def transcribe_audio_chunk(audio_chunk, sr):
    """Transcribe a single chunk of audio."""
    inputs = processor(audio_chunk, return_tensors="pt", sampling_rate=sr)

    gen_kwargs = {}
    if not IS_ENGLISH_ONLY:
        gen_kwargs["language"] = WHISPER_LANGUAGE
        gen_kwargs["task"] = "transcribe"

    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features, **gen_kwargs)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription


def transcribe_audio(audio_path, segment_duration=30):
    """Transcribe the entire audio file by splitting it into chunks."""
    audio, sr = librosa.load(audio_path, sr=16000)
    audio_chunks = split_audio(audio, sr, segment_duration)

    transcriptions = []
    for chunk in audio_chunks:
        transcription = transcribe_audio_chunk(chunk, sr)
        transcriptions.append(transcription)

    full_transcription = ' '.join(transcriptions)
    return full_transcription


def transcribe_video(video_path):
    """Complete process of extracting and transcribing audio from video."""
    audio_path = os.path.join(os.path.dirname(video_path), 'extracted_audio.wav')

    extract_audio(video_path, audio_path)
    transcription = transcribe_audio(audio_path, segment_duration=30)
    if os.path.exists(video_path):
        os.remove(video_path)

    return transcription


def _load_store(json_file):
    if not os.path.exists(json_file):
        return {}
    with open(json_file, 'r') as f:
        return json.load(f)


def _write_store(json_file, data):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


def save_transcription_to_json(video_link, transcription, json_file='video_transcriptions.json'):
    """
    Save the transcription AND its precomputed BERT embedding to the JSON
    store so search doesn't have to recompute embeddings on every query.
    Schema: {video_link: {"transcription": str, "embedding": list[float]}}.
    """
    data = _load_store(json_file)

    cleaned = clean_text(transcription)
    embedding = compute_bert_embeddings(cleaned).tolist() if is_meaningful(cleaned) else None

    data[video_link] = {
        "transcription": transcription,
        "embedding": embedding,
    }

    _write_store(json_file, data)
    print(f"Transcription saved to {json_file}.")


nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))


def clean_text(text):
    """Preprocess text to remove special characters, numbers, and extra whitespace."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def is_meaningful(text):
    """Check if the text has a significant number of non-stopword words."""
    words = text.split()
    meaningful_words = [w for w in words if w not in STOP_WORDS]
    return len(meaningful_words) > 2


def compute_bert_embeddings(text, max_chunk_tokens=510):
    """
    Compute a BERT embedding for `text` using the [CLS] token of bert-base-uncased.

    Long inputs are chunked to fit BERT's 512-token limit (510 + [CLS] + [SEP])
    and the resulting per-chunk embeddings are mean-pooled to a single 768-d vector.
    """
    if not text or not text.strip():
        return np.zeros(768, dtype=np.float32)

    token_ids = bert_tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return np.zeros(768, dtype=np.float32)

    chunk_embeddings = []
    cls_id = bert_tokenizer.cls_token_id
    sep_id = bert_tokenizer.sep_token_id

    for start in range(0, len(token_ids), max_chunk_tokens):
        chunk = token_ids[start:start + max_chunk_tokens]
        input_ids = torch.tensor([[cls_id, *chunk, sep_id]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = bert_model(input_ids=input_ids, attention_mask=attention_mask)

        cls_vector = output.last_hidden_state[0, 0, :].cpu().numpy()
        chunk_embeddings.append(cls_vector)

    return np.mean(chunk_embeddings, axis=0)


def normalize_embeddings(embeddings):
    """Normalize embeddings to unit vectors."""
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True) if embeddings.ndim > 1 else np.linalg.norm(embeddings)
    return embeddings / (norm + 1e-10)


def _resolve_entry(video_link, entry, store):
    """
    Resolve a store entry to (cleaned_text, embedding_array_or_None).

    Handles both the new dict schema and the legacy "transcription as a bare
    string" schema. Legacy entries are migrated in-place: their embedding is
    computed once and persisted so the next search is fast.
    """
    if isinstance(entry, str):
        cleaned = clean_text(entry)
        emb = compute_bert_embeddings(cleaned) if is_meaningful(cleaned) else None
        store[video_link] = {
            "transcription": entry,
            "embedding": emb.tolist() if emb is not None else None,
        }
        return cleaned, emb

    transcription = entry.get("transcription", "")
    cleaned = clean_text(transcription)
    cached = entry.get("embedding")
    if cached is not None:
        return cleaned, np.array(cached, dtype=np.float32)
    if not is_meaningful(cleaned):
        return cleaned, None
    emb = compute_bert_embeddings(cleaned)
    entry["embedding"] = emb.tolist()
    return cleaned, emb


def search_transcriptions_by_context(
    input_text,
    json_file='video_transcriptions.json',
    similarity_threshold=0.6,
):
    """Search for videos whose transcriptions match the context of the input."""
    if not os.path.exists(json_file):
        print("No transcriptions found.")
        return []

    with open(json_file, 'r') as f:
        data = json.load(f)

    input_text = clean_text(input_text)
    if not is_meaningful(input_text):
        print("Input text is not meaningful enough for searching.")
        return []

    if len(input_text.split()) == 1:
        input_text = f"The topic is about {input_text}."

    input_embeddings = normalize_embeddings(
        compute_bert_embeddings(input_text).reshape(1, -1)
    )

    matching_videos = []
    store_dirty = False

    for video_link, entry in list(data.items()):
        before_emb = entry.get("embedding") if isinstance(entry, dict) else None
        cleaned, transcription_emb = _resolve_entry(video_link, entry, data)
        if isinstance(data.get(video_link), dict):
            after_emb = data[video_link].get("embedding")
            if after_emb != before_emb:
                store_dirty = True

        if transcription_emb is None:
            continue

        transcription_emb = normalize_embeddings(transcription_emb.reshape(1, -1))
        similarity = cosine_similarity(input_embeddings, transcription_emb)[0][0]

        if similarity > similarity_threshold:
            matching_videos.append(video_link)

    if store_dirty:
        _write_store(json_file, data)

    return matching_videos
