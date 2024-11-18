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

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def download_youtube_video(video_url, output_folder=r"/Users/kuljeetsinghshekhawat/Documents/coding/YTtranscribe/download"):
    """Download a YouTube video to the specified output folder."""
    video_file_name = "VideoForTranscription.webm"  # Fixed file name
    video_path = os.path.join(output_folder, video_file_name)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        'outtmpl': video_path,  # File will be saved as VideoForTranscription.webm
        'quiet': False,  # Set to False to show download progress
        'ffmpeg_location': r"/opt/homebrew/bin/ffmpeg",  # Provide the path to ffmpeg if necessary
    }



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

    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features)
    
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

    # Combine all transcriptions
    full_transcription = ' '.join(transcriptions)
    return full_transcription

def transcribe_video(video_path):
    """Complete process of extracting and transcribing audio from video."""
    audio_path = os.path.join(os.path.dirname(video_path), 'extracted_audio.wav')

    # Step 1: Extract audio from video
    extract_audio(video_path, audio_path)

    # Step 2: Transcribe the extracted audio with chunking
    transcription = transcribe_audio(audio_path, segment_duration=30)
    os.remove(r"/Users/kuljeetsinghshekhawat/Documents/coding/YTtranscribe/download/VideoForTranscription.webm")

    return transcription

def save_transcription_to_json(video_link, transcription, json_file='video_transcriptions.json'):
    """Save the transcription to a JSON file in the specified format."""
    # Load existing data
    data = {}
    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)

    # Add new transcription
    data[video_link] = transcription

    # Save updated data back to the JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Transcription saved to {json_file}.")

nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Preprocess text to remove special characters, numbers, and extra whitespace.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text.lower()

def is_meaningful(text):
    """
    Check if the text has a significant number of meaningful words.
    """
    words = text.split()
    meaningful_words = [word for word in words if word not in STOP_WORDS]
    return len(meaningful_words) > 2  # Require at least 3 meaningful words

def compute_bert_embeddings(text):
    """
    Compute BERT embeddings for a given text.
    Replace this function with your preferred BERT embedding library (e.g., Hugging Face Transformers).
    """
    # Dummy implementation; replace with actual embedding computation.
    return np.random.rand(768)

def normalize_embeddings(embeddings):
    """Normalize embeddings to unit vectors."""
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True) if embeddings.ndim > 1 else np.linalg.norm(embeddings)
    return embeddings / (norm + 1e-10)

def search_transcriptions_by_context(input_text, json_file='/Users/kuljeetsinghshekhawat/Documents/coding/YTtranscribe/video_transcriptions.json'):
    """
    Search for videos whose transcriptions match the context of the input text.
    """
    if not os.path.exists(json_file):
        print("No transcriptions found.")
        return []

    with open(json_file, 'r') as file:
        data = json.load(file)

    # Clean and validate input text
    input_text = clean_text(input_text)
    if not is_meaningful(input_text):
        print("Input text is not meaningful enough for searching.")
        return []

    # Enrich single-word input by adding context
    if len(input_text.split()) == 1:
        input_text = f"The topic is about {input_text}."

    # Compute normalized input text embeddings
    input_embeddings = normalize_embeddings(compute_bert_embeddings(input_text).reshape(1, -1))

    matching_videos = []
    for video_link, transcription in data.items():
        # Clean and validate transcription
        transcription = clean_text(transcription)
        if not is_meaningful(transcription):
            continue

        # Compute transcription embeddings and normalize
        transcription_embeddings = normalize_embeddings(compute_bert_embeddings(transcription).reshape(1, -1))

        # Calculate cosine similarity
        similarity = cosine_similarity(input_embeddings, transcription_embeddings)[0][0]

        # Use a higher threshold for better filtering
        if similarity > 0.6:  # You can tweak this threshold as per the use case
            matching_videos.append(video_link)

    return matching_videos
