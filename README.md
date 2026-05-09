# YTtranscribe

Streamlit app that downloads a YouTube video, transcribes it with Whisper,
and lets you semantic-search across all stored transcriptions using BERT
embeddings.

## Setup

```sh
pip install -r requirements.txt
```

`yt-dlp` shells out to `ffmpeg`. If it isn't on `PATH`, set
`FFMPEG_PATH=/path/to/ffmpeg` in your environment.

## Run

```sh
streamlit run app.py
```

UI surfaces:
- **Search** — natural-language query over previously transcribed videos.
- **Add Video** — paste a YouTube URL; downloads, transcribes, indexes.
- **Clear Library** — wipes the local download folder + transcription store.

## How it works

1. `yt-dlp` downloads the video; `moviepy` extracts the audio track.
2. The audio is chunked at 30 s and run through `distil-whisper/distil-small.en`
   (English-only by default; override with `WHISPER_MODEL=...` and optionally
   `WHISPER_LANGUAGE=...` for multilingual sources).
3. The full transcript and a BERT `[CLS]` mean-pool embedding are persisted
   to `video_transcriptions.json`. Embeddings are cached so search doesn't
   recompute on every query.
4. Search compares the input text's BERT embedding to each stored embedding
   via cosine similarity, threshold 0.6.

## Notes

- First run downloads ~500 MB of model weights (Whisper + BERT). They cache
  under your Hugging Face cache after that.
- For non-English videos, set `WHISPER_MODEL=openai/whisper-base` (or any
  multilingual variant) and `WHISPER_LANGUAGE=hi` (or whatever).
