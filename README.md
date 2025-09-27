# RAG-based AI Teaching Assistant Project

This repository contains a Retrieval-Augmented Generation (RAG) based AI Teaching Assistant project for processing video course content and providing relevant responses to user queries.

## Project Description

Developed an AI system that converts video tutorials into searchable chunks, generates embeddings, and uses cosine similarity to retrieve the most relevant segments to answer user queries through a language model.

## File Overview

### 1. `toMp3Throughffmpeg.py`
- Converts course video files in the `videos/` folder to MP3 format and saves them in `audios/`.

### 2. `whisperSpeechToText.py`
- Uses Whisper model to transcribe audio files.
- Splits transcriptions into chunks with metadata: video number, name, chunk ID, timestamps, and text.
- Saves each transcription as a JSON in `audioToJsons/`.

### 3. `embeddingTextChunks.py`
- Converts text chunks into vector embeddings using the BGE-M3 model.
- Stores all chunks and embeddings in a single DataFrame and saves it as a Joblib file `allJsonsWithEmbeddsDataFrame.joblib`.

### 4. `QueryResolving.py`
- Accepts user queries and embeds them using the same BGE-M3 model.
- Finds top 25 relevant text chunks using cosine similarity.
- Passes the filtered chunks to a language model (`phi3`) to generate a concise answer.

## Tools & Libraries

- Python
- Whisper
- BGE-M3 model for embeddings
- scikit-learn (cosine similarity)
- Joblib
- FFmpeg for audio extraction
- Requests module for API calls

## Folder Structure

- `videos/` – input video files
- `audios/` – converted audio files
- `audioToJsons/` – transcription JSON files with chunk metadata
- `allJsonsWithEmbeddsDataFrame.joblib` – saved DataFrame with embeddings
- `QueryResolving.py` – main query interface

## How to Use

1. Place course videos in the `videos/` folder.
2. Run `toMp3Throughffmpeg.py` to convert videos to MP3.
3. Run `whisperSpeechToText.py` to generate transcription JSONs.
4. Run `embeddingTextChunks.py` to create embeddings and save the DataFrame.
5. Run `QueryResolving.py` and input a user query to get relevant answers from the video content.
