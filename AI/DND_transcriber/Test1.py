
import os
import openai
from pydub import AudioSegment

openai.api_key =""



def transcribe_audio(file_path):
    # Load your .wav file
    audio_file = open(file_path, "rb")

    # Transcribe the audio file
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    # Get the text from the transcription
    transcript_text = transcript['text']

    return transcript_text

def summarize_text(text):
    # Summarize the text using GPT-3
    prompt = f"{text}\n\nSummarize:"
    response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=800, temperature=0.6, top_p=0.5, frequency_penalty=0, presence_penalty=0.6, stop=["\n\n"])

    # Get the summary
    summary = response.choices[0].text.strip()

    return summary

# Load your large audio file
audio = AudioSegment.from_wav("DND-introTest.wav")

# Maximum chunk size in bytes (25 MB)
max_chunk_size = 25 * 1024 * 1024

# Initial chunk duration in milliseconds (10 minutes)
chunk_duration = 10 * 60 * 1000

# Directory to save chunks
chunks_dir = "chunks"

# Ensure the directory exists
os.makedirs(chunks_dir, exist_ok=True)

# Full transcript
full_transcript = ""

i = 0
start = 0
while start < len(audio):
    # Create a chunk
    chunk = audio[start:start + chunk_duration]
    chunk_path = os.path.join(chunks_dir, f"chunk{i}.wav")
    chunk.export(chunk_path, format="wav")

    # If the chunk is too large, reduce its duration and try again
    while os.path.getsize(chunk_path) > max_chunk_size:
        chunk_duration -= 60 * 1000  # Reduce duration by 1 minute
        chunk = audio[start:start + chunk_duration]
        chunk.export(chunk_path, format="wav")

    # Transcribe the chunk and add it to the full transcript
    transcript = transcribe_audio(chunk_path)
    full_transcript += transcript + "\n"

    # Move to the next chunk
    start += chunk_duration
    i += 1

# Save the full transcript to a text file
with open("transcript.txt", "w") as file:
    file.write(full_transcript)

# Summarize the full transcript
summary = summarize_text(full_transcript)

# Print the summary
print(f"Summary: {summary}\n")
