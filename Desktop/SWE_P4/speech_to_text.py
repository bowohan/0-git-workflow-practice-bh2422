import assemblyai as aai
import os

from functions import parse_text, most_common_dict, vocab_diversity

aai.settings.api_key = "2a72f8c7617e4985b9a3495e16375e07"  
transcriber = aai.Transcriber()

audio_file = "/Users/bohanhou/Desktop/test.m4a"

config = aai.TranscriptionConfig()

transcript = transcriber.transcribe(audio_file, config)

if transcript.status == aai.TranscriptStatus.error:
    print(f"Transcription failed: {transcript.error}")
    exit(1)

output_folder = "/Users/bohanhou/Desktop/SWE_P4"
transcript_file = os.path.join(output_folder, "transcript.txt")
word_count_file = os.path.join(output_folder, "count.txt")

os.makedirs(output_folder, exist_ok=True)

with open(transcript_file, 'w') as f:
    f.write(transcript.text + '\n')

most_common_word, diversity, word_freqs = vocab_diversity(transcript.text)

#sort word frequencies (descending)
sorted_word_counts = dict(sorted(word_freqs.items(), key=lambda item: item[1], reverse=True))

with open(word_count_file, 'w') as f:
    for word, count in sorted_word_counts.items():
        f.write(f"{word}: {count}\n")

#print most common word and vocab diversity
print(f"Most common word: {most_common_word}")
print(f"Vocabulary diversity: {diversity}")

for word, count in sorted_word_counts.items():
    print(f"{word}: {count}")