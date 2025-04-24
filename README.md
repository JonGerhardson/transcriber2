Work in progress. Currently, this will transcribe an audio file, add per-word timestamps, and label each speaker. 

To run: 
- git clone this repo, cd transcriber2
- accept the terms for gated model access for pyannote.audio: 
[https://huggingface.co/pyannote/segmentation-3.0
](https://huggingface.co/pyannote/speaker-diarization-3.1)

[https://huggingface.co/pyannote/speaker-diarization-3.1
](https://huggingface.co/pyannote/speaker-diarization-3.1
)

-Generate a hugging face token with read access to your gated repos, open hf-token.txt, delete the filler text, and save your token as the only thing in the file. 

- make a folder called "audio_input" and save your audio file there

- Adjust file paths as nessecary, and then run the following docker command. The command below assumes you saved this repo in your home folder.

  ```
  sudo docker run --gpus all --rm --name dear-diary \
  -v "/home/YOURUSERNAME/transcriber2/audio_input:/app/audio_input" \
  -v "/home/YOURUSERNAME/transcriber2/transcripts_output:/app/transcripts_output" \
  -v "/home/YOURUSERNAME/transcriber2//hf-token.txt:/app/hf-token.txt" \
  whispxpyan-image
  ```

Transcription happens on CPU due to some weird pytorch dependency errors I couldn't figure out that may have to do with the age of my particular GPU. Diarization uses CUDA. 


  
Current output is json structured like this 

```
  {
    "start":125.66,
    "end":130.78,
    "text":" And then we will open it up for public testimony.",
    "words":[
      {
        "word":" And",
        "start":125.66,
        "end":126.4,
        "probability":0.9895156622,
        "speaker":"SPEAKER_06"
      },
      {
        "word":" then",
        "start":126.4,
        "end":127.62,
        "probability":0.995975554,
        "speaker":"UNKNOWN"
      },
      {
        "word":" we",
        "start":127.62,
        "end":127.8,
        "probability":0.9940746427,
        "speaker":"SPEAKER_06"
      },
      {
        "word":" will",
        "start":127.8,
        "end":128.06,
        "probability":0.9984428287,
        "speaker":"SPEAKER_06"
      },
      {
        "word":" open",
        "start":128.06,
        "end":129.12,
        "probability":0.9997921586,
        "speaker":"UNKNOWN"
      },
      {
        "word":" it",
        "start":129.12,
        "end":129.28,
        "probability":0.9995173216,
        "speaker":"SPEAKER_06"
      },
      {
        "word":" up",
        "start":129.28,
        "end":129.44,
        "probability":0.9995858073,
        "speaker":"SPEAKER_06"
      },
      {
        "word":" for",
        "start":129.44,
        "end":129.66,
        "probability":0.9997250438,
        "speaker":"SPEAKER_06"
      },
      {
        "word":" public",
        "start":129.66,
        "end":130.26,
        "probability":0.998608768,
        "speaker":"SPEAKER_06"
      },
      {
        "word":" testimony.",
        "start":130.26,
        "end":130.78,
        "probability":0.9995806813,
        "speaker":"SPEAKER_06"
      }
    ]
  },
  {
```
