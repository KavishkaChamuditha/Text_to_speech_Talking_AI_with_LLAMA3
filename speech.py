# commands to run before starting this
# pip install ollama
# python -m pip install pyaudio
# pip install "assemblyai[extras]"
# pip install elevenlabs


import assemblyai as aai
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import ollama

class AI_Speech:
    def __init__(self):
        print("Initializing AI_Speech...")
        aai.settings.api_key = ""
        self.client = ElevenLabs(api_key="")
        self.transcriber = None
        self.full_transcript = [
            {"role": "system", "content": "Language Model"},
        ]
        print("AI_Speech initialized.")

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("Session opened with ID:", session_opened.session_id)

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            print("Final Transcript:", transcript.text)
            self.generate_ai_response(transcript)
        else:
            print("Partial Transcript:", transcript.text, end="\r")

    def on_error(self, error: aai.RealtimeError):
        print("An error occurred:", error)

    # Using to real-time speaking stream
    def start_transcription(self):
        def on_close():
            print("Closing Session")
            return

        print("Starting transcription")
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16_000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=on_close,
        )

        self.transcriber.connect()
        print("Connected to transcriber.")

        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
        self.transcriber.stream(microphone_stream)
        print("Streaming from microphone...")

    def close_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None
            print("Transcription closed.")

    # Pass real-time LLMA 3
    def generate_ai_response(self, transcript):
        self.close_transcription()

        self.full_transcript.append({"role": "user", "content": transcript.text})
        print(f"\nUser: {transcript.text}", end="\r\n")

        ollama_stream = ollama.chat(
            model="llama3",
            messages=self.full_transcript,
            stream=True,
        )

        print("Llama 3:", end="\r\n")
        text_buffer = ""
        full_text = ""
        for chunk in ollama_stream:
            text_buffer += chunk['message']['content']
            if text_buffer.endswith('.'):
                audio_stream = self.client.generate(
                    text=text_buffer,
                    model="eleven_turbo_v2",
                    stream=True
                )
                print(text_buffer, end="\n", flush=True)
                stream(audio_stream)
                full_text += text_buffer
                text_buffer = ""

        if text_buffer:
            audio_stream = self.client.generate(
                text=text_buffer,
                model="eleven_turbo_v2",
                stream=True
            )
            print(text_buffer, end="\n", flush=True)
            stream(audio_stream)
            full_text += text_buffer

        self.full_transcript.append({"role": "assistant", "content": full_text})

        self.start_transcription()

if __name__ == "__main__":
    print("Creating AI_Speech")
    ai_assistant = AI_Speech()
    ai_assistant.start_transcription()
    print("AI_Speech instance created and transcription started.")
