import vosk
import json
import queue
import sounddevice as sd
import pyttsx3
from rapidfuzz import process

# Load Vosk model for offline speech recognition
MODEL_PATH = "models/vosk-model-en-us-0.22"
asr_model = vosk.Model(MODEL_PATH)

def recognize_speech():
    """Capture audio and return recognized text."""
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                            channels=1, callback=callback):
        recognizer = vosk.KaldiRecognizer(asr_model, 16000)
        print("Listening...")
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                return result.get("text", "")


def intent_recognition(text):
    """Fuzzy match intent recognition with input normalization."""
    text = text.lower().strip()  # Normalize text input
    
    commands = {
        "turn off": "power_off",
        "power off": "power_off",
        "turn on": "power_on",
        "power on": "power_on",
        "increase volume": "volume_up",
        "volume up": "volume_up",
        "decrease volume": "volume_down",
        "volume down": "volume_down",
        "lower volume": "volume_down",
        "play music": "play_music",
        "start music": "play_music",
        "pause music": "pause_music",
        "stop music": "pause_music",
        "exit": "exit"
    }

    result = process.extractOne(text, commands.keys())

    if result:
        match, score = result[0], result[1]  # Unpack correctly
        if score > 85:  # Increase confidence threshold
            return commands[match]

    return "unknown"


def execute_command(intent):
    """Execute predefined commands based on intent."""
    if intent == "exit":
        print("Exiting...")
        speak_response("Goodbye!")
        exit()
    
    actions = {
        "power_off": lambda: print("Turning off the device..."),
        "power_on": lambda: print("Turning on the device..."),
        "volume_up": lambda: print("Increasing volume..."),
        "volume_down": lambda: print("Decreasing volume..."),
        "play_music": lambda: print("Playing music..."),
        "pause_music": lambda: print("Pausing music..."),
    }
    action = actions.get(intent, lambda: print("Unknown command."))
    action()


def speak_response(text):
    """Convert text to speech using pyttsx3."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def main():
    """Main loop for the AI agent."""
    while True:
        spoken_text = recognize_speech()
        print(f"User said: {spoken_text}")
        intent = intent_recognition(spoken_text)
        execute_command(intent)
        speak_response(f"Command executed: {intent}")

if __name__ == "__main__":
    main()
