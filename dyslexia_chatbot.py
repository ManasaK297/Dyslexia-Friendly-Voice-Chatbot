import speech_recognition as sr
import pyttsx3
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load Hugging Face model (Facebook Blenderbot)
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech speed

def speak_text(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def listen():
    """Capture speech and convert to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Speak now.")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            print(f"User said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return "error"
        except sr.RequestError:
            print("Speech Recognition service is unavailable.")
            return "error"
        except sr.WaitTimeoutError:
            print("No speech detected. Please try again.")
            return "error"

def get_dyslexia_friendly_response(user_text):
    """Generate a dyslexia-friendly AI response using Blenderbot."""
    if user_text == "error":
        return "I couldn't understand you. Please try speaking again."
    
    inputs = tokenizer([user_text], return_tensors="pt")
    output_ids = model.generate(**inputs)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def main():
    """Main chatbot loop."""
    print("\nDyslexia Communication AI - Speak Something Mode\n")
    speak_text("Hello! I'm here to help. Speak now.")
    
    while True:
        user_input = listen()
        if user_input.lower() in ["exit", "quit", "stop"]:
            speak_text("Goodbye! Have a great day.")
            break

        response = get_dyslexia_friendly_response(user_input)
        print(f"AI: {response}")
        speak_text(response)

if __name__ == "__main__":
    main()
