import cv2
import speech_recognition as sr
import threading
from g4f.client import Client
import queue

# Initialize the GPT-3.5 model client
client = Client()

import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Function to perform speech recognition asynchronously
def recognize_speech(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio)
        print("Recognized:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print("Error fetching results:", e)
        return ""

# Function to interact with the GPT-3.5 model
def interact_with_model(user_response):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """You are an interviewer and you have to ask technical questions to a college graduate
            in aiml. Ask questions one by one till you are satisfied with the candidate's knowledge and give the feedback once the user says thank you.
            """},
            {"role": "user", "content": user_response}
        ]
    )
    return response.choices[0].message.content

# Function to continuously capture and display video
def video_capture(response_queue):
    # Initialize the video capture
    cap = cv2.VideoCapture(0)
    fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the SpeechRecognition recognizer
    recognizer = sr.Recognizer()

    # Function for speech recognition thread
    def speech_thread():
        while True:
            with sr.Microphone() as source:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
                recognized_text = recognize_speech(recognizer, audio)
                if recognized_text:
                    response_queue.put(recognized_text)

    # Start the speech recognition thread
    speech_recognition_thread = threading.Thread(target=speech_thread)
    speech_recognition_thread.start()

    # Loop for interaction with the model
    while True:
        # Get question from the model
        model_question = interact_with_model("")

        # Wait for user response
        user_response = response_queue.get()

        # Interact with the model using the user's response
        model_response = interact_with_model(user_response)
        print("Model:", model_response)

        # Check if the conversation should end
        if "thank you" in model_response.lower():
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

# Start the video capture thread
response_queue = queue.Queue()
video_capture_thread = threading.Thread(target=video_capture, args=(response_queue,))
video_capture_thread.start()
