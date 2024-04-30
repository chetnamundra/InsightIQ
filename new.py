from g4f.client import Client
import cv2
import speech_recognition as sr
import threading

import asyncio
from asyncio import WindowsSelectorEventLoopPolicy

asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

client = Client()

# Define threading locks
model_lock = threading.Lock()
speech_lock = threading.Lock()

def model_response(system, user):
    with model_lock:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}]
        )
        return response.choices[0].message.content

# Function to perform speech recognition asynchronously

def recognize_speech(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio)
        print("Recognized:", text)
        return text
    except sr.RequestError as e:
        print("Error fetching results:", e)
        return ""
    except Exception as e:
        # Other exceptions can occur, but we handle them silently here
        return ""

# Start a thread for speech recognition

def speech_thread(recognizer):
    global recognized_text
    timeout_occurred = False
    while not timeout_occurred:
        recognized_text = ""
        with speech_lock:
            with sr.Microphone() as source:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(source)
                try:
                    audio = recognizer.listen(source, timeout=20)  # Listen for 20 seconds
                    recognized_text += recognize_speech(recognizer, audio)
                    if recognized_text=="":
                        return
                except sr.WaitTimeoutError:
                    print("Timeout reached. Moving to the next question.")
                    timeout_occurred = True



                
# Initialize the SpeechRecognition recognizer
recognizer = sr.Recognizer()

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Function to show video frames
def show_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start the video display thread
video_thread = threading.Thread(target=show_frames)
video_thread.start()

# Start the speech recognition thread


# Start interview
user = "Hello, my name is xxx, we can start with the interview, please ask me my first question"
system = "you are an interviewer and you have to ask technical question on python to a college graduate. Ask first question"

qno = "1"
overall_conversation = "user: " + user + "\nsystem: " + system + "\n"

x = model_response(system, user)
print(x)
overall_conversation += "system (question number " + qno + "): " + x + "\n"

# Wait for the user's response
while True:
    recognized_text = ""
    # Start the speech recognition thread
    speech_recognition_thread = threading.Thread(target=speech_thread, args=(recognizer,))
    speech_recognition_thread.start()
    
    # Join the speech recognition thread to ensure it has finished before proceeding
    speech_recognition_thread.join()
    
    response = recognized_text
    
    # Proceed to ask the next question if "end interview" is not detected in the response
    if "end interview" not in response or qno=="2":
        qno = str(int(qno) + 1)
        system = "ask question number " + qno
        
        x = model_response(system, response)
        overall_conversation += "system (question number " + qno + "): " + x + "\n"
        print(x)
    else:
        # End the interview if "end interview" is detected
        break

# Analyze user's performance in the interview and give tips
system = "analyze user's performance in the interview and give tips. Keep your answer short and to the point."


print(overall_conversation)
x = model_response(system, overall_conversation)
print(x)

# Release the capture
cap.release()
cv2.destroyAllWindows()
