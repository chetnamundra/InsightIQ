import cv2
import speech_recognition as sr
import threading

from g4f.client import Client

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

# Initialize the video capture
cap = cv2.VideoCapture(0)
fps = 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the SpeechRecognition recognizer
recognizer = sr.Recognizer()

# Start a thread for speech recognition
def speech_thread():
    global recognized_text
    while True:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            recognized_text = recognize_speech(recognizer, audio)

# Start the speech recognition thread
recognized_text = ""
speech_recognition_thread = threading.Thread(target=speech_thread)
speech_recognition_thread.start()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the recognized text on the frame
    cv2.putText(frame, recognized_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
