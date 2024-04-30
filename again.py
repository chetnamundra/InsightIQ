import cv2
from deepface import DeepFace
from g4f.client import Client
import speech_recognition as sr
import threading

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the GPT-3.5 Turbo client
client = Client()

# Initialize the SpeechRecognition recognizer
recognizer = sr.Recognizer()

# Dictionary to track emotion frequencies
emotion_frequency = {}

# Function to perform emotion analysis on a face ROI
def analyze_emotion(face_roi):
    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']
    return emotion

# Function to send a message to GPT-3.5 Turbo and get the response
def model_response(system, user):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}]
    )
    return response.choices[0].message.content

# Function to recognize speech using the microphone
def recognize_speech(recognizer):
    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=20)  # Listen for 20 seconds
            text = recognizer.recognize_google(audio)
            print("Recognized:", text)
            return text
    except sr.WaitTimeoutError:
        print("Timeout reached. Moving to the next question.")
        return ""
    except sr.RequestError as e:
        print("Error fetching results:", e)
        return ""
    except Exception as e:
        print("Error:", e)
        return ""

# Function to capture video frames and analyze emotions
def capture_frames():
    cap = cv2.VideoCapture(0)
    qno = 1
    overall_conversation = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert grayscale frame to RGB format
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]
            
            # Analyze emotion
            emotion = analyze_emotion(face_roi)
            
            # Update emotion frequency dictionary
            emotion_frequency[emotion] = emotion_frequency.get(emotion, 0) + 1
            
            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Ask a question
        system = f"Ask question number {qno} to the user for the interview."
        question = model_response(system, "")
        print("Question:", question)
        overall_conversation += f"system (question number {qno}): {question}\n"
        
        # Wait for the user's response
        response = recognize_speech(recognizer)
        overall_conversation += f"user (answer number {qno}): {response}\n"
        print("Response:", response)
        
        # Check if the interview is over
        if response.strip().lower() == "end interview":
            break
        
        qno += 1
    
    # Analyze user's performance in the interview and give tips
    system = "analyze user's performance in the interview and give tips."
    feedback = model_response(system, overall_conversation)
    print("Feedback:", feedback)
    
    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Start video capture and emotion analysis in a separate thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()
