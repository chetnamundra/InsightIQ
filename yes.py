from g4f.client import Client
import cv2
import speech_recognition as sr
import threading

client = Client()

def model_response(system, user):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}]
    )
    return response.choices[0].message.content

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

# Initialize the SpeechRecognition recognizer
recognizer = sr.Recognizer()

print("yes")
user = "Hello, my name is xxx, we can start with the interview, please ask me my first question"
system = "you are an interviewer and you have to ask technical question on python to a college graduate. Ask first question"

qno = "1"
overall_conversation = "user: " + user + "\nsystem: " + system + "\n"

x = model_response(system, user)
print(x)
overall_conversation += "system (question number " + qno + "): " + x + "\n"

# Wait for the user's response
while True:
    
    
    response = recognize_speech(recognizer)
    
    if response.strip().lower() == "end interview":
        break
    
    qno = str(int(qno) + 1)
    system = "ask question number " + qno +"to the user for the inertview. do not give
    
    x = model_response(system, response)
    overall_conversation += "system (question number " + qno + "): " + x + "\n"
    print(x)
    

# Analyze user's performance in the interview and give tips
system = "analyze user's performance in the interview and give tips. Keep your answer short and to the point."

print(overall_conversation)
x = model_response(system, overall_conversation)
print(x)

# Release the capture
cap.release()
cv2.destroyAllWindows()
