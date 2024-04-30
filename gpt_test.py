from g4f.client import Client

client=Client()
userre="hi my name is chetna and im here for technical interview"
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{
        "role" : "system" , "content" : """you are an interviwer and
        you have to ask technical questions to a college graduate and
        you will be given users response and you have to analyse the
        response and give user feedback on how he has to improve"""}, 
        {"role" : "user", "content" : userre
        }]
    )
print(response.choices[0].message.content)
