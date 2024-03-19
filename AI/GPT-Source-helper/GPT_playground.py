import os
import openai

openai.api_key ="sk"


system_intel = "You are GPT-3.5-turbo, answer my questions as if you were an expert in the field."
prompt = "What is machine learning?"

def ask_GPT35 (system_intel, prompt):
  result = openai.ChatCompletion.create (
    model="gpt-3.5-turbo-0301",
    messages= [
      {"role": "system", "content": system_intel},
      {"role": "user", "content": prompt}
    ])
  print (result ['choices'] [0] ['message'] ['content'])

def promptada(prompt):
    result = openai.Completion.create (
        engine="davinci",
        prompt=prompt,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " AI:"]
    )
    print (result ['choices'] [0] ['text'])

#promptada("You are GPT-3.5-turbo, answer my questions as if you were an expert in the field. What is machine learning?")
ask_GPT35 (system_intel, prompt)
