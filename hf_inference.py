from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
client = InferenceClient(
	# provider="together", # optional, default is huggingface's own inference API
	api_key = os.getenv("HUGGINGFACE_API_KEY")
)

messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct", 
	messages=messages, 
	max_tokens=500
)

print(completion.choices[0].message.content)