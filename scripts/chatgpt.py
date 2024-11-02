from openai import OpenAI
client = OpenAI()


# Create the chat completion request
response = client.chat.completions.create(
    model="gpt-4",  # Change to a valid model name
    messages=[
        {
            "role": "system",
            "content": (
                "You are a knowledge base for Traditional Chinese Medicine. "
                "Provide detailed information about the given TCM concept, "
                "including its definition, symptoms, causes, and potential treatments."
            ),
        },
        {
            "role": "user",
            "content": "Liver qi stagnation.",
        },
    ],
)

# Print the response from the assistant
print(response.choices[0].message["content"])

