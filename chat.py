import ollama

messages = []

print("Chat with your local model! Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    messages.append({'role': 'user', 'content': user_input})

    print("Bot: ", end="", flush=True)

    full_response = ""
    stream = ollama.chat(
        model='llama3.2',
        messages=messages,
        stream=True
    )

    for chunk in stream:
        content = chunk['message']['content']
        print(content, end="", flush=True)
        full_response += content

    print("\n")  # blank line after each response

    messages.append({'role': 'assistant', 'content': full_response})