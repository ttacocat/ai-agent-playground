from langchain_ollama import ChatOllama

if __name__ == "__main__":
    llm = ChatOllama(
        model="llama3.1:8b"
    )
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    # res = llm.invoke(messages)
    # print(res.content)
    res = llm.stream(messages)
    for chunk in res:
        print(chunk.content, end = "")
