from langchain.agents.openai_assistant import OpenAIAssistantRunnable
import os

os.environ["OPENAI_API_KEY"] = "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"

interpreter_assistant = OpenAIAssistantRunnable(
    assistant_id="asst_3nj6RUOBClO7ZusVBQgl1yjY"
)
output = interpreter_assistant.invoke({"content": "What's 10 - 4 raised to the 2.7"})

print(output[0].content[0].text.value)
