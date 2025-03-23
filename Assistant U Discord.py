from openai import OpenAI
client = OpenAI()

"""assistant = client.beta.assistants.create(
  name="Math Tutor",
  instructions="You are a personal math tutor. Write and run code to answer math questions.",
  tools=[{"type": "code_interpreter"}],
  model="gpt-4o",
)"""



assistant = client.beta.assistants.retrieve(assistant_id="asst_N8LzHJY29NSDsNpC9bc8Oh6E")


thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Who have had the biggest damage?"
)

run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Answer in a snappy wit, but be always thruthful to the documentation"
)

if run.status == 'completed':
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    for message in messages:
        print(f"{message.role}: {message.content[0].text.value}")
    print(messages)
else:
    print(run.status)