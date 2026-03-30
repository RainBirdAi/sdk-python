# Rainbird Python SDK

A Python SDK for interacting with the Rainbird API. This library provides a simple interface for working with Rainbird knowledge maps, allowing you to start sessions, inject facts, query knowledge, and interact with the reasoning engine.

---

## Installation

```bash
pip install rainbird-python-sdk
Quick Start
from rainbird_python_sdk import RainbirdAPIClient

# Initialise the client
client = RainbirdAPIClient(
    api_host="https://api.rainbird.ai",
    api_key="your-api-key",
    km_id="your-knowledge-map-id"
)

# Start a session
session = client.start()
print(f"Session started: {session['id']}")

# Inject a fact
client.inject_fact(
    subject="John",
    relationship="age",
    object_value="30",
    certainty=100
)

# Query the knowledge map
result = client.query(
    subject="John",
    relationship="has age"
)

# Handle questions
if 'question' in result:
    # Answer the latest question directly
    response = client.answer("30", certainty=100)

# Close the session
client.close()
Using Context Manager
from rainbird_python_sdk import RainbirdAPIClient

with RainbirdAPIClient(
    api_host="https://api.rainbird.ai",
    api_key="your-api-key",
    km_id="your-km-id"
) as client:
    client.start()
    result = client.query(subject="John", relationship="age")
    # Session automatically closed when exiting the context
Improved Question Handling

The SDK provides flexible and Pythonic ways to respond to questions from the reasoning engine.

Key Features
response() accepts:
a single answer dictionary
or a list of answer dictionaries
High-level helper methods:
answer_question() — answer a specific question
answer() — answer the latest question automatically
Built-in validation for incorrect or incomplete inputs
API Reference
Initialisation
client = RainbirdAPIClient(api_host, api_key, km_id)
api_host: The Rainbird API host URL
api_key: Your API key for authentication
km_id: The knowledge map ID
Core Methods
start(use_draft=None, version=None)

Start a new Rainbird session.

Returns: Session information dictionary

inject(facts)

Inject multiple facts into the session.

inject_fact(subject, relationship, object_value, certainty=100)

Inject a single fact.

query(subject=None, relationship=None, object=None)

Query the knowledge map.

relationship is required

Returns: May include question or result

response(answers)

Provide answers to the reasoning engine.

Accepts:
a single answer dict
or a list of answer dicts
Each answer includes:
subject
relationship
object
certainty
First Form answers also include:
answer ("yes" / "no")
answer_question(question, answer_or_answers, certainty=100)

Answer a specific question.

First Form → single value ("yes"/"no")
Second Form → single value or list
answer(answer_or_answers, certainty=100)

Answer the latest question automatically.

answer_first_form(...)

Answer a yes/no question.

answer_second_form(...)

Answer a value-based question.

undo()

Undo the last action.

Natural Language Methods
interact(user_prompt)

Send natural language queries.

explain(fact_id, language='en')

Get explanation for a fact.

Analysis Methods
evidence(fact_id)

Get reasoning evidence.

interaction_log(format='csv')

Get session interaction log.

session_info()

Get session details.

Platform Methods
version()

Get API version info.

Examples
Using answer()
client.start()

result = client.query(subject="John", relationship="age")

if 'question' in result:
    result = client.answer("30", certainty=100)
Using answer_question()
result = client.query(subject="John", relationship="skills")

if 'question' in result:
    question = result['question']
    result = client.answer_question(question, "Python", certainty=100)
Multiple Answers (Second Form)
result = client.query(subject="John", relationship="skills")

if 'question' in result:
    question = result['question']
    result = client.answer_question(
        question,
        ["Python", "SQL"],
        certainty=100
    )
Advanced: Using response() directly
client.start()

result = client.query(relationship="employee_eligible")

if 'question' in result:
    question = result['question']
    client.response([
        {
            "subject": question['subject'],
            "relationship": question['relationship'],
            "object": "Spanish",
            "certainty": 100
        }
    ])
Error Handling
try:
    result = client.query(subject="test", relationship="age")
except ValueError as e:
    print(e)
except Exception as e:
    print(e)
Requirements
Python >= 3.8
requests >= 2.25.0
License

MIT License

Support
GitHub Issues: TODO
Email: julio@rainbird.ai
Docs: https://rainbird.redoc.ly/#operation/explain
About Rainbird

Rainbird is an AI platform for building knowledge maps for automated reasoning and decision-making.
