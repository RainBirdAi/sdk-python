# Rainbird Python SDK

A Python SDK for interacting with the Rainbird API. This library provides a simple interface for working with Rainbird knowledge maps, allowing you to start sessions, inject facts, query knowledge, and interact with the reasoning engine.

---

## Installation

```bash
pip install rainbird-python-sdk
```

---

## Quick Start

```python
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
    response = client.answer("30", certainty=100)

# Close the session
client.close()
```

---

## Using Context Manager

```python
from rainbird_python_sdk import RainbirdAPIClient

with RainbirdAPIClient(
    api_host="https://api.rainbird.ai",
    api_key="your-api-key",
    km_id="your-km-id"
) as client:
    client.start()
    result = client.query(subject="John", relationship="age")
```

---

## API Reference

### Initialisation

```python
client = RainbirdAPIClient(api_host, api_key, km_id)
```

---

## Core Methods

### start(use_draft=None, version=None)
Start a new Rainbird session.

### inject(facts)
Inject multiple facts into the session.

### inject_fact(subject, relationship, object_value, certainty=100)
Inject a single fact.

### query(subject=None, relationship=None, object=None)
Query the knowledge map.

### response(answers)
Provide answers to the reasoning engine.

### answer(answer_or_answers, certainty=100)
Answer the latest question automatically.

### answer_first_form(...)
Answer yes/no questions.

### answer_second_form(...)
Answer value-based questions.

### undo()
Undo the last action.

---

## Example: Complete Workflow

```python
from rainbird_python_sdk import RainbirdAPIClient

client = RainbirdAPIClient(
    api_host="https://api.rainbird.ai",
    api_key="your-api-key",
    km_id="your-km-id"
)

client.start()

result = client.query(subject="Alice", relationship="has skill level")

while 'question' in result:
    result = client.answer("advanced", certainty=90)

print(result)
```

---

## Error Handling

```python
try:
    result = client.query(subject="test", relationship="age")
except Exception as e:
    print(e)
```

---

## Requirements

- Python >= 3.8
- requests >= 2.25.0

---

## License

MIT License

---

## Support

- GitHub Issues: TODO  
- Email: julio@rainbird.ai  
- Docs: https://rainbird.redoc.ly/#operation/explain  
