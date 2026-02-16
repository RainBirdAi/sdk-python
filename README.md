# Rainbird Python SDK

A Python SDK for interacting with the Rainbird API. This library provides a simple interface for working with Rainbird knowledge maps, allowing you to start sessions, inject facts, query knowledge, and interact with the reasoning engine.

## Installation

```bash
pip install rainbird-python-sdk
```

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
    # Answer the question
    response = client.answer(object_value="30", certainty=100)

# Close the session
client.close()
```

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
    # Session automatically closed when exiting the context
```

## API Reference

### Initialisation

```python
client = RainbirdAPIClient(api_host, api_key, km_id)
```

- `api_host`: The Rainbird API host URL
- `api_key`: Your API key for authentication
- `km_id`: The knowledge map ID to interact with

### Core Methods

#### `start(use_draft=None, version=None)`
Start a new Rainbird session.

- `use_draft`: Optional boolean to use draft version of knowledge map
- `version`: Optional integer to specify knowledge map version
- Returns: Session information dictionary

#### `inject(facts)`
Inject multiple facts into the session.

- `facts`: List of fact dictionaries with subject, relationship, object, and certainty
- Returns: Response dictionary

#### `inject_fact(subject, relationship, object_value, certainty=100)`
Inject a single fact into the session.

- `subject`: The subject of the fact
- `relationship`: The relationship/predicate
- `object_value`: The object value
- `certainty`: Certainty level (0-100), defaults to 100
- Returns: Response dictionary

#### `query(subject=None, relationship=None, object=None)`
Query the knowledge map.

- `subject`: Optional subject to query
- `relationship`: Relationship to query
- `object`: Optional object to query
- Returns: Query result, may include a question

#### `response(answers)`
Provide answers to questions from the reasoning engine.

- `answers`: List of complete answer triplets with subject, relationship, object, and certainty
- Returns: Response dictionary, may include follow-up questions

#### `answer_first_form(subject, relationship, object, object_value, certainty=100)`
Answer a First Form question (yes/no questions).

- `subject`: The subject from the question
- `relationship`: The relationship from the question
- `object`: The object being asked about
- `object_value`: The answer ("yes" or "no")
- `certainty`: Certainty level (0-100), defaults to 100
- Returns: Response dictionary

#### `answer_second_form(subject, relationship, object, certainty=100)`
Answer a Second Form question (provide a value).

- `subject`: The subject from the question
- `relationship`: The relationship from the question
- `object`: The answer value
- `certainty`: Certainty level (0-100), defaults to 100
- Returns: Response dictionary

#### `undo()`
Undo the last action in the session.

- Returns: Response dictionary

### Natural Language Methods

#### `interact(user_prompt)`
Interact with the knowledge map using natural language.

- `user_prompt`: Natural language prompt
- Returns: Response dictionary

#### `explain(fact_id, language='en')`
Get a natural language explanation for a fact.

- `fact_id`: The ID of the fact to explain
- `language`: Language code (default: 'en')
- Returns: Explanation dictionary

### Analysis Methods

#### `evidence(fact_id, evidence_key=None)`
Get evidence for a specific fact.

- `fact_id`: The ID of the fact
- `evidence_key`: Optional evidence key for authentication
- Returns: Evidence dictionary

#### `interaction_log(interaction_key=None, format='csv')`
Get the interaction log for the session.

- `interaction_key`: Optional interaction key
- `format`: Format for the log (default: 'csv')
- Returns: Interaction log as data

#### `session_info(filters=None, relationships=None)`
Get information about the current session.

- `filters`: Optional list of filters
- `relationships`: Optional list of relationships to include
- Returns: Session information dictionary

### Platform Methods

#### `version()`
Get the Rainbird API version information.

- Returns: Version information dictionary


## Example: Complete Workflow

```python
from rainbird_python_sdk import RainbirdAPIClient

# Initialise
client = RainbirdAPIClient(
    api_host="https://api.rainbird.ai",
    api_key="your-api-key",
    km_id="your-km-id"
)

try:
    # Start session
    client.start(use_draft=False)

    # Inject some facts
    client.inject_fact("Alice", "know", "Python", certainty=100)
    client.inject_fact("Alice", "has experience years", "5", certainty=95)

    # Query (relationship is mandatory)
    result = client.query(subject="Alice", relationship="has skill level")

    # Handle questions interactively
    while 'question' in result:
        question = result['question']
        print(f"Question: {question['prompt']}")

        question_type = question.get('type', '')

        # Handle based on question type
        if 'First Form' in question_type:
            # Yes/no question
            result = client.answer_first_form(
                subject=question['subject'],
                relationship=question['relationship'],
                object=question['object'],
                object_value="yes",
                certainty=90
            )
        else:
            # Second Form - provide value
            result = client.answer_second_form(
                subject=question['subject'],
                relationship=question['relationship'],
                object="advanced",
                certainty=90
            )

    # Get final results
    if 'result' in result:
        print(f"Results: {result['result']}")

    # Get session information
    session_info = client.session_info()
    print(f"Session info: {session_info}")

```

## Example: Using `response()` Directly

For advanced use cases, you can use the low-level `response()` method with complete answer triplets:

```python
# Start session
client.start()

# Query
result = client.query(relationship="employee_eligible")

# Provide multiple answers at once using complete triplets
if 'question' in result:
    question = result['question']
    client.response([
        {
            "subject": question['subject'],
            "relationship": question['relationship'],
            "object": "Spanish",
            "certainty": 100
        },
        {
            "subject": question['subject'],
            "relationship": question['relationship'],
            "object": "French",
            "certainty": 95
        }
    ])
```

## Error Handling

The SDK raises standard Python exceptions:

```python
try:
    # Relationship is mandatory
    result = client.query(subject="test", relationship="age")
except ValueError as e:
    print(f"Value error: {e}")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Requirements

- Python >= 3.8
- requests >= 2.25.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
- GitHub Issues: TODO - Add GitHub link
- Email: julio@rainbird.ai
- Rainbird AI API Docs: https://rainbird.redoc.ly/#operation/explain

## About Rainbird

Rainbird is an AI platform that enables you to build knowledge maps for automated reasoning and decision-making. Visit [rainbird.ai](https://rainbird.ai) for more information.
