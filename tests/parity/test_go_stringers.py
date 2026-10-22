"""Port of the Go SDK stringer tests: ``answer_test.go`` and ``question_test.go``.

Source cases
    - ``TestAnswerToString`` (answer_test.go:9): "Empty", "John speaks English 50%",
      "Dan speaks German 100%", "John favourite number 137 35%". Go's ``Answer`` (a decision
      returned by the engine) is Python's :class:`rainbird_ai.ResultItem`.
    - ``TestQuestionToString`` (question_test.go:9): "Empty", "Unknown object", "Unknown
      subject", "Unknown object as empty string". Go's ``Question`` is Python's
      :class:`rainbird_ai.Question`.

Each Go table entry builds the struct literally; here the same data goes through
``from_dict`` (the only constructor the SDK uses for API payloads) and ``str()`` is asserted.

Differences found (all asserted below, none of them fixed here)
    - Format: Go renders ``"Subject - relationship - object [ 50%]"`` (certainty right-aligned in
      three columns); Python renders ``"Subject relationship object (50%)"``.
    - Format: Go's ``Question.String()`` is ``"prompt (subject - relationship - object)"``;
      Python returns the prompt alone and only falls back to the ``"subject relationship
      object"`` triple when the prompt is empty.
    - Placeholders: Go prints ``"?"`` for an unknown question subject/object and ``""`` for a nil
      answer object; since 1.1.0 Python prints ``"?"`` for a missing (``None`` or ``""``)
      subject/object of a question, result item or session fact and joins only the non-empty
      parts (before, it interpolated the literal ``"None"``; the two tests asserting "no None in
      the text" were strict xfails then).
    - Certainty: Go's zero value renders ``[  0%]``; Python distinguishes a missing certainty
      (nothing appended) from an explicit ``0`` (``"(0%)"``).
    - Go quirk (not a Go test case): ``objectStr`` formats the *string-asserted* value, so a
      non-string ``Question.Object`` such as ``137`` renders as ``""`` in Go; Python keeps the
      number on ``Question.object``.
"""

from __future__ import annotations

from typing import Any

import pytest

from rainbird_ai import Question, ResultItem

# --------------------------------------------------------------------------- TestAnswerToString


# source: /tmp/rb-sdk-go/answer_test.go:16 TestAnswerToString/"Empty"
def test_answer_to_string_empty() -> None:
    # Go: Answer{} (zero value: Subject "", Relationship "", Object nil, Certainty 0)
    # renders " -  -  [  0%]".
    item = ResultItem.from_dict({})

    assert item.subject is None
    assert item.relationship == ""
    assert item.object is None
    assert item.certainty is None
    assert item.fact_id is None
    # DIFFERENCE(go): Go renders the zero value as " -  -  [  0%]" (empty strings, 0%).
    # Python does not crash on an empty payload, renders the unknown subject/object as "?" and
    # appends no certainty when the API sent none.
    assert str(item) == "? ?"

    # The closest JSON to Go's zero value: explicit empty strings, a null object and 0%.
    zero = ResultItem.from_dict({"subject": "", "relationship": "", "object": None, "certainty": 0})
    assert zero.certainty == 0
    # DIFFERENCE(go): an explicit 0 certainty is kept (Go: "[  0%]", Python: "(0%)"); the empty
    # subject and nil object are rendered "?" where Go prints "".
    assert str(zero) == "? ? (0%)"


# source: /tmp/rb-sdk-go/answer_test.go:21 TestAnswerToString/"John speaks English 50%"
def test_answer_to_string_john_speaks_english_50() -> None:
    item = ResultItem.from_dict(
        {"subject": "John", "relationship": "speaks", "object": "English", "certainty": 50}
    )

    assert item.subject == "John"
    assert item.relationship == "speaks"
    assert item.object == "English"
    assert item.certainty == 50
    # DIFFERENCE(go): Go expects "John - speaks - English [ 50%]" (dash separators, certainty
    # right-aligned to 3 columns in square brackets). Python uses spaces and "(50%)".
    assert str(item) == "John speaks English (50%)"


# source: /tmp/rb-sdk-go/answer_test.go:31 TestAnswerToString/"Dan speaks German 100%"
def test_answer_to_string_dan_speaks_german_100() -> None:
    item = ResultItem.from_dict(
        {"subject": "Dan", "relationship": "speaks", "object": "German", "certainty": 100}
    )

    assert item.certainty == 100
    # DIFFERENCE(go): Go expects "Dan - speaks - German [100%]"; Python: "(100%)".
    assert str(item) == "Dan speaks German (100%)"


# source: /tmp/rb-sdk-go/answer_test.go:41 TestAnswerToString/"John favourite number 137 35%"
def test_answer_to_string_john_favourite_number_137_35() -> None:
    # Go's Object is interface{}; the test uses an untyped int (137), not a string.
    item = ResultItem.from_dict(
        {"subject": "John", "relationship": "favourite number", "object": 137, "certainty": 35}
    )

    # Both SDKs keep the number as a number (Python's ResultItem.object is Any).
    assert item.object == 137
    assert isinstance(item.object, int)
    assert item.certainty == 35
    # DIFFERENCE(go): Go expects "John - favourite number - 137 [ 35%]"; Python renders the
    # numeric object the same way ("137") but with its own separators: "(35%)".
    assert str(item) == "John favourite number 137 (35%)"


# Addition (not a Go table entry): Go's String() never prints "None"/"nil" for a missing part
# (it prints "" for a nil object). Python prints "?" since 1.1.0 (strict xfail before).
def test_answer_to_string_missing_parts_do_not_render_none() -> None:
    assert "None" not in str(ResultItem.from_dict({}))
    assert "None" not in str(ResultItem.from_dict({"subject": "John", "relationship": "speaks"}))


# --------------------------------------------------------------------------- TestQuestionToString


# source: /tmp/rb-sdk-go/question_test.go:16 TestQuestionToString/"Empty"
def test_question_to_string_empty() -> None:
    # Go: Question{} renders " (? -  - ?)": empty prompt, "?" for the unknown subject and object.
    question = Question.from_dict({})

    assert question.prompt == ""
    assert question.subject is None
    assert question.relationship == ""
    assert question.object is None
    assert question.type == ""
    # DIFFERENCE(go): Go renders " (? -  - ?)". Python does not crash on an empty payload; with
    # no prompt it falls back to "subject relationship object", renders the unknown parts as
    # "?" like Go and drops the empty relationship instead of leaving double spaces.
    assert str(question) == "? ?"

    # The closest JSON to Go's zero value (all strings empty) renders the same way.
    zero = Question.from_dict({"prompt": "", "subject": "", "relationship": "", "object": ""})
    assert str(zero) == "? ?"


# source: /tmp/rb-sdk-go/question_test.go:21 TestQuestionToString/"Unknown object"
def test_question_to_string_unknown_object() -> None:
    question = Question.from_dict(
        {"prompt": "Which language does John speak?", "subject": "John", "relationship": "speaks"}
    )

    assert question.prompt == "Which language does John speak?"
    assert question.subject == "John"
    assert question.relationship == "speaks"
    # Python's representation of Go's "?" (nil Object): None on the model.
    assert question.object is None
    # DIFFERENCE(go): Go expects "Which language does John speak? (John - speaks - ?)", i.e. the
    # prompt followed by the triple with "?" for the unknown object. Python's str() is the prompt
    # alone whenever the engine supplied one.
    assert str(question) == "Which language does John speak?"

    # Without a prompt Python falls back to the triple; the unknown object is rendered "?".
    no_prompt = Question.from_dict({"subject": "John", "relationship": "speaks"})
    # DIFFERENCE(go): Go would render " (John - speaks - ?)"; Python: "John speaks ?".
    assert str(no_prompt) == "John speaks ?"


# source: /tmp/rb-sdk-go/question_test.go:30 TestQuestionToString/"Unknown subject"
def test_question_to_string_unknown_subject() -> None:
    question = Question.from_dict(
        {"prompt": "Who lives in England?", "relationship": "lives in", "object": "England"}
    )

    assert question.prompt == "Who lives in England?"
    # Python's representation of Go's "?" (empty Subject): None on the model.
    assert question.subject is None
    assert question.relationship == "lives in"
    assert question.object == "England"
    # DIFFERENCE(go): Go expects "Who lives in England? (? - lives in - England)"; Python
    # returns the prompt alone.
    assert str(question) == "Who lives in England?"

    # Without a prompt Python falls back to the triple; the unknown subject is rendered "?".
    no_prompt = Question.from_dict({"relationship": "lives in", "object": "England"})
    # DIFFERENCE(go): Go would render " (? - lives in - England)"; Python: "? lives in England".
    assert str(no_prompt) == "? lives in England"


# source: /tmp/rb-sdk-go/question_test.go:39 TestQuestionToString/"Unknown object as empty string"
def test_question_to_string_unknown_object_as_empty_string() -> None:
    question = Question.from_dict(
        {"prompt": "Who lives in England?", "relationship": "lives in", "object": ""}
    )

    assert question.subject is None
    assert question.relationship == "lives in"
    # DIFFERENCE(go): Go's objectStr() normalises an empty-string Object to "?" (same as nil).
    # Python keeps the value the API sent: "" stays "" and is distinct from a missing object.
    assert question.object == ""
    assert question.object is not None
    # DIFFERENCE(go): Go expects "Who lives in England? (? - lives in - ?)"; Python returns the
    # prompt alone.
    assert str(question) == "Who lives in England?"

    # Without a prompt the empty object and the missing subject are both rendered "?", as in Go.
    no_prompt = Question.from_dict({"relationship": "lives in", "object": ""})
    # DIFFERENCE(go): Go would render " (? - lives in - ?)"; Python: "? lives in ?".
    assert str(no_prompt) == "? lives in ?"


# Addition (not a Go table entry): Go prints "?" for an unknown subject/object, so its String()
# never contains "None"/"nil". Python does the same since 1.1.0 (strict xfail before).
@pytest.mark.parametrize(
    "data",
    [
        {},
        {"subject": "John", "relationship": "speaks"},
        {"relationship": "lives in", "object": "England"},
        {"relationship": "lives in", "object": ""},
    ],
    ids=["empty", "unknown-object", "unknown-subject", "unknown-object-as-empty-string"],
)
def test_question_to_string_missing_parts_do_not_render_none(data: dict[str, Any]) -> None:
    assert "None" not in str(Question.from_dict(data))


# Addition (not a Go table entry): a numeric question object. Go's objectStr() does
# `strObj, ok := q.Object.(string)` and then formats strObj, so a non-string object renders as
# "" (e.g. " (? - favourite number - )"). Python keeps the number on the model.
def test_question_to_string_numeric_object_is_kept() -> None:
    question = Question.from_dict(
        {
            "prompt": "Is 137 John's favourite number?",
            "relationship": "favourite number",
            "object": 137,
        }
    )

    # DIFFERENCE(go): Go loses a non-string Object in String() (renders ""); Python keeps 137.
    assert question.object == 137
    assert str(question) == "Is 137 John's favourite number?"
    assert str(
        Question.from_dict({"subject": "John", "relationship": "favourite number", "object": 137})
    ) == ("John favourite number 137")
