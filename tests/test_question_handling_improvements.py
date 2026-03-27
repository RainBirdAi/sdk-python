"""Tests for improved question handling behaviour."""

import pytest
from unittest.mock import patch

from rainbird_python_sdk.rainbird_api_client import RainbirdAPIClient


class TestImprovedQuestionHandling:
    """Test improved response/answer helper methods."""

    @patch("requests.Session.post")
    def test_response_accepts_single_answer_dict(self, mock_post, api_credentials, mock_response):
        """response() should accept a single answer dict and normalize it to a list."""
        response_data = {"result": []}
        mock_post.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"

        answer = {
            "subject": "John",
            "relationship": "age",
            "object": "30",
            "certainty": 100,
        }

        result = client.response(answer)

        assert result == response_data
        call_args = mock_post.call_args
        assert call_args[1]["json"] == {"answers": [answer]}

    @patch("requests.Session.post")
    def test_response_accepts_multiple_answers(self, mock_post, api_credentials, mock_response):
        """response() should still accept a list of answers unchanged."""
        response_data = {"result": []}
        mock_post.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"

        answers = [
            {
                "subject": "John",
                "relationship": "skill",
                "object": "Python",
                "certainty": 100,
            },
            {
                "subject": "John",
                "relationship": "skill",
                "object": "SQL",
                "certainty": 95,
            },
        ]

        result = client.response(answers)

        assert result == response_data
        call_args = mock_post.call_args
        assert call_args[1]["json"] == {"answers": answers}

    def test_response_rejects_empty_answers(self, api_credentials):
        """response() should reject an empty list of answers."""
        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"

        with pytest.raises(ValueError, match="answers must not be empty"):
            client.response([])

    def test_response_rejects_invalid_type(self, api_credentials):
        """response() should reject invalid answer input types."""
        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"

        with pytest.raises(ValueError, match="answers must be a dictionary or a list of dictionaries"):
            client.response("not a valid answer")

    def test_response_rejects_missing_required_fields(self, api_credentials):
        """response() should reject malformed answer dicts."""
        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"

        with pytest.raises(ValueError, match="Answer is missing required field"):
            client.response({"subject": "John"})

    @patch("requests.Session.post")
    def test_answer_question_first_form(self, mock_post, api_credentials, mock_response):
        """answer_question() should build the correct First Form payload."""
        response_data = {"result": []}
        mock_post.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"

        question = {
            "type": "First Form",
            "subject": "John",
            "relationship": "age",
            "object": "30",
        }

        result = client.answer_question(question, "yes")

        assert result == response_data
        call_args = mock_post.call_args
        assert call_args[1]["json"] == {
            "answers": [
                {
                    "subject": "John",
                    "relationship": "age",
                    "object": "30",
                    "answer": "yes",
                    "certainty": 100,
                }
            ]
        }

    @patch("requests.Session.post")
    def test_answer_question_second_form_single_answer(self, mock_post, api_credentials, mock_response):
        """answer_question() should build the correct Second Form payload for one answer."""
        response_data = {"result": []}
        mock_post.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"

        question = {
            "type": "Second Form",
            "subject": "John",
            "relationship": "age",
        }

        result = client.answer_question(question, "30")

        assert result == response_data
        call_args = mock_post.call_args
        assert call_args[1]["json"] == {
            "answers": [
                {
                    "subject": "John",
                    "relationship": "age",
                    "object": "30",
                    "certainty": 100,
                }
            ]
        }

    @patch("requests.Session.post")
    def test_answer_question_second_form_multiple_answers(self, mock_post, api_credentials, mock_response):
        """answer_question() should build the correct Second Form payload for many answers."""
        response_data = {"result": []}
        mock_post.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"

        question = {
            "type": "Second Form",
            "subject": "John",
            "relationship": "skill",
        }

        result = client.answer_question(question, ["Python", "SQL"])

        assert result == response_data
        call_args = mock_post.call_args
        assert call_args[1]["json"] == {
            "answers": [
                {
                    "subject": "John",
                    "relationship": "skill",
                    "object": "Python",
                    "certainty": 100,
                },
                {
                    "subject": "John",
                    "relationship": "skill",
                    "object": "SQL",
                    "certainty": 100,
                },
            ]
        }

    def test_answer_question_rejects_unknown_question_type(self, api_credentials):
        """answer_question() should reject unsupported question types."""
        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"

        question = {
            "type": "Unknown Form",
            "subject": "John",
            "relationship": "age",
        }

        with pytest.raises(ValueError, match="Unsupported question type"):
            client.answer_question(question, "30")

    def test_answer_question_rejects_missing_type(self, api_credentials):
        """answer_question() should reject questions with no type."""
        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"

        question = {
            "subject": "John",
            "relationship": "age",
        }

        with pytest.raises(ValueError, match="question must include a 'type' field"):
            client.answer_question(question, "30")

    @patch("requests.Session.post")
    def test_answer_uses_last_question(self, mock_post, api_credentials, mock_response):
        """answer() should use the stored latest question automatically."""
        response_data = {"result": []}
        mock_post.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"
        client.last_question = {
            "type": "Second Form",
            "subject": "John",
            "relationship": "age",
        }

        result = client.answer("30")

        assert result == response_data
        call_args = mock_post.call_args
        assert call_args[1]["json"] == {
            "answers": [
                {
                    "subject": "John",
                    "relationship": "age",
                    "object": "30",
                    "certainty": 100,
                }
            ]
        }

    def test_answer_rejects_when_no_last_question(self, api_credentials):
        """answer() should fail if there is no stored question."""
        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"

        with pytest.raises(ValueError, match="No outstanding question"):
            client.answer("30")

    @patch("requests.Session.post")
    def test_query_stores_last_question(self, mock_post, api_credentials, mock_response):
        """query() should store a returned question on the client."""
        response_data = {
            "question": {
                "type": "Second Form",
                "subject": "John",
                "relationship": "age",
            }
        }
        mock_post.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"

        result = client.query(subject="John", relationship="age")

        assert result == response_data
        assert client.last_question == response_data["question"]

    @patch("requests.Session.post")
    def test_response_stores_follow_up_question(self, mock_post, api_credentials, mock_response):
        """response() should store a follow-up question if one is returned."""
        response_data = {
            "question": {
                "type": "First Form",
                "subject": "John",
                "relationship": "eligible",
                "object": "yes",
            }
        }
        mock_post.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = "session-123"

        answer = {
            "subject": "John",
            "relationship": "age",
            "object": "30",
            "certainty": 100,
        }

        result = client.response(answer)

        assert result == response_data
        assert client.last_question == response_data["question"]