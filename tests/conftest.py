"""Pytest configuration and fixtures for rainbird-python-sdk tests."""

import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_response():
    """Create a mock response object."""
    def _mock_response(json_data, status_code=200):
        mock_resp = Mock()
        mock_resp.json.return_value = json_data
        mock_resp.status_code = status_code
        mock_resp.raise_for_status = Mock()
        mock_resp.text = str(json_data)
        return mock_resp
    return _mock_response


@pytest.fixture
def api_credentials():
    """Provide test API credentials."""
    return {
        'api_host': 'https://api.rainbird.ai',
        'api_key': 'test-api-key-12345',
        'km_id': 'test-km-id-67890'
    }


@pytest.fixture
def sample_session_response():
    """Sample session start response."""
    return {
        'id': 'session-123',
        'kmId': 'test-km-id-67890',
        'status': 'active'
    }


@pytest.fixture
def sample_question_response():
    """Sample question response."""
    return {
        'question': {
            'type': 'Second Form',
            'prompt': 'What is the age?',
            'subject': 'John',
            'relationship': 'age',
            'concepts': ['20', '30', '40']
        }
    }


@pytest.fixture
def sample_first_form_question():
    """Sample First Form question response."""
    return {
        'question': {
            'type': 'First Form',
            'prompt': 'Is John 30 years old?',
            'subject': 'John',
            'relationship': 'age',
            'object': '30'
        }
    }


@pytest.fixture
def sample_result_response():
    """Sample result response without questions."""
    return {
        'result': [
            {
                'subject': 'John',
                'relationship': 'age',
                'object': '30',
                'certainty': 100
            }
        ]
    }
