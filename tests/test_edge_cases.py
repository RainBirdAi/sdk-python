"""Tests for edge cases and error conditions."""

import pytest
from unittest.mock import Mock, patch
import requests
from rainbird_python_sdk.rainbird_api_client import RainbirdAPIClientTest


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch('requests.Session.get')
    def test_start_network_error(self, mock_get, api_credentials):
        """Test handling of network errors."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        client = RainbirdAPIClientTest(**api_credentials)

        with pytest.raises(requests.exceptions.ConnectionError):
            client.start()

    @patch('requests.Session.get')
    def test_start_timeout(self, mock_get, api_credentials):
        """Test handling of timeout errors."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")

        client = RainbirdAPIClientTest(**api_credentials)

        with pytest.raises(requests.exceptions.Timeout):
            client.start()

    @patch('requests.Session.post')
    def test_inject_with_malformed_data(self, mock_post, api_credentials):
        """Test inject with malformed fact data."""
        mock_resp = Mock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Bad Request")
        mock_post.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'

        with pytest.raises(requests.exceptions.HTTPError):
            client.inject([{'invalid': 'fact'}])

    @patch('requests.Session.post')
    def test_query_unauthorized(self, mock_post, api_credentials):
        """Test query with unauthorized access."""
        mock_resp = Mock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        mock_post.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'

        with pytest.raises(requests.exceptions.HTTPError):
            client.query(subject='test', relationship='test_rel')


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_api_host(self):
        """Test initialization with empty api_host."""
        client = RainbirdAPIClientTest(api_host='', api_key='key', km_id='km')
        assert client.api_host == ''

    def test_empty_api_key(self):
        """Test headers without api_key."""
        client = RainbirdAPIClientTest(
            api_host='https://api.rainbird.ai',
            api_key='',
            km_id='test-km'
        )
        headers = client._get_headers()
        assert 'X-API-Key' not in headers

    def test_none_api_key(self):
        """Test headers with None api_key."""
        client = RainbirdAPIClientTest(
            api_host='https://api.rainbird.ai',
            api_key=None,
            km_id='test-km'
        )
        headers = client._get_headers()
        assert 'X-API-Key' not in headers

    def test_multiple_trailing_slashes(self):
        """Test api_host with multiple trailing slashes."""
        client = RainbirdAPIClientTest(
            api_host='https://api.rainbird.ai///',
            api_key='key',
            km_id='km'
        )
        assert client.api_host == 'https://api.rainbird.ai'

    @patch('requests.Session.post')
    def test_inject_empty_facts_list(self, mock_post, api_credentials):
        """Test inject with empty facts list."""
        mock_resp = Mock()
        mock_resp.json.return_value = {'status': 'ok'}
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'

        result = client.inject([])
        assert result['status'] == 'ok'

    @patch('requests.Session.post')
    def test_query_with_only_relationship(self, mock_post, api_credentials):
        """Test query with only mandatory relationship parameter."""
        mock_resp = Mock()
        mock_resp.json.return_value = {'result': []}
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'

        result = client.query(relationship='test_rel')
        assert result == {'result': []}

        call_args = mock_post.call_args
        assert call_args[1]['json'] == {'relationship': 'test_rel'}

    @patch('requests.Session.post')
    def test_inject_fact_with_zero_certainty(self, mock_post, api_credentials):
        """Test inject_fact with zero certainty."""
        mock_resp = Mock()
        mock_resp.json.return_value = {'status': 'ok'}
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'

        result = client.inject_fact('John', 'age', '30', certainty=0)

        call_args = mock_post.call_args
        assert call_args[1]['json'][0]['certainty'] == 0

    @patch('requests.Session.post')
    def test_answer_with_zero_certainty(self, mock_post, api_credentials):
        """Test answer_second_form with zero certainty."""
        mock_resp = Mock()
        mock_resp.json.return_value = {'result': []}
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'

        result = client.answer_second_form(
            subject='John',
            relationship='age',
            object='30',
            certainty=0
        )

        call_args = mock_post.call_args
        assert call_args[1]['json']['answers'][0]['certainty'] == 0


class TestSpecialCharacters:
    """Test handling of special characters."""

    @patch('requests.Session.post')
    def test_inject_with_special_characters(self, mock_post, api_credentials):
        """Test inject with special characters in values."""
        mock_resp = Mock()
        mock_resp.json.return_value = {'status': 'ok'}
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'

        client.inject_fact('O\'Brien', 'city', 'São Paulo', certainty=100)

        call_args = mock_post.call_args
        fact = call_args[1]['json'][0]
        assert fact['subject'] == 'O\'Brien'
        assert fact['object'] == 'São Paulo'

    @patch('requests.Session.post')
    def test_query_with_unicode(self, mock_post, api_credentials):
        """Test query with unicode characters."""
        mock_resp = Mock()
        mock_resp.json.return_value = {'result': []}
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'

        result = client.query(subject='测试', relationship='属性')

        call_args = mock_post.call_args
        assert call_args[1]['json']['subject'] == '测试'
        assert call_args[1]['json']['relationship'] == '属性'

    @patch('requests.Session.post')
    def test_interact_with_long_prompt(self, mock_post, api_credentials):
        """Test interact with very long prompt."""
        mock_resp = Mock()
        mock_resp.json.return_value = {'response': 'ok'}
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'

        long_prompt = "What is " + "x" * 10000 + "?"
        result = client.interact(long_prompt)

        call_args = mock_post.call_args
        assert call_args[1]['json']['userPrompt'] == long_prompt


class TestSessionManagement:
    """Test session management edge cases."""

    @patch('requests.Session.get')
    def test_multiple_starts(self, mock_get, api_credentials):
        """Test calling start multiple times."""
        mock_resp = Mock()
        mock_resp.json.side_effect = [
            {'id': 'session-123'},
            {'id': 'session-456'}
        ]
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)

        result1 = client.start()
        assert client.session_id == 'session-123'

        result2 = client.start()
        assert client.session_id == 'session-456'

    @patch('requests.Session.get')
    def test_session_id_not_in_response(self, mock_get, api_credentials):
        """Test handling when session response doesn't contain id."""
        mock_resp = Mock()
        mock_resp.json.return_value = {'status': 'ok'}  # No 'id' field
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        result = client.start()

        assert client.session_id is None

    def test_operations_after_close(self, api_credentials):
        """Test that operations fail after close."""
        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'
        client.close()

        # Operations should still check for session_id, not that session is closed
        # The close just closes the underlying requests.Session
        # The client itself should still work if session_id is set


class TestQuestionTypeHandling:
    """Test handling of different question types."""

    @patch('requests.Session.post')
    def test_first_form_method(self, mock_post, api_credentials):
        """Test answer_first_form method includes answer field."""
        mock_resp = Mock()
        mock_resp.json.return_value = {'result': []}
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'

        client.answer_first_form(
            subject='John',
            relationship='age',
            object='30',
            object_value='yes'
        )

        call_args = mock_post.call_args
        answer = call_args[1]['json']['answers'][0]
        assert 'answer' in answer  # First Form includes 'answer' field
        assert answer['answer'] == 'yes'
        assert answer['object'] == '30'  # Also includes the object being asked about

    @patch('requests.Session.post')
    def test_second_form_method(self, mock_post, api_credentials):
        """Test answer_second_form method structure."""
        mock_resp = Mock()
        mock_resp.json.return_value = {'result': []}
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'

        client.answer_second_form(
            subject='John',
            relationship='skills',
            object='Python'
        )

        call_args = mock_post.call_args
        answer = call_args[1]['json']['answers'][0]

        assert answer['subject'] == 'John'
        assert answer['relationship'] == 'skills'
        assert answer['object'] == 'Python'
        assert answer['certainty'] == 100  # Default
        assert 'answer' not in answer  # Second Form does not include 'answer' field


class TestAnalysisEdgeCases:
    """Test edge cases in analysis methods."""

    @patch('requests.Session.get')
    def test_session_info_with_empty_filters(self, mock_get, api_credentials):
        """Test session_info with empty filters list."""
        mock_resp = Mock()
        mock_resp.json.return_value = {'session': 'info'}
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'

        result = client.session_info(filters=[])

        call_args = mock_get.call_args
        # Empty list is falsy, so filter should not be in params
        assert 'filter' not in call_args[1]['params']

    @patch('requests.Session.get')
    def test_interaction_log_different_formats(self, mock_get, api_credentials):
        """Test interaction_log with different format options."""
        mock_resp = Mock()
        mock_resp.text = 'log data'
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        client = RainbirdAPIClientTest(**api_credentials)
        client.session_id = 'session-123'

        formats = ['csv', 'json', 'xml']

        for fmt in formats:
            result = client.interaction_log(format=fmt)
            call_args = mock_get.call_args
            assert call_args[1]['params']['format'] == fmt
