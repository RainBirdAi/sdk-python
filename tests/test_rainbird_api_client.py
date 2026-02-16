"""Tests for RainbirdAPIClient class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from rainbird_python_sdk.rainbird_api_client import RainbirdAPIClient


class TestInitialization:
    """Test client initialization."""

    def test_init_basic(self, api_credentials):
        """Test basic initialization."""
        client = RainbirdAPIClient(**api_credentials)

        assert client.api_host == 'https://api.rainbird.ai'
        assert client.api_key == 'test-api-key-12345'
        assert client.km_id == 'test-km-id-67890'
        assert client.session_id is None
        assert isinstance(client.session, requests.Session)

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from api_host."""
        client = RainbirdAPIClient(
            api_host='https://api.rainbird.ai/',
            api_key='test-key',
            km_id='test-km'
        )
        assert client.api_host == 'https://api.rainbird.ai'


class TestHeaderGeneration:
    """Test header generation method."""

    def test_get_headers_basic(self, api_credentials):
        """Test basic headers with only API key."""
        client = RainbirdAPIClient(**api_credentials)
        headers = client._get_headers()

        assert headers == {'X-API-Key': 'test-api-key-12345'}

    def test_get_headers_with_content_type(self, api_credentials):
        """Test headers with content type."""
        client = RainbirdAPIClient(**api_credentials)
        headers = client._get_headers(content_type='application/json')

        assert headers['X-API-Key'] == 'test-api-key-12345'
        assert headers['Content-Type'] == 'application/json'

    def test_get_headers_with_version(self, api_credentials):
        """Test headers with version."""
        client = RainbirdAPIClient(**api_credentials)
        headers = client._get_headers(version='v1')

        assert headers['Version'] == 'v1'

    def test_get_headers_with_evidence_key(self, api_credentials):
        """Test headers with evidence key."""
        client = RainbirdAPIClient(**api_credentials)
        headers = client._get_headers(evidence_key='evidence-123')

        assert headers['x-evidence-key'] == 'evidence-123'

    def test_get_headers_with_interaction_key(self, api_credentials):
        """Test headers with interaction key."""
        client = RainbirdAPIClient(**api_credentials)
        headers = client._get_headers(interaction_key='interaction-456')

        assert headers['x-interaction-key'] == 'interaction-456'

    def test_get_headers_all_parameters(self, api_credentials):
        """Test headers with all parameters."""
        client = RainbirdAPIClient(**api_credentials)
        headers = client._get_headers(
            content_type='application/json',
            version='v1',
            evidence_key='evidence-123',
            interaction_key='interaction-456'
        )

        assert len(headers) == 5
        assert headers['X-API-Key'] == 'test-api-key-12345'
        assert headers['Content-Type'] == 'application/json'
        assert headers['Version'] == 'v1'
        assert headers['x-evidence-key'] == 'evidence-123'
        assert headers['x-interaction-key'] == 'interaction-456'


class TestStartSession:
    """Test session start functionality."""

    @patch('requests.Session.get')
    def test_start_basic(self, mock_get, api_credentials, sample_session_response, mock_response):
        """Test basic session start."""
        mock_get.return_value = mock_response(sample_session_response)

        client = RainbirdAPIClient(**api_credentials)
        result = client.start()

        assert result == sample_session_response
        assert client.session_id == 'session-123'
        mock_get.assert_called_once()

        # Verify URL
        call_args = mock_get.call_args
        assert 'https://api.rainbird.ai/start/test-km-id-67890' in str(call_args)

    @patch('requests.Session.get')
    def test_start_with_use_draft(self, mock_get, api_credentials, sample_session_response, mock_response):
        """Test session start with use_draft parameter."""
        mock_get.return_value = mock_response(sample_session_response)

        client = RainbirdAPIClient(**api_credentials)
        client.start(use_draft=True)

        call_args = mock_get.call_args
        assert call_args[1]['params']['useDraft'] == 'true'

    @patch('requests.Session.get')
    def test_start_with_version(self, mock_get, api_credentials, sample_session_response, mock_response):
        """Test session start with version parameter."""
        mock_get.return_value = mock_response(sample_session_response)

        client = RainbirdAPIClient(**api_credentials)
        client.start(version=5)

        call_args = mock_get.call_args
        assert call_args[1]['params']['version'] == '5'

    @patch('requests.Session.get')
    def test_start_http_error(self, mock_get, api_credentials):
        """Test session start with HTTP error."""
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Error")

        client = RainbirdAPIClient(**api_credentials)

        with pytest.raises(requests.exceptions.HTTPError):
            client.start()


class TestInjectFacts:
    """Test fact injection functionality."""

    @patch('requests.Session.post')
    def test_inject_single_fact(self, mock_post, api_credentials, mock_response):
        """Test injecting a single fact."""
        response_data = {'status': 'ok'}
        mock_post.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        facts = [{'subject': 'John', 'relationship': 'age', 'object': '30', 'certainty': 100}]
        result = client.inject(facts)

        assert result == response_data
        mock_post.assert_called_once()

        call_args = mock_post.call_args
        assert 'session-123/inject' in call_args[0][0]
        assert call_args[1]['json'] == facts

    @patch('requests.Session.post')
    def test_inject_fact_helper(self, mock_post, api_credentials, mock_response):
        """Test inject_fact helper method."""
        response_data = {'status': 'ok'}
        mock_post.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.inject_fact('John', 'age', '30', certainty=100)

        assert result == response_data
        call_args = mock_post.call_args
        expected_fact = [{'subject': 'John', 'relationship': 'age', 'object': '30', 'certainty': 100}]
        assert call_args[1]['json'] == expected_fact

    def test_inject_without_session(self, api_credentials):
        """Test inject raises error without session."""
        client = RainbirdAPIClient(**api_credentials)

        with pytest.raises(ValueError, match="Session not started"):
            client.inject([])


class TestQuery:
    """Test query functionality."""

    @patch('requests.Session.post')
    def test_query_basic(self, mock_post, api_credentials, sample_result_response, mock_response):
        """Test basic query."""
        mock_post.return_value = mock_response(sample_result_response)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.query(subject='John', relationship='age')

        assert result == sample_result_response
        call_args = mock_post.call_args
        assert 'session-123/query' in call_args[0][0]
        assert call_args[1]['json']['subject'] == 'John'
        assert call_args[1]['json']['relationship'] == 'age'

    @patch('requests.Session.post')
    def test_query_without_relationship_fails(self, mock_post, api_credentials, mock_response):
        """Test query requires relationship parameter."""
        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        with pytest.raises(ValueError, match="Relationship is mandatory"):
            client.query(subject='John')

    @patch('requests.Session.post')
    def test_query_with_object(self, mock_post, api_credentials, sample_result_response, mock_response):
        """Test query with object parameter."""
        mock_post.return_value = mock_response(sample_result_response)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        client.query(subject='John', relationship='age', object='30')

        call_args = mock_post.call_args
        assert call_args[1]['json']['object'] == '30'

    def test_query_without_session(self, api_credentials):
        """Test query raises error without session."""
        client = RainbirdAPIClient(**api_credentials)

        with pytest.raises(ValueError, match="Session not started"):
            client.query(subject='John', relationship='age')


class TestResponse:
    """Test response functionality."""

    @patch('requests.Session.post')
    def test_response_basic(self, mock_post, api_credentials, sample_result_response, mock_response):
        """Test basic response."""
        mock_post.return_value = mock_response(sample_result_response)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        answers = [{'subject': 'John', 'relationship': 'age', 'object': '30', 'certainty': 100}]
        result = client.response(answers)

        assert result == sample_result_response
        call_args = mock_post.call_args
        assert 'session-123/response' in call_args[0][0]
        assert call_args[1]['json']['answers'] == answers

    @patch('requests.Session.post')
    def test_response_returns_result(self, mock_post, api_credentials, sample_question_response, mock_response):
        """Test response returns API result."""
        mock_post.return_value = mock_response(sample_question_response)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.response([])

        assert result == sample_question_response

    def test_response_without_session(self, api_credentials):
        """Test response raises error without session."""
        client = RainbirdAPIClient(**api_credentials)

        with pytest.raises(ValueError, match="Session not started"):
            client.response([])


class TestAnswerMethods:
    """Test answer form methods."""

    @patch('requests.Session.post')
    def test_answer_first_form(self, mock_post, api_credentials, sample_result_response, mock_response):
        """Test answer_first_form method."""
        mock_post.return_value = mock_response(sample_result_response)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.answer_first_form(
            subject='John',
            relationship='age',
            object='30',
            object_value='yes',
            certainty=100
        )

        assert result == sample_result_response
        call_args = mock_post.call_args
        answers = call_args[1]['json']['answers']

        assert len(answers) == 1
        assert answers[0]['subject'] == 'John'
        assert answers[0]['relationship'] == 'age'
        assert answers[0]['object'] == '30'
        assert answers[0]['answer'] == 'yes'
        assert answers[0]['certainty'] == 100

    @patch('requests.Session.post')
    def test_answer_second_form(self, mock_post, api_credentials, sample_result_response, mock_response):
        """Test answer_second_form method."""
        mock_post.return_value = mock_response(sample_result_response)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.answer_second_form(
            subject='John',
            relationship='age',
            object='30',
            certainty=95
        )

        assert result == sample_result_response
        call_args = mock_post.call_args
        answers = call_args[1]['json']['answers']

        assert len(answers) == 1
        assert answers[0]['subject'] == 'John'
        assert answers[0]['relationship'] == 'age'
        assert answers[0]['object'] == '30'
        assert answers[0]['certainty'] == 95

    @patch('requests.Session.post')
    def test_answer_first_form_default_certainty(self, mock_post, api_credentials, sample_result_response, mock_response):
        """Test answer_first_form with default certainty."""
        mock_post.return_value = mock_response(sample_result_response)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.answer_first_form(
            subject='John',
            relationship='age',
            object='30',
            object_value='yes'
        )

        call_args = mock_post.call_args
        answers = call_args[1]['json']['answers']
        assert answers[0]['certainty'] == 100  # Default

    @patch('requests.Session.post')
    def test_answer_second_form_default_certainty(self, mock_post, api_credentials, sample_result_response, mock_response):
        """Test answer_second_form with default certainty."""
        mock_post.return_value = mock_response(sample_result_response)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.answer_second_form(
            subject='John',
            relationship='age',
            object='30'
        )

        call_args = mock_post.call_args
        answers = call_args[1]['json']['answers']
        assert answers[0]['certainty'] == 100  # Default


class TestUndo:
    """Test undo functionality."""

    @patch('requests.Session.post')
    def test_undo(self, mock_post, api_credentials, mock_response):
        """Test undo method."""
        response_data = {'status': 'ok'}
        mock_post.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.undo()

        assert result == response_data
        call_args = mock_post.call_args
        assert 'session-123/undo' in call_args[0][0]
        assert call_args[1]['json'] == {}

    def test_undo_without_session(self, api_credentials):
        """Test undo raises error without session."""
        client = RainbirdAPIClient(**api_credentials)

        with pytest.raises(ValueError, match="Session not started"):
            client.undo()


class TestNaturalLanguage:
    """Test natural language methods."""

    @patch('requests.Session.post')
    def test_interact(self, mock_post, api_credentials, mock_response):
        """Test interact method."""
        response_data = {'response': 'John is 30 years old'}
        mock_post.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.interact("How old is John?")

        assert result == response_data
        call_args = mock_post.call_args
        assert 'nl/interact' in call_args[0][0]
        assert call_args[1]['json']['sessionID'] == 'session-123'
        assert call_args[1]['json']['userPrompt'] == "How old is John?"
        assert call_args[1]['headers']['Version'] == 'v1'

    @patch('requests.Session.post')
    def test_explain(self, mock_post, api_credentials, mock_response):
        """Test explain method."""
        response_data = {'explanation': 'John is 30 because...'}
        mock_post.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.explain('fact-123', language='en')

        assert result == response_data
        call_args = mock_post.call_args
        assert 'nl/explain' in call_args[0][0]
        assert call_args[1]['json']['factID'] == 'fact-123'
        assert call_args[1]['json']['language'] == 'en'
        assert call_args[1]['json']['sessionID'] == 'session-123'

    def test_interact_without_session(self, api_credentials):
        """Test interact raises error without session."""
        client = RainbirdAPIClient(**api_credentials)

        with pytest.raises(ValueError, match="Session not started"):
            client.interact("test prompt")


class TestAnalysisMethods:
    """Test analysis methods."""

    @patch('requests.Session.get')
    def test_evidence(self, mock_get, api_credentials, mock_response):
        """Test evidence method."""
        response_data = {'evidence': 'data'}
        mock_get.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.evidence('fact-456')

        assert result == response_data
        call_args = mock_get.call_args
        assert 'analysis/evidence/fact-456/session-123' in call_args[0][0]

    @patch('requests.Session.get')
    def test_evidence_with_key(self, mock_get, api_credentials, mock_response):
        """Test evidence method with evidence key."""
        response_data = {'evidence': 'data'}
        mock_get.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.evidence('fact-456', evidence_key='key-789')

        call_args = mock_get.call_args
        assert call_args[1]['headers']['x-evidence-key'] == 'key-789'

    @patch('requests.Session.get')
    def test_interaction_log(self, mock_get, api_credentials):
        """Test interaction_log method."""
        log_data = {'timestamp': '2024-01-01', 'action': 'query', 'result': 'success'}
        mock_resp = Mock()
        mock_resp.json.return_value = log_data
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.interaction_log(format='csv')

        assert result == log_data
        call_args = mock_get.call_args
        assert 'analysis/interactions/session-123' in call_args[0][0]
        assert call_args[1]['params']['format'] == 'csv'

    @patch('requests.Session.get')
    def test_session_info(self, mock_get, api_credentials, mock_response):
        """Test session_info method."""
        response_data = {'session': 'info'}
        mock_get.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        client.session_id = 'session-123'

        result = client.session_info(filters=['filter1'], relationships=['rel1', 'rel2'])

        assert result == response_data
        call_args = mock_get.call_args
        assert 'analysis/session/session-123' in call_args[0][0]
        assert call_args[1]['params']['filter'] == 'filter1'
        assert call_args[1]['params']['relationships'] == 'rel1,rel2'

    def test_evidence_without_session(self, api_credentials):
        """Test evidence raises error without session."""
        client = RainbirdAPIClient(**api_credentials)

        with pytest.raises(ValueError, match="Session not started"):
            client.evidence('fact-123')

    def test_interaction_log_without_session(self, api_credentials):
        """Test interaction_log raises error without session."""
        client = RainbirdAPIClient(**api_credentials)

        with pytest.raises(ValueError, match="Session not started"):
            client.interaction_log()

    def test_session_info_without_session(self, api_credentials):
        """Test session_info raises error without session."""
        client = RainbirdAPIClient(**api_credentials)

        with pytest.raises(ValueError, match="Session not started"):
            client.session_info()

    def test_explain_without_session(self, api_credentials):
        """Test explain raises error without session."""
        client = RainbirdAPIClient(**api_credentials)

        with pytest.raises(ValueError, match="Session not started"):
            client.explain('fact-123')


class TestUtilityMethods:
    """Test utility methods."""

    @patch('requests.Session.get')
    def test_version(self, mock_get, api_credentials, mock_response):
        """Test version method."""
        response_data = {'version': '1.0.0', 'build': '12345'}
        mock_get.return_value = mock_response(response_data)

        client = RainbirdAPIClient(**api_credentials)
        result = client.version()

        assert result == response_data
        call_args = mock_get.call_args
        assert 'https://api.rainbird.ai/version' == call_args[0][0]
        assert call_args[1]['headers']['Accept'] == '*/*'

    def test_close(self, api_credentials):
        """Test close method."""
        client = RainbirdAPIClient(**api_credentials)
        mock_session = Mock()
        client.session = mock_session

        client.close()

        mock_session.close.assert_called_once()


class TestContextManager:
    """Test context manager functionality."""

    def test_context_manager_enter(self, api_credentials):
        """Test context manager __enter__."""
        client = RainbirdAPIClient(**api_credentials)

        with client as ctx_client:
            assert ctx_client is client

    def test_context_manager_exit(self, api_credentials):
        """Test context manager __exit__ closes session."""
        client = RainbirdAPIClient(**api_credentials)
        mock_session = Mock()
        client.session = mock_session

        with client:
            pass

        mock_session.close.assert_called_once()

    def test_context_manager_with_exception(self, api_credentials):
        """Test context manager closes even with exception."""
        client = RainbirdAPIClient(**api_credentials)
        mock_session = Mock()
        client.session = mock_session

        try:
            with client:
                raise ValueError("Test error")
        except ValueError:
            pass

        mock_session.close.assert_called_once()


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @patch('requests.Session.post')
    @patch('requests.Session.get')
    def test_complete_workflow(self, mock_get, mock_post, api_credentials, mock_response):
        """Test a complete workflow: start, inject, query, answer."""
        # Setup mocks
        mock_get.return_value = mock_response({'id': 'session-123'})

        inject_response = mock_response({'status': 'ok'})
        query_response = mock_response({
            'question': {
                'type': 'Second Form',
                'subject': 'John',
                'relationship': 'age',
                'prompt': 'What is age?'
            }
        })
        answer_response = mock_response({'result': [{'subject': 'John', 'relationship': 'age', 'object': '30'}]})

        mock_post.side_effect = [inject_response, query_response, answer_response]

        # Execute workflow
        client = RainbirdAPIClient(**api_credentials)

        # Start session
        session = client.start()
        assert session['id'] == 'session-123'

        # Inject fact
        inject_result = client.inject_fact('John', 'name', 'John Doe')
        assert inject_result['status'] == 'ok'

        # Query
        query_result = client.query(subject='John', relationship='age')
        assert 'question' in query_result

        # Answer using response
        answer_result = client.response([
            {'subject': 'John', 'relationship': 'age', 'object': '30', 'certainty': 100}
        ])
        assert 'result' in answer_result

        client.close()

    @patch('requests.Session.post')
    @patch('requests.Session.get')
    def test_multi_turn_conversation(self, mock_get, mock_post, api_credentials, mock_response):
        """Test multiple question-answer turns."""
        mock_get.return_value = mock_response({'id': 'session-123'})

        # Multiple questions and final result
        responses = [
            mock_response({'question': {'type': 'Second Form', 'subject': 'John', 'relationship': 'age'}}),
            mock_response({'question': {'type': 'Second Form', 'subject': 'John', 'relationship': 'city'}}),
            mock_response({'result': [{'subject': 'John', 'relationship': 'eligible', 'object': 'yes'}]})
        ]
        mock_post.side_effect = responses

        client = RainbirdAPIClient(**api_credentials)
        client.start()

        # First question
        result1 = client.query(subject='John', relationship='eligible')
        assert 'question' in result1

        # Answer and get second question
        result2 = client.response([
            {'subject': 'John', 'relationship': 'age', 'object': '30', 'certainty': 100}
        ])
        assert 'question' in result2

        # Answer and get final result
        result3 = client.response([
            {'subject': 'John', 'relationship': 'city', 'object': 'New York', 'certainty': 100}
        ])
        assert 'result' in result3

        client.close()
