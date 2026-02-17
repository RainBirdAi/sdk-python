
from typing import Dict, List, Any
import requests


class RainbirdAPIClient:

    def __init__(self, api_host: str, api_key: str, km_id: str):
        """
        Initialize the Rainbird API client.

        Args:
            api_host: Base URL of the Rainbird API (e.g., 'https://api.rainbird.ai')
            api_key: API authentication key
            km_id: Knowledge Map ID
        """
        self.api_host = api_host.rstrip('/')
        self.api_key = api_key
        self.km_id = km_id
        self.session_id = None  # Set after calling start()
        self.session = requests.Session()


    def _get_headers(self, content_type: str = None, version: str = None,
                     evidence_key: str = None, interaction_key: str = None) -> Dict[str, str]:
        """
        Build HTTP headers for API requests.

        Args:
            content_type: Content-Type header (e.g., 'application/json')
            version: API version header (e.g., 'v1')
            evidence_key: Optional key for evidence requests
            interaction_key: Optional key for interaction log requests

        Returns:
            Dictionary of headers to include in the request
        """
        headers = {}

        if self.api_key:
            headers['X-API-Key'] = self.api_key

        if content_type:
            headers['Content-Type'] = content_type

        if version:
            headers['Version'] = version

        if evidence_key:
            headers['x-evidence-key'] = evidence_key

        if interaction_key:
            headers['x-interaction-key'] = interaction_key

        return headers

    def start(self, use_draft: bool = None, version: int = None) -> Dict[str, Any]:
        """
        Start a new session with the knowledge map.

        This must be called before any session-dependent operations
        (inject, query, response, etc.). The session_id is automatically
        stored in the client instance.

        Args:
            use_draft: If True, use the draft version of the knowledge map
            version: Specific version number of the knowledge map to use

        Returns:
            Session data including 'id' (session ID)

        Raises:
            requests.HTTPError: If the API request fails
        """
        url = f"{self.api_host}/start/{self.km_id}"
        headers = self._get_headers()

        params = {}
        if use_draft is not None:
            params['useDraft'] = 'true' if use_draft else 'false'
        if version is not None:
            params['version'] = str(version)

        response = self.session.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        self.session_id = data.get('id')  # Store session ID for subsequent requests
        return data


    def inject(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Inject one or more facts into the current reasoning session.

        Facts are represented as triplets with subject-relationship-object
        structure plus a certainty value (0-100).

        Args:
            facts: List of fact dictionaries, each containing:
                - subject: The subject entity
                - relationship: The relationship
                - object: The object value
                - certainty: Confidence level (0-100), default 100

        Returns:
            Response data from the inject operation

        Raises:
            ValueError: If no session is started
            requests.HTTPError: If the API request fails
        """
        if not self.session_id:
            raise ValueError("Session not started. Call start() first.")

        url = f"{self.api_host}/{self.session_id}/inject"
        headers = self._get_headers(content_type='application/json')

        response = self.session.post(url, headers=headers, json=facts)
        response.raise_for_status()
        return response.json()


    def inject_fact(self, subject: str, relationship: str, object_value: str, certainty: int = 100) -> Dict[str, Any]:
        """
        Convenience method to inject a single fact.

        Args:
            subject: The subject entity
            relationship: The relationship
            object_value: The object value
            certainty: Confidence level (0-100), defaults to 100

        Returns:
            Response data from the inject operation
        """
        fact = {
            "subject": subject,
            "relationship": relationship,
            "object": object_value,
            "certainty": certainty
        }
        return self.inject([fact])


    def query(self, subject: str = None, relationship: str = None,
              object: str = None) -> Dict[str, Any]:
        """
        Query a relationship of the knowledge map.

        Args:
            subject: Optional subject to query about
            relationship: Relationship to query (REQUIRED)
            object: Optional object to query about

        Returns:
            Query result containing:
            - question: Question that needs answering (if any)
            - result: List of results (if query can be answered)
            - state: Current session state

        Raises:
            ValueError: If no session is started or relationship is missing
            requests.HTTPError: If the API request fails
        """
        if not self.session_id:
            raise ValueError("Session not started. Call start() first.")

        url = f"{self.api_host}/{self.session_id}/query"
        headers = self._get_headers(content_type='application/json')

        body = {}
        if subject:
            body['subject'] = subject

        if not relationship:
            raise ValueError("Relationship is mandatory for a query.")

        body['relationship'] = relationship

        if object:
            body['object'] = object

        response = self.session.post(url, headers=headers, json=body)
        response.raise_for_status()
        result = response.json()

        return result


    def response(self, answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Submit answers to questions posed by the reasoning engine.

        Rainbird questions come in two forms:
        - First Form: "Is X true?" (yes/no question) - use answer_first_form()
        - Second Form: "What is the value?" - use answer_second_form()

        You can also construct answer dictionaries manually and pass them here.

        Args:
            answers: List of answer dictionaries with the structure:
                - First Form: {subject, relationship, object, answer, certainty}
                - Second Form: {subject, relationship, object, certainty}

        Returns:
            Response containing next questions or final results

        Raises:
            ValueError: If no session is started
            requests.HTTPError: If the API request fails
        """
        if not self.session_id:
            raise ValueError("Session not started. Call start() first.")

        url = f"{self.api_host}/{self.session_id}/response"
        headers = self._get_headers(content_type='application/json')

        body = {'answers': answers}

        response = self.session.post(url, headers=headers, json=body)
        response.raise_for_status()
        result = response.json()

        return result
    

    def answer_first_form(self, subject: str, relationship: str, object: str, object_value: str, certainty: int = 100) -> Dict[str, Any]:
        """
        Answer a First Form question (yes/no question).

        Args:
            subject: The subject from the question
            relationship: The relationship from the question
            object: The object value from the question
            object_value: Your answer ("yes" or "no")
            certainty: Confidence level (0-100), defaults to 100

        Returns:
            Response containing next questions or final results
        """
        answer_triplet = {
            "subject": subject,
            "relationship": relationship,
            "object": object,  # The object being asked about
            "answer": object_value,  # yes or no
            "certainty": certainty
        }

        return self.response([answer_triplet])


    def answer_second_form(self, subject: str, relationship: str, object: str, certainty: int = 100):
        """
        Answer a Second Form question (provide a value).

        Args:
            subject: The subject from the question
            relationship: The relationship from the question
            object: Your answer (the value you are providing)
            certainty: Confidence level (0-100), defaults to 100

        Returns:
            Response containing next questions or final results
        """
        answer_triplet = {
            "subject": subject,
            "relationship": relationship,
            "object": object,
            "certainty": certainty
        }

        return self.response([answer_triplet])


    def undo(self) -> Dict[str, Any]:
        """
        Undo the last action in the reasoning session.

        This reverts the most recent inject, query, or response action,
        returning the session to its previous state.

        Returns:
            Response data confirming the undo operation

        Raises:
            ValueError: If no session is started
            requests.HTTPError: If the API request fails
        """
        if not self.session_id:
            raise ValueError("Session not started. Call start() first.")

        url = f"{self.api_host}/{self.session_id}/undo"
        headers = self._get_headers(content_type='application/json')

        response = self.session.post(url, headers=headers, json={})
        response.raise_for_status()
        return response.json()


    def interact(self, user_prompt: str) -> Dict[str, Any]:
        """
        Use natural language to interact with the reasoning engine.

        Args:
            user_prompt: Natural language text to send to the engine

        Returns:
            Response containing the engine's natural language reply

        Raises:
            ValueError: If no session is started
            requests.HTTPError: If the API request fails
        """
        if not self.session_id:
            raise ValueError("Session not started. Call start() first.")

        url = f"{self.api_host}/nl/interact"
        headers = self._get_headers(
            content_type='application/json',
            version='v1'
        )

        body = {
            'sessionID': self.session_id,
            'userPrompt': user_prompt
        }

        response = self.session.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()
    

    def evidence(self, fact_id: str, evidence_key: str = None) -> Dict[str, Any]:
        """
        View the source of the fact, including how it was derived when inferred from a rule.

        Args:
            fact_id: The ID of the fact to get evidence for
            evidence_key: Optional key

        Returns:
            Evidence data showing the reasoning chain

        Raises:
            ValueError: If no session is started
            requests.HTTPError: If the API request fails
        """
        if not self.session_id:
            raise ValueError("Session not started. Call start() first.")

        url = f"{self.api_host}/analysis/evidence/{fact_id}/{self.session_id}"
        headers = self._get_headers(evidence_key=evidence_key)

        response = self.session.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    

    def interaction_log(self, interaction_key: str = None,
                       format: str = 'csv') -> str:
        """
        Retrieve a log of all interactions in the current session.

        Args:
            interaction_key: Optional key
            format: Output format, 'csv' (default) or 'json'

        Returns:
            Interaction log data in the requested format

        Raises:
            ValueError: If no session is started
            requests.HTTPError: If the API request fails
        """
        if not self.session_id:
            raise ValueError("Session not started. Call start() first.")

        url = f"{self.api_host}/analysis/interactions/{self.session_id}"
        headers = self._get_headers(interaction_key=interaction_key)

        params = {'format': format}

        response = self.session.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()


    def session_info(self, filters: List[str] = None,
                    relationships: List[str] = None) -> Dict[str, Any]:
        """
        Access session information, such as all of the facts available within a session, or version information for the knowledge map used in a session.

        Args:
            filters: Optional list of filter criteria
            relationships: Optional list of relationships to include

        Returns:
            Session information including state and metadata

        Raises:
            ValueError: If no session is started
            requests.HTTPError: If the API request fails
        """
        if not self.session_id:
            raise ValueError("Session not started. Call start() first.")

        url = f"{self.api_host}/analysis/session/{self.session_id}"
        headers = self._get_headers()

        params = {}
        if filters:
            params['filter'] = ','.join(filters)
        if relationships:
            params['relationships'] = ','.join(relationships)

        response = self.session.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    

    def explain(self, fact_id: str, language: str = 'en') -> Dict[str, Any]:
        """
        Get a natural language explanation of the chain of reasoning for any given fact.

        Args:
            fact_id: The ID of the fact to explain
            language: Language code for the explanation (default: 'en')

        Returns:
            Natural language explanation of the reasoning

        Raises:
            ValueError: If no session is started
            requests.HTTPError: If the API request fails
        """
        if not self.session_id:
            raise ValueError("Session not started. Call start() first.")

        url = f"{self.api_host}/nl/explain"
        headers = self._get_headers(
            content_type='application/json',
            version='v1'
        )

        body = {
            'language': language,
            'sessionID': self.session_id,
            'factID': fact_id
        }

        response = self.session.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()


    def version(self) -> Dict[str, Any]:
        """
        Get the current version of the Rainbird API.

        Returns:
            API version information

        Raises:
            requests.HTTPError: If the API request fails
        """
        url = "https://api.rainbird.ai/version"
        headers = {'Accept': '*/*'}

        response = self.session.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP session and release resources."""
        self.session.close()

    def __enter__(self):
        """Context manager entry - returns the client instance."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically closes the session."""
        self.close()
