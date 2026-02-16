
from typing import Dict, List, Any
import requests

class RainbirdAPIClient:

    def __init__(self, api_host: str, api_key: str, km_id: str):

        self.api_host = api_host.rstrip('/')
        self.api_key = api_key
        self.km_id = km_id
        self.session_id = None
        self.session = requests.Session()


    def _get_headers(self, content_type: str = None, version: str = None,
                     evidence_key: str = None, interaction_key: str = None) -> Dict[str, str]:
        """Build request headers."""
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
        self.session_id = data.get('id')
        return data


    def inject(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
      
        if not self.session_id:
            raise ValueError("Session not started. Call start() first.")

        url = f"{self.api_host}/{self.session_id}/inject"
        headers = self._get_headers(content_type='application/json')

        response = self.session.post(url, headers=headers, json=facts)
        response.raise_for_status()
        return response.json()


    def inject_fact(self, subject: str, relationship: str, object_value: str, certainty: int = 100) -> Dict[str, Any]:
        fact = {
            "subject": subject,
            "relationship": relationship,
            "object": object_value,
            "certainty": certainty
        }
        return self.inject([fact])


    def query(self, subject: str = None, relationship: str = None,
              object: str = None) -> Dict[str, Any]:
     
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

        # First Form: include BOTH 'object' (from question) and 'answer' (yes/no)
        answer_triplet = {
            "subject": subject,
            "relationship": relationship,
            "object": object,  # The object being asked about
            "answer": object_value,  # yes or no
            "certainty": certainty
        }

        return self.response([answer_triplet])


    def answer_second_form(self, subject: str, relationship: str, object: str, certainty: int = 100):
        # Second Form: use 'object' field only
        answer_triplet = {
            "subject": subject,
            "relationship": relationship,
            "object": object,
            "certainty": certainty
        }

        return self.response([answer_triplet])


    def undo(self) -> Dict[str, Any]:

        if not self.session_id:
            raise ValueError("Session not started. Call start() first.")

        url = f"{self.api_host}/{self.session_id}/undo"
        headers = self._get_headers(content_type='application/json')

        response = self.session.post(url, headers=headers, json={})
        response.raise_for_status()
        return response.json()


    def interact(self, user_prompt: str) -> Dict[str, Any]:

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
        
        if not self.session_id:
            raise ValueError("Session not started. Call start() first.")

        url = f"{self.api_host}/analysis/evidence/{fact_id}/{self.session_id}"
        headers = self._get_headers(evidence_key=evidence_key)

        response = self.session.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    

    def interaction_log(self, interaction_key: str = None,
                       format: str = 'csv') -> str:

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
        
        url = "https://api.rainbird.ai/version"
        headers = {'Accept': '*/*'}

        response = self.session.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
