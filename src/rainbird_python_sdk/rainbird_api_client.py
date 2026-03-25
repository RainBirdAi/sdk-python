from typing import Dict, List, Any, Union
import requests


AnswerDict = Dict[str, Any]
AnswersInput = Union[AnswerDict, List[AnswerDict]]


class RainbirdAPIClient:
    def __init__(self, api_host: str, api_key: str, km_id: str):
        """
        Initialize the Rainbird API client.

        Args:
            api_host: Base URL of the Rainbird API (e.g., 'https://api.rainbird.ai')
            api_key: API authentication key
            km_id: Knowledge Map ID
        """
        self.api_host = api_host.rstrip("/")
        self.api_key = api_key
        self.km_id = km_id
        self.session_id = None
        self.session = requests.Session()

    def _get_headers(
        self,
        content_type: str = None,
        version: str = None,
        evidence_key: str = None,
        interaction_key: str = None,
    ) -> Dict[str, str]:
        """
        Build HTTP headers for API requests.
        """
        headers: Dict[str, str] = {}

        if self.api_key:
            headers["X-API-Key"] = self.api_key

        if content_type:
            headers["Content-Type"] = content_type

        if version:
            headers["Version"] = version

        if evidence_key:
            headers["x-evidence-key"] = evidence_key

        if interaction_key:
            headers["x-interaction-key"] = interaction_key

        return headers

    def _ensure_session_started(self) -> None:
        """Raise if there is no active session."""
        if not self.session_id:
            raise ValueError("Session not started. Call start() first.")

    def _build_second_form_answer(
        self,
        subject: str,
        relationship: str,
        object_value: Any,
        certainty: int = 100,
    ) -> AnswerDict:
        """Build an answer payload for a second-form question."""
        return {
            "subject": subject,
            "relationship": relationship,
            "object": object_value,
            "certainty": certainty,
        }

    def _build_first_form_answer(
        self,
        subject: str,
        relationship: str,
        object_value: Any,
        answer_value: Any,
        certainty: int = 100,
    ) -> AnswerDict:
        """Build an answer payload for a first-form question."""
        return {
            "subject": subject,
            "relationship": relationship,
            "object": object_value,
            "answer": answer_value,
            "certainty": certainty,
        }

    def _validate_answer_dict(self, answer: AnswerDict) -> None:
        """Validate the minimum structure of an answer dictionary."""
        if not isinstance(answer, dict):
            raise ValueError("Each answer must be a dictionary.")

        required_fields = {"subject", "relationship", "object", "certainty"}
        missing = required_fields - set(answer.keys())
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Answer is missing required field(s): {missing_str}")

    def _normalize_answers(self, answers: AnswersInput) -> List[AnswerDict]:
        """
        Normalize answer input so response() can accept either:
        - a single answer dictionary
        - a list of answer dictionaries
        """
        if isinstance(answers, dict):
            normalized = [answers]
        elif isinstance(answers, list):
            normalized = answers
        else:
            raise ValueError("answers must be a dictionary or a list of dictionaries.")

        if not normalized:
            raise ValueError("answers must not be empty.")

        for answer in normalized:
            self._validate_answer_dict(answer)

        return normalized

    def start(self, use_draft: bool = None, version: int = None) -> Dict[str, Any]:
        """
        Start a new session with the knowledge map.
        """
        url = f"{self.api_host}/start/{self.km_id}"
        headers = self._get_headers()

        params = {}
        if use_draft is not None:
            params["useDraft"] = "true" if use_draft else "false"
        if version is not None:
            params["version"] = str(version)

        response = self.session.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        self.session_id = data.get("id")
        return data

    def inject(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Inject one or more facts into the current reasoning session.
        """
        self._ensure_session_started()

        url = f"{self.api_host}/{self.session_id}/inject"
        headers = self._get_headers(content_type="application/json")

        response = self.session.post(url, headers=headers, json=facts)
        response.raise_for_status()
        return response.json()

    def inject_fact(
        self,
        subject: str,
        relationship: str,
        object_value: str,
        certainty: int = 100,
    ) -> Dict[str, Any]:
        """
        Convenience method to inject a single fact.
        """
        fact = {
            "subject": subject,
            "relationship": relationship,
            "object": object_value,
            "certainty": certainty,
        }
        return self.inject([fact])

    def query(
        self,
        subject: str = None,
        relationship: str = None,
        object: str = None,
    ) -> Dict[str, Any]:
        """
        Query a relationship of the knowledge map.
        """
        self._ensure_session_started()

        url = f"{self.api_host}/{self.session_id}/query"
        headers = self._get_headers(content_type="application/json")

        body = {}
        if subject:
            body["subject"] = subject

        if not relationship:
            raise ValueError("Relationship is mandatory for a query.")

        body["relationship"] = relationship

        if object:
            body["object"] = object

        response = self.session.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()

    def response(self, answers: AnswersInput) -> Dict[str, Any]:
        """
        Submit answers to questions posed by the reasoning engine.

        This method now accepts either:
        - a single answer dictionary
        - a list of answer dictionaries
        """
        self._ensure_session_started()

        url = f"{self.api_host}/{self.session_id}/response"
        headers = self._get_headers(content_type="application/json")

        normalized_answers = self._normalize_answers(answers)
        body = {"answers": normalized_answers}

        response = self.session.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()

    def answer_first_form(
        self,
        subject: str,
        relationship: str,
        object: str,
        object_value: str,
        certainty: int = 100,
    ) -> Dict[str, Any]:
        """
        Answer a First Form question (yes/no question).
        """
        answer_triplet = self._build_first_form_answer(
            subject=subject,
            relationship=relationship,
            object_value=object,
            answer_value=object_value,
            certainty=certainty,
        )
        return self.response(answer_triplet)

    def answer_second_form(
        self,
        subject: str,
        relationship: str,
        object: str,
        certainty: int = 100,
    ) -> Dict[str, Any]:
        """
        Answer a Second Form question (provide a value).
        """
        answer_triplet = self._build_second_form_answer(
            subject=subject,
            relationship=relationship,
            object_value=object,
            certainty=certainty,
        )
        return self.response(answer_triplet)

    def answer_question(
        self,
        question: Dict[str, Any],
        answer_or_answers: Union[Any, List[Any]],
        certainty: int = 100,
    ) -> Dict[str, Any]:
        """
        Higher-level helper to answer a question returned by the reasoning engine.

        Supports:
        - First Form questions with a single yes/no answer
        - Second Form questions with one value or many values

        Args:
            question: The question object returned by the API.
            answer_or_answers:
                - First Form: a single answer value, e.g. "yes" or "no"
                - Second Form: a single object value or a list of object values
            certainty: Certainty level to apply to generated answers

        Returns:
            Response containing next questions or final results
        """
        if not isinstance(question, dict):
            raise ValueError("question must be a dictionary.")

        question_type = question.get("type")
        subject = question.get("subject")
        relationship = question.get("relationship")

        if not question_type:
            raise ValueError("question must include a 'type' field.")
        if not subject:
            raise ValueError("question must include a 'subject' field.")
        if not relationship:
            raise ValueError("question must include a 'relationship' field.")

        if "First Form" in question_type:
            object_value = question.get("object")
            if object_value is None:
                raise ValueError("First Form question must include an 'object' field.")

            if isinstance(answer_or_answers, list):
                if len(answer_or_answers) != 1:
                    raise ValueError("First Form questions expect a single answer.")
                answer_value = answer_or_answers[0]
            else:
                answer_value = answer_or_answers

            return self.response(
                self._build_first_form_answer(
                    subject=subject,
                    relationship=relationship,
                    object_value=object_value,
                    answer_value=answer_value,
                    certainty=certainty,
                )
            )

        if "Second Form" in question_type:
            if isinstance(answer_or_answers, list):
                answers = [
                    self._build_second_form_answer(
                        subject=subject,
                        relationship=relationship,
                        object_value=value,
                        certainty=certainty,
                    )
                    for value in answer_or_answers
                ]
                return self.response(answers)

            return self.response(
                self._build_second_form_answer(
                    subject=subject,
                    relationship=relationship,
                    object_value=answer_or_answers,
                    certainty=certainty,
                )
            )

        raise ValueError(f"Unsupported question type: {question_type}")

    def undo(self) -> Dict[str, Any]:
        """
        Undo the last action in the reasoning session.
        """
        self._ensure_session_started()

        url = f"{self.api_host}/{self.session_id}/undo"
        headers = self._get_headers(content_type="application/json")

        response = self.session.post(url, headers=headers, json={})
        response.raise_for_status()
        return response.json()

    def interact(self, user_prompt: str) -> Dict[str, Any]:
        """
        Use natural language to interact with the reasoning engine.
        """
        self._ensure_session_started()

        url = f"{self.api_host}/nl/interact"
        headers = self._get_headers(content_type="application/json", version="v1")

        body = {
            "sessionID": self.session_id,
            "userPrompt": user_prompt,
        }

        response = self.session.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()

    def evidence(self, fact_id: str, evidence_key: str = None) -> Dict[str, Any]:
        """
        View the source of the fact, including how it was derived when inferred from a rule.
        """
        self._ensure_session_started()

        url = f"{self.api_host}/analysis/evidence/{fact_id}/{self.session_id}"
        headers = self._get_headers(evidence_key=evidence_key)

        response = self.session.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def interaction_log(self, interaction_key: str = None, format: str = "csv") -> str:
        """
        Retrieve a log of all interactions in the current session.
        """
        self._ensure_session_started()

        url = f"{self.api_host}/analysis/interactions/{self.session_id}"
        headers = self._get_headers(interaction_key=interaction_key)

        params = {"format": format}

        response = self.session.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def session_info(
        self,
        filters: List[str] = None,
        relationships: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Access session information for the current session.
        """
        self._ensure_session_started()

        url = f"{self.api_host}/analysis/session/{self.session_id}"
        headers = self._get_headers()

        params = {}
        if filters:
            params["filter"] = ",".join(filters)
        if relationships:
            params["relationships"] = ",".join(relationships)

        response = self.session.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def explain(self, fact_id: str, language: str = "en") -> Dict[str, Any]:
        """
        Get a natural language explanation of the chain of reasoning for any given fact.
        """
        self._ensure_session_started()

        url = f"{self.api_host}/nl/explain"
        headers = self._get_headers(content_type="application/json", version="v1")

        body = {
            "language": language,
            "sessionID": self.session_id,
            "factID": fact_id,
        }

        response = self.session.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()

    def version(self) -> Dict[str, Any]:
        """
        Get the current version of the Rainbird API.
        """
        url = "https://api.rainbird.ai/version"
        headers = {"Accept": "*/*"}

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