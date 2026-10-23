
from __future__ import annotations

from typing import List, Protocol, Type, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class LLMClient(Protocol):

    name: str

    def complete(
        self,
        system: str,
        user: str,
        response_schema: Type[BaseModel],
    ) -> BaseModel:
        ...

    def generate_text(self, system: str, user: str, n: int = 1) -> List[str]:
        ...


class LLMError(RuntimeError):