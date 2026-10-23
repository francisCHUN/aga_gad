
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Dict, List, Optional, Type

import requests
from pydantic import BaseModel, ValidationError

from .client import LLMError

log = logging.getLogger("aga.llm.openai")


DEFAULT_TOKENS = [
    "token_key"
]

ANTCHAT_URL = "url"
DEFAULT_MODEL = "model_names"


class DummyBackend:

    name = "dummy"

    @staticmethod
    def _dummy_extract_ids(user: str, marker: str, key: str = "sample_id"):
        import json as _json

        def _ok(v):
            return isinstance(v, (int, str)) and not isinstance(v, bool)

        try:
            prefix = marker[:-1] if marker.endswith(":") else marker
            pos = user.find(prefix)
            if pos == -1:
                return []
            tail = user[pos:]
            block = tail.split("```", 2)[1]
            if block.lstrip().startswith("json"):
                block = block.lstrip()[4:]
            data = _json.loads(block)
            if isinstance(data, list):
                return [d[key] for d in data if isinstance(d, dict) and _ok(d.get(key))]
            if isinstance(data, dict):
                out = []
                for d in data.get("cards", data.get("gaps", [])):
                    if isinstance(d, dict) and _ok(d.get(key)):
                        out.append(d[key])
                return out
        except Exception:
            return []
        return []

    def complete(
        self,
        system: str,
        user: str,
        response_schema: Type[BaseModel],
    ) -> BaseModel:
        schema_name = response_schema.__name__

        if schema_name == "MissingPatterns":
            return response_schema.model_validate(
                {
                    "existing_patterns": ["attribute", "structure"],
                    "missing": [
                        {
                            "name": "context_attack",
                            "severity": 0.85,
                            "rationale": "No node inherits features inconsistent with its neighborhood; context anomalies are entirely absent.",
                            "proposed": False,
                        },
                        {
                            "name": "semantic_camouflage",
                            "severity": 0.8,
                            "rationale": "Attribute-side re-skinning (node mimics a neighbor class while keeping edges) is not represented.",
                            "proposed": False,
                        },
                        {
                            "name": "bridge_attack",
                            "severity": 0.7,
                            "rationale": "No structural bridges linking distant communities exist as anomalies; structure anomalies are local-only.",
                            "proposed": False,
                        },
                        {
                            "name": "hybrid_attack",
                            "severity": 0.6,
                            "rationale": "attr_anomaly and stru_anomaly never co-occur (overlap=0), so hybrid anomalies are missing.",
                            "proposed": False,
                        },
                        {
                            "name": "star_injection",
                            "severity": 0.5,
                            "rationale": "No single hub node is surrounded by many anomalous neighbours; a star anomaly would expose hub-targeting detectors.",
                            "proposed": True,
                            "proposed_spec": (
                                "All new nodes attach to a single shared anchor node in community 0 "
                                "(star topology); features near the global mean so the hub stands out "
                                "structurally. Pure structure-side anomaly; family_type=structure."
                            ),
                        },
                    ],
                    "global_note": "Dataset covers attribute and pure-structural anomalies but lacks contextual, camouflage, bridge, hybrid, and hub-star families.",
                }
            )

        if schema_name == "GenerationStrategy":
            return response_schema.model_validate(
                {
                    "total_new_anomalies": 1200,
                    "per_pattern": [
                        {"pattern": "context_attack", "count": 260, "target_community": 0, "keep_degree_distribution": True},
                        {"pattern": "semantic_camouflage", "count": 80, "target_class": None, "transfer_attribute_from": 0, "keep_degree_distribution": True},
                        {"pattern": "bridge_attack", "count": 160, "keep_degree_distribution": False},
                        {"pattern": "hybrid_attack", "count": 140, "target_community": 0, "keep_degree_distribution": False},
                        {"pattern": "hub_attack", "count": 200, "keep_degree_distribution": False},
                        {
                            "pattern": "star_injection",
                            "count": 120,
                            "keep_degree_distribution": False,
                            "spec": {
                                "family_name": "star_injection",
                                "family_type": "structure",
                                "feature_source": {
                                    "kind": "mean_plus_noise",
                                    "noise_scale": 0.05,
                                    "label": 0,
                                },
                                "edge_topology": {
                                    "kind": "star",
                                    "target_community": 0,
                                    "edges_per_node": 1,
                                    "anchor_spread": False,
                                    "extra_neighbor_prob": 0.0,
                                },
                                "anchor_selection": {
                                    "pool": "normal_nodes",
                                    "target_community": 0,
                                    "spread_across_communities": False,
                                },
                                "degree_cap": 5,
                                "keep_degree_distribution": False,
                                "rationale": "All new nodes attach to one shared hub -- a structural star anomaly.",
                            },
                        },
                    ],
                    "global_constraints": {
                        "preserve_homophily_of_normals": True,
                        "max_degree_delta": 2,
                    },
                    "rationale": "Insert ~5% of |V| spread across the four missing fixed-core families plus one proposed star-injection family (carrying an executable AnomalySpec), keeping normal-node degree distributions and homophily intact.",
                }
            )

        if schema_name == "SampleJudgements":
            ids = self._dummy_extract_ids(user, "SAMPLED NODE EVIDENCE CARDS")
            return response_schema.model_validate({
                "judgements": [
                    {"sample_id": int(i), "assessments": [],
                     "reason": "dummy pre-review: cannot judge without a real model"}
                    for i in ids if isinstance(i, int)
                ],
            })

        if schema_name == "GapSuggestions":
            marker = "GAP HYPOTHESES WITH DEFINITIONS (JSON):" \
                if "GAP HYPOTHESES WITH DEFINITIONS (JSON):" in user \
                else "GAP HYPOTHESES (JSON):"
            gids = self._dummy_extract_ids(user, marker, key="gap_id")
            return response_schema.model_validate({
                "suggestions": [
                    {"gap_id": str(g), "suggested_status": "insufficient_evidence",
                     "evidence_summary": "dummy pre-review: no sample-layer outcome "
                                         "was judged matches; cannot confirm or reject.",
                     "reason": "dummy backend cannot read evidence"}
                    for g in gids
                ],
            })

        try:
            return response_schema.model_validate({})
        except Exception as exc:
            raise LLMError(f"DummyBackend has no canned output for {schema_name}") from exc

    def generate_text(self, system: str, user: str, n: int = 1) -> List[str]:
        seed_topic = ""
        exemplars: List[str] = []
        for line in user.splitlines():
            low = line.strip().lower()
            if low.startswith("seed topic:"):
                seed_topic = line.split(":", 1)[1].strip()
            elif low.startswith("-"):
                exemplars.append(line[1:].strip())
        if not exemplars:
            exemplars = [
                "We study representation learning on attributed graphs and "
                "propose a method that jointly models structure and attributes."
            ]

        out: List[str] = []
        for i in range(n):
            base = exemplars[i % len(exemplars)]
            title = seed_topic or "On graph anomaly detection"
            synth = f"{title}. {base}"
            out.append(synth[:1500])
        return out


class _Stub:

    def __init__(self, **cfg):
        self.cfg: Dict = cfg

    def _not_implemented(self, schema_name: str):
        raise NotImplementedError(
            f"[{self.name}] backend is a stub. To use a real LLM, implement the "
            f"`complete()` method in aga/llm/registry.py to call your provider "
            f"with a response schema for `{schema_name}`, then validate with "
            f"`response_schema.model_validate_json(text)`. Configure via "
            f"`aga.llm.registry.set_backend('{self.name}', **cfg)`."
        )

    def complete(self, system, user, response_schema):
        self._not_implemented(response_schema.__name__)

    def generate_text(self, system, user, n=1) -> List[str]:
        raise NotImplementedError(
            f"[{self.name}] backend is a stub. To generate node text, implement "
            f"`generate_text()` in aga/llm/registry.py to call your provider with "
            f"the system/user prompt and return {n} string samples."
        )


class OpenAIBackend(_Stub):

    name = "openai"

    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.url = self.cfg.get("url", ANTCHAT_URL)
        self.model = self.cfg.get("model", DEFAULT_MODEL)
        self.temperature = self.cfg.get("temperature", 0.2)
        self.max_tokens = self.cfg.get("max_tokens", 1500)
        self.timeout = self.cfg.get("timeout", 120)
        self.tokens: List[str] = list(self.cfg.get("tokens", DEFAULT_TOKENS))
        self._token_idx = 0
        self._lock = threading.Lock()

    def _next_token(self) -> str:
        with self._lock:
            tok = self.tokens[self._token_idx % len(self.tokens)]
            self._token_idx += 1
        return tok

    def _post(self, system: str, user: str, *, enable_sec_check: bool = False) -> str:
        token = self._next_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "SOFA-RpcId": "0.1",
            "SOFA-TraceId": str(uuid.uuid4()),
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        req = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enable_sec_check": enable_sec_check,
        }
        log.debug("AntChat call model=%s tokens_in_pool=%d", self.model, len(self.tokens))
        resp = requests.post(self.url, headers=headers, json=req, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def complete(self, system, user, response_schema):
        schema_name = response_schema.__name__
        schema_json = json.dumps(response_schema.model_json_schema(), ensure_ascii=False)
        sys_prompt = (
            f"{system}\n\nYou MUST reply with ONLY valid JSON that conforms to "
            f"this JSON schema. No prose, no code fences, no explanation.\n\n"
            f"Schema ({schema_name}):\n{schema_json}"
        )
        schema_dict = response_schema.model_json_schema()
        feedback = ""
        last_err: Optional[Exception] = None
        for attempt in range(3):
            raw = None
            try:
                raw = self._post(sys_prompt, user + feedback)
                text = self._extract_json(raw)
                return response_schema.model_validate_json(text)
            except Exception as exc:
                last_err = exc
                if isinstance(exc, ValidationError) and raw is not None:
                    try:
                        data = json.loads(self._extract_json(raw))
                        obj = response_schema.model_validate(
                            self._coerce_value(data, schema_dict, schema_dict)
                        )
                        log.warning(
                            "OpenAI complete() attempt %d: %d schema violation(s) "
                            "salvaged by deterministic coercion into DSL bounds "
                            "(e.g. out-of-range knobs clamped to [min,max]).",
                            attempt + 1, exc.error_count(),
                        )
                        return obj
                    except Exception:
                        pass
                log.warning("OpenAI complete() attempt %d failed: %s", attempt + 1, exc)
                feedback = (
                    f"\n\nYOUR PREVIOUS REPLY WAS INVALID and was rejected with:\n"
                    f"{str(exc)[:600]}\n"
                    f"Fix these problems and return ONLY a corrected, "
                    f"schema-valid JSON object now."
                )
        raise LLMError(
            f"OpenAI backend could not produce schema-valid {schema_name} after retries: {last_err}"
        )

    @staticmethod
    def _coerce_value(value, schema, root):
        if value is None:
            return None
        if "$ref" in schema:
            target = root.get("$defs", {}).get(str(schema["$ref"]).split("/")[-1])
            if target is not None:
                return OpenAIBackend._coerce_value(value, target, root)
            return value
        if "anyOf" in schema:
            for sub in schema["anyOf"]:
                if sub.get("type") == "null":
                    continue
                return OpenAIBackend._coerce_value(value, sub, root)
            return value
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if schema.get("type") == "integer" and isinstance(value, float) \
                    and float(value).is_integer():
                value = int(value)
            lo = schema.get("minimum")
            hi = schema.get("maximum")
            if lo is not None and value < lo:
                value = lo
            if hi is not None and value > hi:
                value = hi
            return value
        if isinstance(value, list):
            items = schema.get("items")
            if items:
                return [OpenAIBackend._coerce_value(v, items, root) for v in value]
            return value
        if isinstance(value, dict):
            props = schema.get("properties", {})
            return {
                k: (OpenAIBackend._coerce_value(v, props[k], root) if k in props else v)
                for k, v in value.items()
            }
        return value

    @staticmethod
    def _extract_json(raw: str) -> str:
        s = raw.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s).strip()
        for open_c, close_c in (("{", "}"), ("[", "]")):
            i = s.find(open_c)
            j = s.rfind(close_c)
            if i != -1 and j != -1 and j > i:
                return s[i : j + 1]
        return s

    def generate_text(self, system: str, user: str, n: int = 1) -> List[str]:
        sys_prompt = (
            f"{system}\n\nReturn exactly {n} items as a plain numbered list "
            f"(one per line, starting with '1.'). No code, no commentary."
        )
        raw = self._post(sys_prompt, user, enable_sec_check=False)
        items = self._parse_numbered_list(raw, n)
        return items

    @staticmethod
    def _parse_numbered_list(raw: str, n: int) -> List[str]:
        out: List[str] = []
        for line in raw.splitlines():
            line = line.strip().rstrip(";")
            if not line:
                continue
            m = re.match(r"^\d+[\).\]:\-]\s*(.*)$", line)
            if m:
                out.append(m.group(1).strip())
            elif out:
                out[-1] = (out[-1] + " " + line).strip()
        while len(out) < n:
            out.append(out[-1] if out else "(synthetic node)")
        return out[:n]


SOHOYO_URL = "url"
SOHOYO_DEFAULT_MODEL = "gpt-6-astra"
SOHOYO_TIMEOUT = 120
SOHOYO_KEY_FILE = "configs/key.local"
SOHOYO_POST_RETRIES = 3
SOHOYO_BACKOFF = 2.0


class SohoyoBackend(OpenAIBackend):

    name = "sohoyo"

    def __init__(self, **cfg):
        self.cfg = cfg
        self.url = cfg.get("url", SOHOYO_URL)
        self.model = cfg.get("model", SOHOYO_DEFAULT_MODEL)
        self.temperature = cfg.get("temperature", 0.2)
        self.max_tokens = cfg.get("max_tokens", 2500)
        self.timeout = cfg.get("timeout", SOHOYO_TIMEOUT)
        self.key_file = cfg.get("key_file", SOHOYO_KEY_FILE)
        self.api_key = (
            cfg.get("api_key")
            or os.environ.get("SOHOYO_API_KEY", "")
            or self._read_key_file()
        )
        if not self.api_key:
            raise LLMError(
                "sohoyo backend requires an API key. Either (a) write it as a "
                "single line into configs/sohoyo_key.local (gitignored, this is "
                "the recommended local setup), (b) export SOHOYO_API_KEY=<key>, "
                "or (c) pass set_backend('sohoyo', api_key=...). The key is "
                "intentionally never committed -- this repository is public."
            )
        self._post_retries = int(cfg.get("post_retries", SOHOYO_POST_RETRIES))
        self._backoff = float(cfg.get("backoff", SOHOYO_BACKOFF))

    def _read_key_file(self) -> str:
        try:
            with open(self.key_file, encoding="utf-8") as fh:
                key = fh.read().strip()
            return key or ""
        except OSError:
            return ""

    def _post(self, system: str, user: str, *, enable_sec_check: bool = False) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        req = {
            "model": self.model,
            "messages": messages,
        }
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._post_retries + 1):
            try:
                resp = requests.post(
                    self.url, headers=headers, json=req, timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:
                last_exc = exc
                log.warning("sohoyo request failed (attempt %d/%d): %s",
                            attempt, self._post_retries, exc)
                if attempt < self._post_retries:
                    time.sleep(min(self._backoff * (2 ** (attempt - 1)), 60.0))
        raise LLMError(
            f"sohoyo request failed after {self._post_retries} attempts: {last_exc}"
        ) from last_exc


class AnthropicBackend(_Stub):
    name = "anthropic"


class LocalBackend(_Stub):
    name = "local"


_BACKENDS: Dict[str, type] = {
    "dummy": DummyBackend,
    "openai": OpenAIBackend,
    "sohoyo": SohoyoBackend,
    "anthropic": AnthropicBackend,
    "local": LocalBackend,
}

_active: LLMClientLike = None
_active_name: str = "dummy"
_active_cfg: Dict = {}


from .client import LLMClient as LLMClientLike


def set_backend(name: str, **cfg) -> None:
    global _active, _active_name, _active_cfg
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {list(_BACKENDS)}")
    _active_name = name
    _active_cfg = dict(cfg)
    _active = _BACKENDS[name](**cfg)


def get_backend() -> LLMClientLike:
    global _active
    if _active is None:
        _active = _BACKENDS["dummy"]()
    return _active


def active_name() -> str:
    return _active_name


def complete(system: str, user: str, response_schema: Type[BaseModel]) -> BaseModel:
    return get_backend().complete(system, user, response_schema)


def generate_text(system: str, user: str, n: int = 1) -> List[str]:
    return get_backend().generate_text(system, user, n)


__all__ = [
    "DummyBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "LocalBackend",
    "set_backend",
    "get_backend",
    "complete",
    "generate_text",
    "active_name",
]