"""
Unified Multi-Provider LLM Client for the Autonomous Research Mesh.
Coordinates:
1. Local Ollama (Uncensored Qwen 9B / 27B) -> Hypothesis generation & raw factor math.
2. Google Gemini API -> High-capacity execution sandbox & environmental translation.
3. OpenAI API (ChatGPT) -> Adversarial referee, causal audit, & qualitative plausibility.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    tokens_prompt: int = 0
    tokens_completion: int = 0
    duration_sec: float = 0.0


# -------------------------------------------------------------------------
# 1. LOCAL OLLAMA CLIENT (QWEN UNCENSORED)
# -------------------------------------------------------------------------
class LocalOllamaClient:
    """
    Client for local Ollama instance running uncensored reasoning models.
    Default Model: HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive (Q4_K_M)
    Upgrade Model: nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF

    Why Uncensored Models are Required:
    Autonomous quantitative research on market microstructure routinely examines
    aggressive concepts (e.g. order flow exploitation, liquidity drains, front-running
    cascades, toxic flow imbalances). Commercial alignment guardrails frequently trigger
    false-positive safety refusals on financial mechanics. Uncensored fine-tunes allow
    unfettered epistemic reasoning without refusal rate degradation.
    """
    DEFAULT_MODEL = "hf.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q4_K_M"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: int = 120,
    ):
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.environ.get("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self.timeout_sec = timeout_sec

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.6,
        max_tokens: int = 3000,
    ) -> LLMResponse:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        t0 = time.time()
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            return LLMResponse(
                content=res_json.get("response", ""),
                model=self.model,
                provider="OLLAMA_LOCAL",
                tokens_prompt=res_json.get("prompt_eval_count", 0),
                tokens_completion=res_json.get("eval_count", 0),
                duration_sec=elapsed,
            )
        except Exception as e:
            raise RuntimeError(f"Ollama local inference failed on {self.model}: {e}")


# -------------------------------------------------------------------------
# 2. GOOGLE GEMINI API CLIENT (EXECUTION & SANDBOX ANALYSIS)
# -------------------------------------------------------------------------
class GeminiClient:
    """
    Official REST client for Google Gemini API.
    Used for complex multi-asset reasoning, environmental translation, and sandbox evaluation.
    Requires GEMINI_API_KEY environment variable.
    """
    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: int = 60,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("GEMINI_MODEL", self.DEFAULT_MODEL)
        self.timeout_sec = timeout_sec

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.4,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> LLMResponse:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured. Please export GEMINI_API_KEY='your-key'.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"[SYSTEM INSTRUCTIONS]\n{system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions strictly."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
        )

        for attempt in range(1, max_retries + 1):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                elapsed = time.time() - t0
                
                candidates = res_json.get("candidates", [])
                if not candidates:
                    raise ValueError(f"No candidates returned: {res_json}")
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                usage = res_json.get("usageMetadata", {})

                return LLMResponse(
                    content=text,
                    model=self.model,
                    provider="GEMINI_CLOUD",
                    tokens_prompt=usage.get("promptTokenCount", 0),
                    tokens_completion=usage.get("candidatesTokenCount", 0),
                    duration_sec=elapsed,
                )
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Gemini API call failed after {max_retries} attempts: {e}")
                time.sleep(2.0 * attempt)

        raise RuntimeError("Unreachable")


# -------------------------------------------------------------------------
# 3. OPENAI API CLIENT (CHATGPT ADVERSARIAL REFEREE)
# -------------------------------------------------------------------------
class OpenAIClient:
    """
    Official REST client for OpenAI API (ChatGPT).
    Used as the adversarial referee to audit causal rationale and detect curve-fitting.
    Requires OPENAI_API_KEY environment variable.
    """
    DEFAULT_MODEL = "gpt-5.6-luna"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: int = 60,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", self.DEFAULT_MODEL)
        self.timeout_sec = timeout_sec

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        max_retries: int = 3,
    ) -> LLMResponse:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured. Please export OPENAI_API_KEY='your-key'.")

        url = "https://api.openai.com/v1/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        for attempt in range(1, max_retries + 1):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                elapsed = time.time() - t0

                choices = res_json.get("choices", [])
                if not choices:
                    raise ValueError(f"No choices returned: {res_json}")
                text = choices[0].get("message", {}).get("content", "")
                usage = res_json.get("usage", {})

                return LLMResponse(
                    content=text,
                    model=self.model,
                    provider="OPENAI_CLOUD",
                    tokens_prompt=usage.get("prompt_tokens", 0),
                    tokens_completion=usage.get("completion_tokens", 0),
                    duration_sec=elapsed,
                )
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(f"OpenAI API call failed after {max_retries} attempts: {e}")
                time.sleep(2.0 * attempt)

        raise RuntimeError("Unreachable")


# -------------------------------------------------------------------------
# 4. UNIFIED AGENT MESH
# -------------------------------------------------------------------------
class UnifiedAgentMesh:
    """
    Coordinates the tri-model research mesh:
    - Generator: Ollama Local (Uncensored Qwen 9B/27B)
    - Executor: Gemini Cloud API
    - Referee: OpenAI Cloud API (ChatGPT)
    """

    def __init__(
        self,
        ollama_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
        openai_model: Optional[str] = None,
    ):
        self.ollama = LocalOllamaClient(model=ollama_model)
        self.gemini = GeminiClient(model=gemini_model)
        self.openai = OpenAIClient(model=openai_model)

    def propose_hypothesis(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Route to local uncensored Qwen model."""
        return self.ollama.generate(prompt=prompt, system_prompt=system_prompt)

    def execute_sandbox_analysis(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Route to Gemini."""
        return self.gemini.generate(prompt=prompt, system_prompt=system_prompt)

    def audit_referee(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Route to OpenAI (ChatGPT)."""
        return self.openai.generate(prompt=prompt, system_prompt=system_prompt)
