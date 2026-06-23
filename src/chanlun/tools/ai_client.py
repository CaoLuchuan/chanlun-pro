from dataclasses import dataclass
from typing import Any

import openai

from chanlun import config


DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "siliconflow": "https://api.siliconflow.cn/v1/",
    "openrouter": "https://openrouter.ai/api/v1",
}


@dataclass(frozen=True)
class AIConfig:
    provider: str
    api_url: str
    api_key: str
    model: str


def resolve_ai_config(config_module=config) -> AIConfig:
    provider = getattr(config_module, "AI_PROVIDER", "").strip().lower()
    api_url = getattr(config_module, "AI_API_URL", "").strip()
    api_key = getattr(config_module, "AI_API_KEY", "").strip()
    model = getattr(config_module, "AI_MODEL", "").strip()
    if api_url == "":
        api_url = DEFAULT_BASE_URLS.get(provider, "")
    elif provider == "openrouter" and api_url.rstrip("/") == "https://openrouter.ai":
        api_url = DEFAULT_BASE_URLS["openrouter"]
    return AIConfig(provider=provider, api_url=api_url, api_key=api_key, model=model)


def _extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n".join([p for p in parts if p != ""])
    return str(content)


def _extract_message_from_response(response: Any) -> tuple[str, str | None]:
    if isinstance(response, str):
        return response, None

    if isinstance(response, dict):
        choices = response.get("choices") or []
        if len(choices) > 0:
            message = choices[0].get("message", {})
            refusal = message.get("refusal")
            return _extract_text_content(message.get("content")), refusal
        if "content" in response:
            return _extract_text_content(response.get("content")), response.get("refusal")

    output_text = getattr(response, "output_text", None)
    if output_text not in (None, ""):
        return _extract_text_content(output_text), None

    choices = getattr(response, "choices", None)
    if choices:
        message = choices[0].message
        refusal = getattr(message, "refusal", None)
        return _extract_text_content(getattr(message, "content", None)), refusal

    if hasattr(response, "content"):
        return _extract_text_content(getattr(response, "content")), getattr(
            response, "refusal", None
        )

    raise ValueError(f"无法识别的 AI 响应格式：{type(response).__name__}")


def request_ai_model(
    prompt: str,
    *,
    config_module=config,
    client_factory=openai.OpenAI,
    response_format: dict[str, Any] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict:
    ai_config = resolve_ai_config(config_module)
    if ai_config.api_key == "" or ai_config.model == "":
        return {
            "ok": False,
            "msg": "未正确配置大模型的 API key 和模型名称",
            "model": ai_config.model,
        }

    try:
        client = client_factory(api_key=ai_config.api_key, base_url=ai_config.api_url)
        kwargs = {
            "model": ai_config.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = client.chat.completions.create(**kwargs)
        content, refusal = _extract_message_from_response(response)
        if content == "" and refusal is not None:
            return {
                "ok": False,
                "msg": f"**[OpenAI API 错误]**: {refusal}",
                "model": ai_config.model,
            }
        return {"ok": True, "msg": content, "model": ai_config.model}
    except openai.OpenAIError as oe:
        return {
            "ok": False,
            "msg": f"**[OpenAI API 错误]**: {str(oe)}",
            "model": ai_config.model,
        }
    except Exception as e:
        return {
            "ok": False,
            "msg": f"**[系统异常]**: {str(e)}",
            "model": ai_config.model,
        }
