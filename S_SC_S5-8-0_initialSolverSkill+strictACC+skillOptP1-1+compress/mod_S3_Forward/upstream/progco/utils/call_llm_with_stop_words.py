"""
Stream call to OpenAI API and determine whether to stop through string matching
Note: The stop parameter in OpenAI API does not work with Claude model
https://github.com/MacPaw/OpenAI/issues/198 
https://community.openai.com/t/interrupting-completion-stream-in-python/30628/7
"""
import time
import json
import os
import threading
from pathlib import Path
from openai import OpenAI
from mod_C0_compress.request_guard import check_chat_request_budget
from typing import Optional, List

from .sample_budget import consume_llm_call, gpu_queue_client_kwargs


import api_profiles as _api_profiles
_api_config = _api_profiles.get_api_config()
_CALL_STATE = threading.local()


def _set_last_call_stats(stats):
    _CALL_STATE.last_call_stats = dict(stats)


def get_last_call_stats():
    return dict(getattr(_CALL_STATE, "last_call_stats", {}))


_TOKEN_RECORD_LOCK = threading.Lock()


def _token_record_path():
    return (
        os.environ.get("SPQ_TOKEN_LENGTH_RECORD_PATH")
        or os.environ.get("SKILLOPT_TOKEN_LENGTH_RECORD_PATH")
        or ""
    ).strip()


def _optional_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _token_length_exceeded(finish_reason):
    reason = str(finish_reason or "").strip().lower()
    if reason in {"length", "max_tokens", "max_output_tokens"}:
        return True
    return "token" in reason and ("max" in reason or "limit" in reason)


def _record_token_length_event(
    *,
    stage,
    model,
    request_type,
    finish_reason,
    max_tokens_configured,
    output_chars,
):
    path = _token_record_path()
    if not path:
        return
    max_i = _optional_int(max_tokens_configured)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "S3",
        "stage": stage,
        "model": model,
        "request_type": request_type,
        "finish_reason": None if finish_reason is None else str(finish_reason),
        "token_length_exceeded": _token_length_exceeded(finish_reason),
        "max_tokens_configured": max_i,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "output_chars": output_chars,
        "near_configured_limit": False,
    }
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with _TOKEN_RECORD_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        return


def _is_timeout_error(exc):
    text = f"{type(exc).__name__}: {str(exc)}".lower()
    return "timeout" in text or "timed out" in text


def _llm_timeout_seconds():
    try:
        return max(1.0, float(os.environ.get("SPQ_LLM_TIMEOUT_SECONDS", "120")))
    except ValueError:
        return 120.0


def _llm_stop_max_retries():
    try:
        value = os.environ.get("SPQ_LLM_STOP_MAX_RETRIES") or os.environ.get("SPQ_LLM_MAX_RETRIES", "4")
        return max(1, int(value))
    except ValueError:
        return 4


def _apply_stop_words(text, stop_words):
    if not stop_words:
        return text, False, None

    for stop_word in stop_words:
        if stop_word and stop_word in text:
            return text.split(stop_word)[0] + stop_word, True, stop_word

    return text, False, None

client = OpenAI(
    api_key=_api_config["api_key"],
    base_url=_api_config["base_url"],
    timeout=_llm_timeout_seconds(),
    max_retries=0,
    **gpu_queue_client_kwargs(_api_config["base_url"]),
)

def stream_openai(model,messages,temperature,stop_words,max_tokens):
    api_model = _api_profiles.api_model_name(model, _api_config)
    request_kwargs = dict(
        model=api_model,
        messages=messages,
        stream=True,
        temperature=temperature,
    )
    if max_tokens is not None and max_tokens > 0:
        request_kwargs["max_tokens"] = max_tokens
    extra_body = _api_profiles.extra_body_for_model(model, _api_config, api_model=api_model)
    if extra_body is not None:
        request_kwargs["extra_body"] = extra_body

    check_chat_request_budget(client, request_kwargs)
    consume_llm_call()
    response = client.chat.completions.create(**request_kwargs)
    is_stop=False
    gen_stop_word=None
    finish_reason=None

    content_text = ""
    try:
        for index, chunk in enumerate(response):
            if index == 0:
                _api_profiles.record_response_metadata(chunk, stage="work_stream", model=api_model)
            if not getattr(chunk, "choices", None):
                continue

            choice0 = chunk.choices[0]
            chunk_finish_reason = getattr(choice0, "finish_reason", None)
            if chunk_finish_reason:
                finish_reason = chunk_finish_reason
            delta = choice0.delta
            content = getattr(delta, "content", None)

            if content is not None:
                content_text += content
                content_text, is_stop, gen_stop_word = _apply_stop_words(content_text, stop_words)
                if is_stop:
                    finish_reason = "manual_stop_word"
                    break
    finally:
        response.close()

    if content_text.strip():
        return content_text,is_stop,gen_stop_word,finish_reason

    raise RuntimeError("empty content")

def call_llm_with_stop_words(
            model: str, 
            messages: List[dict],
            temperature: float = None, 
            stop_strs: Optional[List[str]] = None,
            max_tokens: int = None):
    if _api_profiles.is_openai_compatible_model(model):
        retries = 0
        max_retries = _llm_stop_max_retries()
        stats = {"error_count": 0, "timeout_count": 0, "last_error": ""}
        _set_last_call_stats(stats)
        while retries < max_retries:
            try:
                response,is_stop,gen_stop_word,finish_reason = stream_openai(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stop_words=stop_strs,
                    max_tokens=max_tokens,
                )
                stats.update({
                    "content_chars": len(response or ""),
                    "finish_reason": finish_reason,
                })
                _record_token_length_event(
                    stage="s3_stream_call",
                    model=_api_profiles.api_model_name(model, _api_config),
                    request_type="chat_stream",
                    finish_reason=finish_reason,
                    max_tokens_configured=max_tokens,
                    output_chars=len(response or ""),
                )
                _set_last_call_stats(stats)
                return response,is_stop,gen_stop_word
            except KeyboardInterrupt:
                print("Operation canceled by user.")
                return ''
            except Exception as e:
                if isinstance(e, ValueError) and not isinstance(e, json.JSONDecodeError):
                    raise
                stats["error_count"] += 1
                stats["last_error"] = f"{type(e).__name__}: {str(e)}"
                if _is_timeout_error(e):
                    stats["timeout_count"] += 1
                if retries + 1 < max_retries:
                    wait_seconds = retries * 10
                    print(
                        f"Someting wrong:{e}. Retrying in "
                        f"{wait_seconds} seconds..."
                    )
                    time.sleep(wait_seconds)
                retries += 1
        _set_last_call_stats(stats)
        return '',False,None
    else:
        raise
