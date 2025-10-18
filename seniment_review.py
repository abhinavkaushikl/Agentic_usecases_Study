# review_workflow_ollama.py
import logging
import os
import re
import time
from dotenv import load_dotenv
from typing import TypedDict, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from langchain.output_parsers import PydanticOutputParser
import ollama
from langgraph.graph import StateGraph, START, END
from langgraph.errors import InvalidUpdateError

# ---------- Configuration ----------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_MAX_RETRIES = int(os.environ.get("OLLAMA_MAX_RETRIES", "2"))
OLLAMA_RETRY_DELAY = float(os.environ.get("OLLAMA_RETRY_DELAY", "1.0"))

# ---------- Ollama client ----------
client = ollama.Client(host=OLLAMA_HOST)

def generate_with_ollama(
    model_name: str,
    prompt: str,
    max_retries: int = OLLAMA_MAX_RETRIES,
    retry_delay: float = OLLAMA_RETRY_DELAY,
    stream: bool = False,
) -> str:
    """
    Call Ollama and return the model's textual output (raw).
    Retries on exceptions.
    """
    last_err = None
    for attempt in range(1, max_retries + 2):  # initial attempt + retries
        try:
            logger.debug("Calling Ollama (attempt %d) model=%s", attempt, model_name)
            res = client.generate(model=model_name, prompt=prompt, stream=stream)
            # Ollama's python client may return dict-like response keys: "response" or "text"
            raw_text = res.get("response") or res.get("text") or str(res)
            if raw_text is None:
                raw_text = str(res)
            logger.debug("Ollama response length=%d", len(raw_text))
            return raw_text
        except Exception as e:
            last_err = e
            logger.warning("Ollama generate attempt %d failed: %s", attempt, e)
            if attempt <= max_retries:
                time.sleep(retry_delay)
                continue
            else:
                break
    raise RuntimeError(f"Ollama generation failed after {max_retries + 1} attempts. Last error: {last_err}")

# ---------- Schemas & parsers ----------
class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of review")

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')

sentiment_parser = PydanticOutputParser(pydantic_object=SentimentSchema)
sentiment_format_instructions = sentiment_parser.get_format_instructions()

diagnosis_parser = PydanticOutputParser(pydantic_object=DiagnosisSchema)
diagnosis_format_instructions = diagnosis_parser.get_format_instructions()

def _extract_json_like(raw_text: str) -> Optional[str]:
    """
    Helper to find a {...} JSON-like substring or fenced code block in the model output.
    """
    if not raw_text:
        return None
    m = re.search(r"(\{[\s\S]*\})", raw_text)
    if m:
        return m.group(1)
    m2 = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
    if m2:
        return m2.group(1).strip()
    return None

def robust_parse_sentiment(raw_text: str) -> SentimentSchema:
    """
    Try using LangChain parser first, fallback to extracting JSON-like substring and parsing.
    """
    raw_text = (raw_text or "").strip()
    logger.debug("Attempting to parse Sentiment. Head: %s", repr(raw_text[:300]))

    # 1) LangChain parser (expects model to follow sentiment_format_instructions)
    try:
        return sentiment_parser.parse(raw_text)
    except Exception as e:
        logger.debug("sentiment_parser.parse failed: %s", e)

    # 2) Extract JSON-like and try parse_raw
    json_like = _extract_json_like(raw_text)
    if json_like:
        logger.debug("Found JSON-like substring for sentiment, trying parse_raw")
        try:
            return SentimentSchema.parse_raw(json_like)
        except ValidationError as ve:
            logger.debug("SentimentSchema.parse_raw validation error: %s", ve)
        except Except
