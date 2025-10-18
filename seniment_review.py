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
        except Exception as ex:
            logger.debug("SentimentSchema.parse_raw other error: %s", ex)

    raise ValueError("Failed to parse model output into SentimentSchema. Raw head:\n" + raw_text[:2000])

def robust_parse_diagnosis(raw_text: str) -> DiagnosisSchema:
    """
    Parse DiagnosisSchema from raw model output -- tries parser, then JSON-like extraction.
    """
    raw_text = (raw_text or "").strip()
    logger.debug("Attempting to parse Diagnosis. Head: %s", repr(raw_text[:300]))

    try:
        return diagnosis_parser.parse(raw_text)
    except Exception as e:
        logger.debug("diagnosis_parser.parse failed: %s", e)

    json_like = _extract_json_like(raw_text)
    if json_like:
        logger.debug("Found JSON-like substring for diagnosis, trying parse_raw")
        try:
            return DiagnosisSchema.parse_raw(json_like)
        except ValidationError as ve:
            logger.debug("DiagnosisSchema.parse_raw validation error: %s", ve)
        except Exception as ex:
            logger.debug("DiagnosisSchema.parse_raw other error: %s", ex)

    raise ValueError("Failed to parse model output into DiagnosisSchema. Raw head:\n" + raw_text[:2000])

# ---------- LangGraph-compatible state typing ----------
class ReviewState(TypedDict, total=False):
    review: str
    sentiment: Literal["positive", "negative"]
    response: str  # raw model output (optional, for debugging)
    diagnosis: Dict[str, Any]

# ---------- Business logic functions (use Ollama) ----------
def find_sentiment(state: ReviewState) -> ReviewState:
    """
    Determine sentiment using Ollama and the sentiment parser.
    """
    review_text = state.get("review", "")
    prompt = (
        f"For the following review find out the sentiment (either 'positive' or 'negative'):\n\n"
        f"\"{review_text}\"\n\n"
        f"{sentiment_format_instructions}\n\n"
        f"Remember: provide only the JSON matching the schema."
    )

    raw = generate_with_ollama(OLLAMA_MODEL, prompt)
    logger.debug("Raw sentiment model output: %s", raw[:500])
    parsed = robust_parse_sentiment(raw)
    state_out: ReviewState = dict(state)  # copy input state
    state_out["sentiment"] = parsed.sentiment
    state_out["response"] = raw
    return state_out

# keep check_sentiment as a helper function but DO NOT register it as a node
def check_sentiment_helper(state: ReviewState) -> str:
    """
    Helper routing function (not a node) — returns the next node name.
    """
    if state.get('sentiment') == 'positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'

def positive_response(state: ReviewState) -> ReviewState:
    """
    Generate a warm thank-you message.
    """
    review_text = state.get("review", "")
    prompt = (
        f"Write a warm thank-you message in response to this review:\n\n\"{review_text}\"\n\n"
        "Also kindly ask the user to leave feedback on our website."
    )
    raw = generate_with_ollama(OLLAMA_MODEL, prompt)
    state_out = dict(state)
    state_out["response"] = raw
    return state_out

def run_diagnosis(state: ReviewState) -> ReviewState:
    """
    Generate a diagnosis for a negative review (issue_type, tone, urgency).
    """
    review_text = state.get("review", "")
    prompt = (
        f"Diagnose this negative review. Return a JSON object with fields: issue_type, tone, urgency.\n\n"
        f"Review:\n{review_text}\n\n"
        f"{diagnosis_format_instructions}\n\n"
        "Return only the JSON matching the schema."
    )
    raw = generate_with_ollama(OLLAMA_MODEL, prompt)
    logger.debug("Raw diagnosis model output: %s", raw[:500])
    parsed = robust_parse_diagnosis(raw)
    state_out = dict(state)
    state_out["diagnosis"] = parsed.model_dump()
    state_out["response"] = raw
    return state_out

def negative_response(state: ReviewState) -> ReviewState:
    """
    Generate an empathetic helpful resolution based on diagnosis.
    """
    diagnosis = state.get("diagnosis", {})
    prompt = (
        "You are a support assistant.\n"
        f"The user had a '{diagnosis.get('issue_type')}' issue, sounded '{diagnosis.get('tone')}', "
        f"and marked urgency as '{diagnosis.get('urgency')}'.\n\n"
        "Write an empathetic, helpful resolution message (one or two short paragraphs)."
    )
    raw = generate_with_ollama(OLLAMA_MODEL, prompt)
    state_out = dict(state)
    state_out["response"] = raw
    return state_out

# ---------- Build & run graph (fixed wiring) ----------
graph = StateGraph(ReviewState)

# Register nodes (do NOT register check_sentiment_helper)
graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

# Entry: start at find_sentiment
graph.add_edge(START, "find_sentiment")

# Conditional routing after find_sentiment using the helper router
def _route_after_find(state: ReviewState) -> str:
    # state has been updated by find_sentiment; route based on sentiment value
    return check_sentiment_helper(state)

graph.add_conditional_edges("find_sentiment", _route_after_find, {
    "positive_response": "positive_response",
    "run_diagnosis": "run_diagnosis",
})

# connect the branches
graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()

if __name__ == "__main__":
    initial_state: ReviewState = {"review": "the product was really bad"}
    try:
        final_state = workflow.invoke(initial_state)
        print("Final state:")
        print(final_state)
    except InvalidUpdateError as e:
        logger.error("Graph update error: %s", e)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
