# review_workflow_ollama.py
import logging
import os
import re
import time
from dotenv import load_dotenv
from typing import TypedDict, Literal, Any, List
from pydantic import BaseModel, Field, ValidationError
from langchain.output_parsers import PydanticOutputParser
import ollama
from langgraph.graph import StateGraph, START, END
from langgraph.errors import InvalidUpdateError
from langchain_core.messages import SystemMessage, HumanMessage

# ---------- Configuration ----------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_MAX_RETRIES = int(os.environ.get("OLLAMA_MAX_RETRIES", "2"))
OLLAMA_RETRY_DELAY = float(os.environ.get("OLLAMA_RETRY_DELAY", "1.0"))
OLLAMA_MAX_TOKENS = int(os.environ.get("OLLAMA_MAX_TOKENS", "2048"))

# ---------- Ollama client ----------
client = ollama.Client(host=OLLAMA_HOST)


# ---------- Pydantic models ----------
class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(..., description="Feedback for the tweet.")


# parser for structured evaluation outputs
evaluation_parser = PydanticOutputParser(pydantic_object=TweetEvaluation)
format_instructions = evaluation_parser.get_format_instructions()


# ---------- Prompt formatting helpers ----------
def _stringify_message_item(item: Any) -> str:
    """
    Return a readable string for a single message item.
    Handles: SystemMessage, HumanMessage, dict with 'content', str, others.
    """
    try:
        if hasattr(item, "content"):
            # Use a short role label
            cls_name = item.__class__.__name__
            if "System" in cls_name:
                prefix = "System: "
            elif "Human" in cls_name or "User" in cls_name:
                prefix = "Human: "
            else:
                prefix = f"{cls_name}: "
            return prefix + str(item.content).strip()
        if isinstance(item, dict) and "content" in item:
            role = item.get("role", "Message")
            return f"{role}: {str(item['content']).strip()}"
        return str(item).strip()
    except Exception:
        # Fallback to safe str conversion
        return str(item)


def format_prompt(prompt: Any) -> str:
    """
    Always return a plain string prompt acceptable to ollama.Client.generate.
    Accepts: string, list/tuple (messages/dicts/strings), dict, or any object.
    """
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, (list, tuple)):
        pieces = []
        for it in prompt:
            pieces.append(_stringify_message_item(it))
        # separate message blocks to keep readability
        return "\n\n".join(pieces)

    if isinstance(prompt, dict):
        # common shapes: {"messages": [...]} or {"content": "..."}
        if "content" in prompt:
            return str(prompt["content"]).strip()
        if "messages" in prompt and isinstance(prompt["messages"], (list, tuple)):
            return format_prompt(prompt["messages"])
        # fallback to stringify dict
        return str(prompt)

    # fallback for any other object
    return str(prompt)

import json
from pydantic import ValidationError as PydanticValidationError

# ---------- Improved generate_with_ollama (safe extraction) ----------
def generate_with_ollama(model_name: str, prompt: Any, max_retries: int = OLLAMA_MAX_RETRIES,
                         retry_delay: float = OLLAMA_RETRY_DELAY, stream: bool = False) -> str:
    """
    Call Ollama and return the model's textual output (raw).
    Extracts 'response' or 'text' fields when client returns richer objects/reprs.
    """
    last_err = None
    prompt_str = format_prompt(prompt)
    logger.debug("generate_with_ollama: prompt type after formatting=%s", type(prompt_str))
    logger.debug("generate_with_ollama: formatted prompt head=%s", repr(prompt_str[:400]))

    for attempt in range(1, max_retries + 2):
        try:
            logger.debug("Calling Ollama (attempt %d) model=%s", attempt, model_name)
            res = client.generate(model=model_name, prompt=prompt_str, stream=stream)

            # Try to extract the model textual output safely:
            raw_text = None
            if isinstance(res, dict):
                raw_text = res.get("response") or res.get("text") or res.get("result")
            else:
                # object-like
                for attr in ("response", "text", "result"):
                    if hasattr(res, attr):
                        raw_text = getattr(res, attr)
                        break
            if raw_text is None:
                # fallback to str()
                raw_text = str(res)

            if raw_text is None:
                raw_text = ""

            logger.debug("Ollama response length=%d", len(raw_text))
            return raw_text
        except Exception as e:
            last_err = e
            logger.warning("Ollama generate attempt %d failed: %s", attempt, e)
            if attempt <= max_retries:
                time.sleep(retry_delay)
            else:
                break
    raise RuntimeError(f"Ollama generation failed after {max_retries + 1} attempts. Last error: {last_err}")


# ---------- Robust evaluation parser (clean + json.loads + Pydantic model_validate) ----------
def robust_parse_evaluation(raw_text: str) -> TweetEvaluation:
    """
    Robust parser:
      1) Try LangChain PydanticOutputParser.parse first (fast path).
      2) Find ```json``` fenced block or first {...} block.
      3) Clean common repr/escaping artifacts and unescape.
      4) json.loads -> TweetEvaluation.model_validate / parse_obj.
    Raises ValueError with a helpful message on failure.
    """
    raw_text = (raw_text or "").strip()
    logger.debug("Attempting to parse raw_text. Head: %s", repr(raw_text[:600]))

    # 1) Try langchain parser first (may succeed if model obeyed instructions exactly)
    try:
        parsed = evaluation_parser.parse(raw_text)
        logger.debug("Parsed with evaluation_parser successfully.")
        return parsed
    except Exception as e:
        logger.debug("evaluation_parser.parse failed (expected sometimes): %s", e)

    # 2) Extract JSON inside ```json``` fences first
    json_like = None
    m = re.search(r"```json\s*([\s\S]*?)```", raw_text, re.IGNORECASE)
    if m:
        json_like = m.group(1)
    else:
        # fallback to first {...} block
        m2 = re.search(r"(\{[\s\S]*\})", raw_text)
        if m2:
            json_like = m2.group(1)

    if not json_like:
        logger.debug("No JSON-like substring found in model output. Full output head:\n%s", raw_text[:1000])
        raise ValueError("Failed to parse model output into TweetEvaluation — no JSON-like block found. Raw head:\n" + raw_text[:2000])

    # 3) Clean the JSON-like string
    clean = json_like.strip()

    # remove bounding single/double quotes if the JSON was double-quoted as a single string
    if (clean.startswith("'") and clean.endswith("'")) or (clean.startswith('"') and clean.endswith('"')):
        clean = clean[1:-1].strip()

    # Normalize newlines and unescape common sequences
    clean = clean.replace('\r\n', '\n').replace('\r', '\n')
    clean = clean.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
    # replace smart quotes
    clean = clean.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")

    logger.debug("Cleaned JSON-like head:\n%s", clean[:800])

    # 4) Parse JSON
    try:
        parsed_json = json.loads(clean)
    except Exception as json_err:
        logger.debug("json.loads failed: %s", json_err)
        # permissive fallback: try a simple heuristic to fix single-quoted keys -> double-quoted keys
        try:
            permissive = re.sub(r"(?m)^\s*'([^']+)'\s*:", r'"\1":', clean)  # convert 'key': -> "key":
            parsed_json = json.loads(permissive)
            logger.debug("Permissive json.loads succeeded.")
        except Exception as perm_err:
            logger.debug("Permissive json.loads also failed: %s", perm_err)
            raise ValueError("Failed to json.loads cleaned JSON-like block. Clean head:\n" + clean[:2000])

    # 5) Validate with Pydantic (support v2 and v1)
    try:
        if hasattr(TweetEvaluation, "model_validate"):
            # pydantic v2
            validated = TweetEvaluation.model_validate(parsed_json)
        elif hasattr(TweetEvaluation, "model_validate_json") and isinstance(parsed_json, str):
            validated = TweetEvaluation.model_validate_json(parsed_json)
        else:
            # pydantic v1 fallback
            validated = TweetEvaluation.parse_obj(parsed_json)
        return validated
    except (PydanticValidationError, ValidationError) as val_err:
        logger.debug("Pydantic validation failed: %s", val_err)
        raise ValueError("TweetEvaluation validation failed on parsed JSON. Parsed JSON:\n" + json.dumps(parsed_json, indent=2)[:2000])


# ---------- TypedDict for state (plain types only) ----------
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iterations: int
    max_iterations: int
    tweet_history: List[str]
    feedback_history: List[str]


# ---------- Workflow functions ----------
def generate_tweet(state: TweetState) -> TweetState:
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}". 

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
""")
    ]

    # generate_with_ollama will convert messages -> string
    raw = generate_with_ollama(OLLAMA_MODEL, messages)
    tweet_text = (raw or "").strip()

    state['tweet'] = tweet_text
    state.setdefault('tweet_history', []).append(tweet_text)
    return state


def evaluate_tweet(state: TweetState) -> TweetState:
    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
        HumanMessage(content=f"""
{format_instructions}

Evaluate the following tweet:

Tweet: "{state['tweet']}"

Use the criteria below to evaluate the tweet:

1. Originality – Is this fresh, or have you seen it a hundred times before?  
2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
3. Punchiness – Is it short, sharp, and scroll-stopping?  
4. Virality Potential – Would people retweet or share it?  
5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- It ends with generic, throwaway, or deflating lines

Respond ONLY in JSON with keys: evaluation (approved|needs_improvement), feedback (single paragraph).
""")
    ]

    raw = generate_with_ollama(OLLAMA_MODEL, messages)
    parsed = robust_parse_evaluation(raw)

    state['evaluation'] = parsed.evaluation
    state['feedback'] = parsed.feedback
    state.setdefault('feedback_history', []).append(parsed.feedback)
    return state


def optimize_tweet(state: TweetState) -> TweetState:
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]

    raw = generate_with_ollama(OLLAMA_MODEL, messages)
    new_tweet = (raw or "").strip()
    state['tweet'] = new_tweet
    state['iterations'] = state.get('iterations', 0) + 1
    state.setdefault('tweet_history', []).append(new_tweet)
    return state


def route_evaluation(state: TweetState) -> str:
    if state.get('evaluation') == 'approved' or state.get('iterations', 0) >= state.get('max_iterations', 3):
        return 'approved'
    else:
        return 'needs_improvement'


# ---------- Graph setup ----------
graph = StateGraph(TweetState)

graph.add_node('generate', generate_tweet)
graph.add_node('evaluate', evaluate_tweet)
graph.add_node('optimize', optimize_tweet)

graph.add_edge(START, 'generate')
graph.add_edge('generate', 'evaluate')

graph.add_conditional_edges('evaluate', route_evaluation, {'approved': END, 'needs_improvement': 'optimize'})
graph.add_edge('optimize', 'evaluate')

workflow = graph.compile()


# ---------- Simple main to test the flow ----------
def main():
    print("\n=== Starting Tweet Generation Workflow ===")

    initial_state: TweetState = {
        "topic": "AI replacing jobs",
        "tweet": "",
        "evaluation": "needs_improvement",
        "feedback": "",
        "iterations": 0,
        "max_iterations": 3,
        "tweet_history": [],
        "feedback_history": []
    }

    try:
        final_state = workflow.invoke(initial_state)

        print("\n=== Workflow Complete ===")
        print(f"Topic: {final_state['topic']}")
        print(f"Final Evaluation: {final_state['evaluation']}")
        print(f"Final Tweet:\n{final_state['tweet']}")
        print("\nFeedback History:")
        for i, fb in enumerate(final_state.get('feedback_history', []), 1):
            print(f"{i}. {fb}\n")
        print("\nTweet History:")
        for i, tw in enumerate(final_state.get('tweet_history', []), 1):
            print(f"{i}. {tw}\n")

    except Exception as e:
        logger.error("Workflow execution failed: %s", e)
        raise


if __name__ == "__main__":
    main()
