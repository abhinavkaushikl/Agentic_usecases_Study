# review_workflow_ollama.py
import logging
import os
import re
import time
from dotenv import load_dotenv
from typing import TypedDict, Literal, Optional, Dict, Any, Annotated, List
from pydantic import BaseModel, Field, ValidationError
from langchain.output_parsers import PydanticOutputParser
import ollama
from langgraph.graph import StateGraph, START, END
from langgraph.errors import InvalidUpdateError
from langchain_core.messages import SystemMessage, HumanMessage
import operator

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


# ---------- Ollama call with retries ----------
def generate_with_ollama(model_name: str, prompt: Any, max_retries: int = OLLAMA_MAX_RETRIES,
                         retry_delay: float = OLLAMA_RETRY_DELAY, stream: bool = False) -> str:
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


# ---------- Helper: robust parse for evaluation ----------
def robust_parse_evaluation(raw_text: str) -> TweetEvaluation:
    """
    Try the LangChain PydanticOutputParser first; if that fails try to extract a JSON blob or fenced block.
    """
    raw_text = (raw_text or "").strip()
    logger.debug("Attempting to parse raw_text. Head: %s", repr(raw_text[:300]))

    # 1) LangChain parser (expects model to follow format_instructions)
    try:
        return evaluation_parser.parse(raw_text)
    except Exception as e:
        logger.debug("PydanticOutputParser.parse failed: %s", e)

    # 2) Try to extract JSON-like object {...}
    json_like = None
    m = re.search(r"(\{[\s\S]*\})", raw_text)
    if m:
        json_like = m.group(1)
    else:
        # 3) Try code fence block (```json ... ``` or ``` ... ```)
        m2 = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
        if m2:
            json_like = m2.group(1).strip()

    if json_like:
        logger.debug("Found JSON-like substring, attempting TweetEvaluation.parse_raw")
        try:
            return TweetEvaluation.parse_raw(json_like)
        except ValidationError as ve:
            logger.debug("TweetEvaluation.parse_raw validation error: %s", ve)
        except Exception as ex:
            logger.debug("TweetEvaluation.parse_raw other error: %s", ex)

    # Nothing worked
    raise ValueError("Failed to parse model output into TweetEvaluation. Raw head:\n" + raw_text[:2000])


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
    # prompt (use plain text messages as you had)
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

    # Use the raw generator (no pydantic parsing)
    raw = generate_with_ollama(OLLAMA_MODEL, messages)
    tweet_text = (raw or "").strip()

    # update state
    state['tweet'] = tweet_text
    state.setdefault('tweet_history', []).append(tweet_text)
    return state


def evaluate_tweet(state: TweetState) -> TweetState:
    # prompt for structured evaluation (format_instructions can be included in your prompt)
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
    # increment iteration count
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

def main():
    """
    Test the full Tweet generation-evaluation-optimization workflow.
    """
    print("\n=== Starting Tweet Generation Workflow ===")

    # Initialize the state
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
        # Run the LangGraph workflow
        final_state = workflow.invoke(initial_state)

        print("\n=== Workflow Complete ===")
        print(f"Topic: {final_state['topic']}")
        print(f"Final Evaluation: {final_state['evaluation']}")
        print(f"Final Tweet:\n{final_state['tweet']}")
        print("\nFeedback History:")
        for i, fb in enumerate(final_state['feedback_history'], 1):
            print(f"{i}. {fb}\n")
        print("\nTweet History:")
        for i, tw in enumerate(final_state['tweet_history'], 1):
            print(f"{i}. {tw}\n")

    except Exception as e:
        logger.error("Workflow execution failed: %s", e)
        raise


if __name__ == "__main__":
    main()
