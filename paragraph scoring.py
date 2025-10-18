#!/usr/bin/env python3
import logging
import os
import re
import time
from dotenv import load_dotenv
from typing import TypedDict
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

# ---------- Output schema + parser ----------
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for an essay")
    score: int = Field(description="score out of 10", ge=0, le=10)

parser = PydanticOutputParser(pydantic_object=EvaluationSchema)
format_instructions = parser.get_format_instructions()

# ---------- Ollama client ----------
client = ollama.Client(host=OLLAMA_HOST)

# ---------- Helper: robust parse ----------
def robust_parse_evaluation(raw_text: str) -> EvaluationSchema:
    """
    Try the LangChain PydanticOutputParser first; if that fails try to extract a JSON blob or fenced block.
    """
    raw_text = (raw_text or "").strip()
    logger.debug("Attempting to parse raw_text. Head: %s", repr(raw_text[:300]))

    # 1) LangChain parser (expects model to follow format_instructions)
    try:
        return parser.parse(raw_text)
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
        logger.debug("Found JSON-like substring, attempting EvaluationSchema.parse_raw")
        try:
            return EvaluationSchema.parse_raw(json_like)
        except ValidationError as ve:
            logger.debug("EvaluationSchema.parse_raw validation error: %s", ve)
        except Exception as ex:
            logger.debug("EvaluationSchema.parse_raw other error: %s", ex)

    # Nothing worked
    raise ValueError("Failed to parse model output into EvaluationSchema. Raw head:\n" + raw_text[:2000])

# ---------- Helper: call Ollama with retries ----------
def generate_with_ollama(model_name: str, prompt: str, max_retries: int = OLLAMA_MAX_RETRIES, retry_delay: float = OLLAMA_RETRY_DELAY) -> str:
    last_err = None
    full_prompt = f"{prompt}\n\n{format_instructions}"
    for attempt in range(1, max_retries + 2):  # initial attempt + retries
        try:
            logger.debug("Calling Ollama (attempt %d) model=%s", attempt, model_name)
            res = client.generate(model=model_name, prompt=full_prompt, stream=False)
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
    raise RuntimeError(f"Ollama generation failed after {max_retries+1} attempts. Last error: {last_err}")

# ---------- Example essay ----------
essay = """
India, officially known as the Republic of India, is a vast and diverse nation located in South Asia.
It is the seventh-largest country in the world by land area and the most populous, with over 1.4 billion people.
India’s civilization dates back thousands of years, making it one of the oldest continuous cultures in human history.
From the Indus Valley Civilization to the modern era of technology and innovation,
India has evolved into a vibrant democracy known for its unity in diversity.
"""

# ---------- TypedDict for state (optional keys allowed) ----------
class UPSCState(TypedDict, total=False):
    essay: str
    language_feedback: str
    language_score: int
    analysis_feedback: str
    analysis_score: int
    clarity_feedback: str
    thought_score: int
    overall_feedback: str
    overall_score: float
    avg_score: float

# ---------- Evaluation helper that uses the robust parse ----------
def run_structured_with_ollama(model_name: str, prompt: str) -> EvaluationSchema:
    raw = generate_with_ollama(model_name, prompt)
    try:
        parsed = robust_parse_evaluation(raw)
        return parsed
    except Exception as e:
        logger.error("Failed to parse model output into EvaluationSchema: %s\nRaw output head: %s", e, (raw or "")[:1000])
        raise

# ---------- Evaluation steps (parallel-safe: return only produced keys) ----------
def evaluate_language(state: UPSCState) -> dict:
    try:
        prompt = f"Evaluate the language quality of the following essay and provide feedback and assign score out of 10:\n\n{state['essay']}"
        out = run_structured_with_ollama(OLLAMA_MODEL, prompt)
        logger.debug("evaluate_language: score=%s", out.score)
        # Return only node-specific keys (do NOT return 'essay' or the whole state)
        return {"language_feedback": out.feedback, "language_score": out.score}
    except Exception:
        logger.exception("evaluate_language failed")
        raise

def evaluate_analysis(state: UPSCState) -> dict:
    try:
        prompt = f"Evaluate the depth of analysis of the following essay and provide feedback and assign score out of 10:\n\n{state['essay']}"
        out = run_structured_with_ollama(OLLAMA_MODEL, prompt)
        logger.debug("evaluate_analysis: score=%s", out.score)
        return {"analysis_feedback": out.feedback, "analysis_score": out.score}
    except Exception:
        logger.exception("evaluate_analysis failed")
        raise

def evaluate_thought(state: UPSCState) -> dict:
    try:
        prompt = f"Evaluate the clarity of thought of the following essay and provide feedback and assign score out of 10:\n\n{state['essay']}"
        out = run_structured_with_ollama(OLLAMA_MODEL, prompt)
        logger.debug("evaluate_thought: score=%s", out.score)
        # Note: use 'clarity_feedback' key to match final_evaluation's reads
        return {"clarity_feedback": out.feedback, "thought_score": out.score}
    except Exception:
        logger.exception("evaluate_thought failed")
        raise

# ---------- Final evaluation reads merged keys ----------
def final_evaluation(state: UPSCState) -> UPSCState:
    try:
        summary_prompt = (
            "Based on the following feedbacks create a summarized feedback and give an overall recommendation:\n\n"
            f"language feedback - {state.get('language_feedback', 'N/A')}\n\n"
            f"analysis feedback - {state.get('analysis_feedback', 'N/A')}\n\n"
            f"clarity feedback - {state.get('clarity_feedback', 'N/A')}\n\n"
            "Provide a concise overall recommendation and mention three clear ways to improve the essay."
        )
        raw_summary = generate_with_ollama(OLLAMA_MODEL, summary_prompt)
        overall_feedback = (raw_summary or "").strip()

        # compute average (defaults to 0 if a score is missing)
        language_score = float(state.get("language_score", 0))
        analysis_score = float(state.get("analysis_score", 0))
        thought_score = float(state.get("thought_score", 0))
        avg_score = (language_score + analysis_score + thought_score) / 3.0

        # write merged results into final state object and return it
        state["overall_feedback"] = overall_feedback
        state["overall_score"] = avg_score
        state["avg_score"] = avg_score

        logger.debug(
            "final_evaluation computed avg=%s (lang=%s, analysis=%s, thought=%s)",
            avg_score, language_score, analysis_score, thought_score
        )
        return state
    except Exception:
        logger.exception("final_evaluation failed")
        raise

# ---------- Build graph ----------
graph = StateGraph(UPSCState)
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Parallel eval nodes from START, final_evaluation depends on all three
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()

# ---------- Run the workflow ----------
if __name__ == "__main__":
    initial_state: UPSCState = {"essay": essay}
    logger.info("Invoking workflow with essay (length=%d)", len(essay))
    try:
        output_state = workflow.invoke(initial_state)
    except InvalidUpdateError as ie:
        logger.exception("Graph merge error (InvalidUpdateError): %s", ie)
        raise
    except Exception:
        logger.exception("Workflow invocation failed")
        raise

    # Print results
    print("\n=== Final state ===")
    print("Language score:", output_state.get("language_score"))
    print("Analysis score:", output_state.get("analysis_score"))
    print("Thought score:", output_state.get("thought_score"))
    print("Average score:", output_state.get("avg_score"))
    print("\nOverall feedback:\n", output_state.get("overall_feedback"))
