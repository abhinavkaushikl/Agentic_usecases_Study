import logging
from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator
from langchain.output_parsers import PydanticOutputParser
import ollama
from langgraph.graph import StateGraph, START, END

# Configure logging
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

# --- Pydantic schema for the structured output ---
class EvaluationSchema(BaseModel):
    feedback: str = Field(description='Detailed feedback for an essay')
    score: int = Field(description='score out of 10', ge=0, le=10)

# Build LangChain parser for the schema
parser = PydanticOutputParser(pydantic_object=EvaluationSchema)
format_instructions = parser.get_format_instructions()

# Configure Ollama client (optional host override)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=OLLAMA_HOST)

# Helper that asks Ollama and parses structured output
def run_structured_with_ollama(model_name: str, prompt: str) -> EvaluationSchema:
    try:
        full_prompt = f"{prompt}\n\n{format_instructions}"
        res = client.generate(model=model_name, prompt=full_prompt, stream=False)
        raw_text = res.get("response") or res.get("text") or str(res)
        return parser.parse(raw_text)
    except Exception as e:
        logging.error(f"Error in run_structured_with_ollama: {e}")
        raise

# Example essay
essay = '''
India, officially known as the Republic of India, is a vast and diverse nation located in South Asia.
It is the seventh-largest country in the world by land area and the most populous, with over 1.4 billion people.
India’s civilization dates back thousands of years, making it one of the oldest continuous cultures in human history.
From the Indus Valley Civilization to the modern era of technology and innovation,
India has evolved into a vibrant democracy known for its unity in diversity.
'''

# State type
class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float

# Pick your Ollama model name
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

# Evaluation stage 1
def evaluate_language(state: UPSCState):
    try:
        prompt = f"Evaluate the language quality of the following essay and provide feedback and assign score out of 10:\n\n{state['essay']}"
        out = run_structured_with_ollama(OLLAMA_MODEL, prompt)
        return {
            'language_feedback': out.feedback,
            'language_score': out.score
        }
    except Exception as e:
        logging.error(f"Error during evaluate_language: {e}")
        raise

# Evaluation stage 2 with error handling and logging
def evaluate_analysis(state: UPSCState):
    try:
        prompt = f"Evaluate the depth of analysis of the following essay and provide feedback and assign score out of 10:\n\n{state['essay']}"
        out = run_structured_with_ollama(OLLAMA_MODEL, prompt)
        result = {
            'analysis_feedback': out.feedback,
            'analysis_score': out.score
        }
        logging.debug(f"evaluate_analysis result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error during evaluate_analysis: {e}")
        raise

# Evaluation stage 3
def evaluate_thought(state: UPSCState):
    prompt = f"Evaluate the clarity of thought of the following essay and provide feedback and assign score out of 10:\n\n{state['essay']}"
    out = run_structured_with_ollama(OLLAMA_MODEL, prompt)
    return {
        'clarity_feedback': out.feedback,
        'thought_score': out.score
    }
    logging.debug(f"evaluate_analysis result: {result}")
 


# Final summary with state inspection
def final_evaluation(state: UPSCState):
    logging.debug(f"State before final_evaluation: {state}")
    summary_prompt = (
        "Based on the following feedbacks create a summarized feedback and give an overall recommendation:\n\n"
        f"language feedback - {state.get('language_feedback', 'N/A')}\n\n"
        f"analysis feedback - {state.get('analysis_feedback', 'N/A')}\n\n"
        f"clarity feedback - {state.get('clarity_feedback', 'N/A')}\n\n"
    )
    res = client.generate(model=OLLAMA_MODEL, prompt=summary_prompt, stream=False)
    print(res)
    overall_feedback = res.get("response") or res.get("text") or str(res)
    avg_score = (state.get('language_score', 0) + state.get('analysis_score', 0) + state.get('thought_score', 0)) / 3.0
    return {
        'overall_score': avg_score,
        'overall_feedback': overall_feedback
    }

# Build the StateGraph
graph = StateGraph(UPSCState)
graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('final_evaluation', final_evaluation)

# Add edges to ensure correct execution order
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_thought')
graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thought', 'final_evaluation')
graph.add_edge('final_evaluation', END)

workflow = graph.compile()

# Run the workflow
initial_state = {'essay': essay}
output = workflow.invoke(initial_state)
print(output)