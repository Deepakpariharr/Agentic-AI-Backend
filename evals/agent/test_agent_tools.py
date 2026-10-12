"""Single-turn evals: does the agent pick the right tool (or none), and is the
answer correct / grounded in what the tool returned?

Tool names are the ones the agent actually sees: calculate / search_jobs /
search_web, all served by our MCP server (mcp_server/).
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall

import eval_metrics as m
from config import RAPIDAPI_KEY, TAVILY_API_KEY
from harness import new_user_id, run_agent

# ── Maths: should go through `calculate`, and get the number right ───────────
CALCULATOR_CASES = [
    pytest.param("What is 245 * 17?", "4165", id="multiply"),
    pytest.param("What is (120 + 80) / 4?", "50", id="parentheses"),
    pytest.param("What is 1250 minus 387.5?", "862.5", id="decimal-subtract"),
    pytest.param(
        "I bought 3 notebooks at 45 rupees each and a pen for 20 rupees. How much did I spend?",
        "155 rupees (3 × 45 + 20)",
        id="word-problem",
    ),
    pytest.param(
        "A monthly salary of 48000 gets a 12% raise. What is the new monthly salary?",
        "53760 (48000 × 1.12)",
        id="percentage",
    ),
]


@pytest.mark.tools
@pytest.mark.parametrize("prompt, expected", CALCULATOR_CASES)
def test_calculator(prompt, expected):
    run = run_agent(prompt, user_id=new_user_id())
    test_case = LLMTestCase(
        input=prompt,
        actual_output=run.output,
        expected_output=expected,
        tools_called=run.tools_called,
        expected_tools=[ToolCall(name="calculate")],
    )
    assert_test(test_case, [m.tool_correctness(), m.correctness()])


# ── No tool needed: the system prompt says not to call tools needlessly ─────
NO_TOOL_CASES = [
    pytest.param("Hi! What kinds of things can you help me with?", id="greeting"),
    pytest.param("Explain what a REST API is in two sentences.", id="general-knowledge"),
    pytest.param("Rewrite this more politely: 'send me the report now'", id="rewrite"),
    pytest.param("What is the capital of France?", id="stable-fact"),
]


@pytest.mark.tools
@pytest.mark.parametrize("prompt", NO_TOOL_CASES)
def test_answers_without_tools(prompt):
    run = run_agent(prompt, user_id=new_user_id())
    test_case = LLMTestCase(
        input=prompt,
        actual_output=run.output,
        tools_called=run.tools_called,
        expected_tools=[],
    )
    assert_test(test_case, [m.tool_correctness(), m.answer_relevancy()])


# ── Live data: job search (RapidAPI) and web search (Tavily) ────────────────
# No fixed expected answer — results change daily — so these check the right
# tool ran and that the answer only reports what the tool actually returned.
JOB_CASES = [
    pytest.param("Find me Python developer jobs in Bangalore posted this week.", id="python-bangalore"),
    pytest.param("Are there any remote data analyst openings for freshers?", id="remote-analyst"),
]


@pytest.mark.tools
@pytest.mark.external
@pytest.mark.skipif(not RAPIDAPI_KEY, reason="RAPIDAPI_KEY not set")
@pytest.mark.parametrize("prompt", JOB_CASES)
def test_job_search(prompt):
    run = run_agent(prompt, user_id=new_user_id())
    test_case = LLMTestCase(
        input=prompt,
        actual_output=run.output,
        tools_called=run.tools_called,
        expected_tools=[ToolCall(name="search_jobs")],
    )
    assert_test(test_case, [m.tool_correctness(), m.grounded_in_tools()])


WEB_CASES = [
    pytest.param("What is the latest stable version of Python right now?", id="python-version"),
    pytest.param("What are today's top technology news headlines?", id="tech-news"),
]


@pytest.mark.tools
@pytest.mark.external
@pytest.mark.skipif(not TAVILY_API_KEY, reason="TAVILY_API_KEY not set")
@pytest.mark.parametrize("prompt", WEB_CASES)
def test_web_search(prompt):
    run = run_agent(prompt, user_id=new_user_id())
    test_case = LLMTestCase(
        input=prompt,
        actual_output=run.output,
        tools_called=run.tools_called,
        expected_tools=[ToolCall(name="search_web")],
    )
    assert_test(test_case, [m.tool_correctness(), m.grounded_in_tools()])
