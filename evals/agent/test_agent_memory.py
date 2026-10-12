"""Multi-turn evals: the agent remembers within a user's thread (MemorySaver,
keyed by user_id) and never leaks one user's history into another's.

Keep conversations short: agent_service trims history to the last
MAX_HISTORY_MESSAGES (5) messages, so facts from much earlier are dropped by
design.
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import ConversationalTestCase

import eval_metrics as m
from harness import new_user_id, run_agent, run_conversation


@pytest.mark.memory
def test_remembers_facts_within_a_thread():
    turns = run_conversation(
        [
            "Hi, my name is Rahul and my favourite colour is red.",
            "What is my favourite colour?",
        ],
        user_id=new_user_id(),
    )
    final = turns[-1].content.lower()
    assert "red" in final, f"Forgot the colour: {turns[-1].content!r}"

    test_case = ConversationalTestCase(
        turns=turns,
        chatbot_role="A helpful assistant that remembers what the user told it earlier in the chat.",
    )
    assert_test(test_case, [m.knowledge_retention()])


@pytest.mark.memory
def test_follow_up_uses_previous_result():
    turns = run_conversation(
        ["What is 12 * 12?", "Now divide that by 4."],
        user_id=new_user_id(),
    )
    assert "36" in turns[-1].content, f"Lost the previous result: {turns[-1].content!r}"


@pytest.mark.memory
def test_users_do_not_share_history():
    """thread_id = user_id is the only thing separating users' chats; a bug
    there would hand one user another user's conversation."""
    run_agent("Remember this: my secret project codename is BLUEHERON.", user_id=new_user_id())

    other = run_agent("What is my secret project codename?", user_id=new_user_id())
    assert "blueheron" not in other.output.lower(), (
        f"Another user's history leaked: {other.output!r}"
    )
