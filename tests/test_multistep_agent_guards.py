import unittest

from agent import multistep_agent as agent


class _NoCallClient:
    def simple_chat(self, *args, **kwargs):
        raise AssertionError("model should not be called when a supported candidate is ready")


def _q5_like_state() -> dict:
    question = "What is the first and last name of the cover designer of this report?"
    question_plan = {
        "answer_type": "person",
        "verification_query": "cover designer Leadership Strategies Graphic Design Malaria Consortium Ogilvy",
        "keywords": ["cover designer", "Graphic Design", "Malaria Consortium"],
    }
    evidence = (
        "Cristina Ortiz Graphic Designer. Graphic Designer Consultant World Health "
        "Organization Jul 2010 - Jan 2019. Editorial Designer Malaria Consortium. "
        "Graphic Designer Ogilvy & Mather. Yale Publishing Course - Leadership "
        "Strategies in Book Publishing. Bachelor's Degree in Graphic Design. "
        "Bachelor of Biology appears in an unrelated WHO biography passage."
    )
    state = agent._init_state(question, question_plan=question_plan)
    state.update(
        {
            "opened_docids": ["72340", "11373"],
            "seen_docids": ["72340", "11373"],
            "opened_passages": [evidence],
            "document_cache": {"72340": evidence, "11373": evidence},
            "candidate_answers": ["Global Fund", "Bachelor of Biology"],
            "verification_results": [
            {
                "supported": True,
                "support_score": 0.9,
                "missing_piece": "",
                "candidate_answer": "Cristina Ortiz",
            },
            {
                "supported": False,
                "support_score": 0.735,
                "missing_piece": "Missing evidence for `cover designer` relation.",
                "candidate_answer": "Bachelor of Biology",
            },
            ],
            "last_action": "verify_claim",
        }
    )
    return state


class MultistepAgentGuardTests(unittest.TestCase):
    def test_person_type_rejects_degree_phrase(self) -> None:
        self.assertTrue(agent._candidate_looks_wrong_type("Bachelor of Biology", "person"))

    def test_decide_next_action_finishes_after_supported_candidate_at_round_three(self) -> None:
        action, _ = agent._decide_next_action(
            question="What is the first and last name of the cover designer of this report?",
            state=_q5_like_state(),
            client=_NoCallClient(),
            model_name="qwen_auto",
            max_rounds=7,
            round_id=3,
            decision_max_tokens=128,
            recent_observation="Verifier found Cristina Ortiz supported.",
        )

        self.assertEqual(action["action"], "finish")
        self.assertEqual(action["answer_hint"], "Cristina Ortiz")

    def test_final_guard_keeps_supported_candidate_over_unsupported_high_score_candidate(self) -> None:
        guarded = agent._apply_final_answer_guard(
            predicted_answer="Cristina Ortiz",
            state=_q5_like_state(),
        )

        self.assertEqual(guarded, "Cristina Ortiz")


if __name__ == "__main__":
    unittest.main()
