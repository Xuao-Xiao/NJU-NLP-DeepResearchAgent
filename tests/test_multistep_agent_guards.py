import unittest

from agent import multistep_agent as agent


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

    def test_final_guard_keeps_supported_candidate_over_unsupported_high_score_candidate(self) -> None:
        guarded = agent._apply_final_answer_guard(
            predicted_answer="Cristina Ortiz",
            state=_q5_like_state(),
        )

        self.assertEqual(guarded, "Cristina Ortiz")

    def test_final_guard_extracts_latin_club_name_from_opened_evidence(self) -> None:
        question = (
            "The club's name has four syllables and begins with \"B.\" "
            "The club opened on the West Coast offering Latin music seven nights a week. "
            "What is the name of the club?"
        )
        state = agent._init_state(
            question,
            question_plan={
                "answer_type": "title",
                "verification_query": "club latin music sound system begins with B",
            },
        )
        state.update(
            {
                "opened_docids": ["11992"],
                "document_cache": {
                    "11992": (
                        "Salsa News: Joe Cuba will be in the Los Angeles area. "
                        "Binochios in North Hollywood will be opening with top Latin salsa "
                        "orchestras from New York and Latin America."
                    )
                },
                "opened_passages": [],
                "confirmed_facts": [],
                "candidate_answers": ["BUSINESS OPPORTUNITIE ROCK CONCERT CORP"],
                "verification_results": [
                    {
                        "supported": True,
                        "support_score": 1.0,
                        "missing_piece": "",
                        "candidate_answer": "BUSINESS OPPORTUNITIE ROCK CONCERT CORP",
                    }
                ],
            }
        )

        guarded = agent._apply_final_answer_guard(
            predicted_answer="BUSINESS OPPORTUNITIE ROCK CONCERT CORP",
            state=state,
        )

        self.assertEqual(guarded, "Binochios")

    def test_final_guard_extracts_country_from_spent_two_years_relation(self) -> None:
        question = (
            "As a teenager, around 1950, Person A spent two years in a foreign country. "
            "What was the country?"
        )
        state = agent._init_state(
            question,
            question_plan={
                "answer_type": "place",
                "verification_query": "country foreign teenager spent two years",
            },
        )
        state.update(
            {
                "opened_docids": ["82044"],
                "document_cache": {
                    "82044": (
                        "Philip Glenn Whalen was born in Portland, Oregon. "
                        "Became a Zen Buddhist and spent two years in Japan. "
                        "His work includes several poetry collections."
                    )
                },
                "opened_passages": [],
                "confirmed_facts": [],
                "candidate_answers": ["India", "Japan"],
                "verification_results": [
                    {
                        "supported": True,
                        "support_score": 0.775,
                        "missing_piece": "",
                        "candidate_answer": "India",
                    }
                ],
            }
        )

        guarded = agent._apply_final_answer_guard(predicted_answer="India", state=state)

        self.assertEqual(guarded, "Japan")


if __name__ == "__main__":
    unittest.main()
