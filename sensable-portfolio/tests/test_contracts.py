import json
from sensable_portfolio.contracts import (
    Intervention, AgentInfo, MoodFrame, AgentActionFrame, Recommendation,
)


def test_intervention_validates_and_satisfies_protocol():
    i = Intervention(
        decision_id="d1", arm_id="breath_coach.v1", ts=1000.0,
        action_type="breath", title="Box breath, 90 s",
        body="Inhale 4s, hold 4s, exhale 4s, hold 4s.",
        duration_s=90.0, intensity="low", rationale="rising stress slope",
    )
    assert i.schema_version == 1
    rec: Recommendation = i
    assert rec.decision_id == "d1"
    assert rec.arm_id == "breath_coach.v1"


def test_mood_frame_shape_matches_spec_a():
    f = MoodFrame(
        v=1, type="mood",
        vector={"focus": -0.11, "stress": -1.91, "valence": -1.96, "arousal": 0.75,
                "joy": 0.0, "calm": 0.0, "excitement": 0.05, "neutral": 0.48},
        ts=1778364427578,
    )
    payload = json.loads(f.model_dump_json())
    assert payload["v"] == 1
    assert payload["type"] == "mood"
    assert set(payload["vector"]) == {
        "focus","stress","valence","arousal","joy","calm","excitement","neutral",
    }
    assert isinstance(payload["ts"], int)


def test_agent_action_frame_carries_agent_and_intervention():
    f = AgentActionFrame(
        v=1, type="agent_action", ts=1778364427578,
        decision_id="d1",
        agent=AgentInfo(id="breath_coach.v1", persona="breath_coach", model="m"),
        intervention=Intervention(
            decision_id="d1", arm_id="breath_coach.v1", ts=1.0, action_type="breath",
            title="t", body="b", duration_s=10.0, intensity="low", rationale="r",
        ),
        signals_at_decision={"focus": 0.0, "stress": 0.0, "valence": 0.0, "arousal": 0.0,
                             "joy": 0.0, "calm": 0.0, "excitement": 0.0, "neutral": 0.0},
    )
    payload = json.loads(f.model_dump_json())
    assert payload["type"] == "agent_action"
    assert payload["agent"]["persona"] == "breath_coach"
    assert payload["intervention"]["action_type"] == "breath"
