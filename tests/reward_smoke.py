import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snn_backend import PongSNN


def world(tick=0, reset_token=0, left_score=0, right_score=0):
    return {
        "resetToken": reset_token,
        "tick": tick,
        "authoritativeTick": tick,
        "score": {"left": left_score, "right": right_score},
    }


def reward_after(model, frame, winner):
    value = model._reward_from_pong(frame, winner_index=winner)
    return value, model.reward_state["components"], model.reward_state["metrics"]


def test_movement_costs_reward_and_hold_is_neutral_before_survival():
    model = PongSNN(seed=7)

    hold_value, hold_components, _ = reward_after(model, world(tick=1), 2)
    move_value, move_components, _ = reward_after(model, world(tick=2), 0)

    assert hold_value == 0.0
    assert hold_components["movement"] == 0.0
    assert move_components["movement"] < 0.0
    assert move_value < 0.0


def test_survival_grows_slowly_with_episode_time():
    model = PongSNN(seed=8)

    reward_after(model, world(tick=1), 2)
    early_value, early_components, early_metrics = reward_after(model, world(tick=2), 2)
    later_value, later_components, later_metrics = reward_after(model, world(tick=602), 2)

    assert early_components["survival"] > 0.0
    assert later_components["survival"] > early_components["survival"]
    assert later_value > early_value
    assert later_metrics["episodeSeconds"] > early_metrics["episodeSeconds"]


def test_score_events_dominate_dense_rewards_and_reset_episode_timer():
    model = PongSNN(seed=9)

    reward_after(model, world(tick=1), 2)
    right_value, right_components, right_metrics = reward_after(model, world(tick=600, right_score=1), 2)

    assert right_components["rightScore"] > 3.0
    assert right_value > 3.0
    assert right_metrics["episodeSeconds"] > 0.0
    assert model.reward_function.episode_ticks == 0

    opponent_value, opponent_components, _ = reward_after(model, world(tick=700, right_score=1, left_score=1), 2)
    assert opponent_components["opponentScore"] < -1.0
    assert opponent_value < -1.0


if __name__ == "__main__":
    test_movement_costs_reward_and_hold_is_neutral_before_survival()
    test_survival_grows_slowly_with_episode_time()
    test_score_events_dominate_dense_rewards_and_reset_episode_timer()
    print("reward smoke ok")
