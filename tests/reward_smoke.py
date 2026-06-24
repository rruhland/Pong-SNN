import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snn_backend import PongSNN


def world(ball_x=640, ball_y=180, ball_vx=200, right_y=170, left_score=0):
    return {
        "settings": {
            "width": 800,
            "height": 450,
            "paddleHeight": 80,
            "ballSize": 10,
        },
        "score": {"left": left_score, "right": 0},
        "ball": {"x": ball_x, "y": ball_y, "vx": ball_vx, "vy": 0},
        "paddles": {"rightY": right_y},
    }


def component_after(model, frame, bars, winner):
    value = model._reward_from_pong(frame, output_bars=bars, winner_index=winner)
    return value, model.reward_state["components"]


def test_confident_smooth_movement_gets_shaped_without_game_state_terms():
    model = PongSNN(seed=7)

    component_after(model, world(right_y=170), [0.95, 0.1, 0.08], 0)
    _, components = component_after(model, world(right_y=164), [0.96, 0.08, 0.05], 0)

    assert components["outputConfidence"] > 0
    assert components["chosenMovementPenalty"] < 0
    assert components["smoothMotionReward"] > 0
    assert components["directionChangePenalty"] == 0

    _, components = component_after(model, world(right_y=172), [0.08, 0.96, 0.05], 1)
    assert components["directionChangePenalty"] < 0


def test_ambiguous_outputs_are_penalized_and_quiet_hold_can_be_rewarded():
    model = PongSNN(seed=8)

    _, ambiguous = component_after(model, world(), [0.82, 0.78, 0.74], 0)
    assert ambiguous["outputConfidence"] < 0
    assert ambiguous["competingOutputPenalty"] < 0
    assert ambiguous["movementOutputPenalty"] < 0

    model.reward_function.reset()
    _, quiet = component_after(model, world(), [0.04, 0.05, 1.0], 2)
    assert quiet["quietOutputReward"] > 0
    assert quiet["chosenMovementPenalty"] == 0


def test_hit_and_miss_still_dominate_dense_shaping_terms():
    hit_model = PongSNN(seed=9)
    component_after(hit_model, world(ball_x=650, ball_vx=200), [0.04, 0.05, 1.0], 2)
    hit_value, hit_components = component_after(hit_model, world(ball_x=660, ball_vx=-200), [0.04, 0.05, 1.0], 2)
    assert hit_components["hit"] == 1.0
    assert hit_value > 0.9

    miss_model = PongSNN(seed=10)
    component_after(miss_model, world(left_score=0), [0.04, 0.05, 1.0], 2)
    miss_value, miss_components = component_after(miss_model, world(left_score=1), [0.04, 0.05, 1.0], 2)
    assert miss_components["miss"] == -1.0
    assert miss_value < -0.9


if __name__ == "__main__":
    test_confident_smooth_movement_gets_shaped_without_game_state_terms()
    test_ambiguous_outputs_are_penalized_and_quiet_hold_can_be_rewarded()
    test_hit_and_miss_still_dominate_dense_shaping_terms()
    print("reward smoke ok")
