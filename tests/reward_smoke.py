import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snn_backend import PongSNN, SNN_ARCHITECTURE, _grid_index


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


def test_default_architecture_config_preserves_existing_shape():
    model = PongSNN(seed=10)

    hidden_size = SNN_ARCHITECTURE["hidden_neurons"]
    local_edges = SNN_ARCHITECTURE["recurrent_local_edges_per_neuron"]
    long_edges = SNN_ARCHITECTURE["recurrent_long_range_edges_per_neuron"]
    prediction_edges = (
        SNN_ARCHITECTURE["prediction_local_targets"]
        + SNN_ARCHITECTURE["prediction_medium_targets"]
        + SNN_ARCHITECTURE["prediction_long_targets"]
    )

    assert model.hidden_size == hidden_size
    assert model.hidden_w == SNN_ARCHITECTURE["hidden_grid_width"]
    assert model.excitatory_count == int(hidden_size * SNN_ARCHITECTURE["excitatory_fraction"])
    assert model.inhibitory_count == hidden_size - model.excitatory_count
    assert len(model.motor_hidden) == 3 * SNN_ARCHITECTURE["motor_hidden_targets_per_action"]
    assert len(model.recurrent) == hidden_size * (local_edges + long_edges)
    assert len(model.hidden_output) == hidden_size * SNN_ARCHITECTURE["output_targets_per_hidden"]
    assert len(model.hidden_prediction) == hidden_size * prediction_edges


def test_custom_architecture_config_controls_hidden_cloud_and_edges():
    model = PongSNN(
        seed=11,
        architecture={
            "input_grid_width": 16,
            "input_grid_height": 9,
            "hidden_neurons": 120,
            "hidden_grid_width": 12,
            "hidden_grid_height": 10,
            "excitatory_fraction": 0.25,
            "motor_hidden_targets_per_action": 7,
            "recurrent_local_edges_per_neuron": 3,
            "recurrent_long_range_edges_per_neuron": 2,
            "output_targets_per_hidden": 2,
            "prediction_local_targets": 4,
            "prediction_medium_targets": 2,
            "prediction_long_targets": 1,
        },
    )

    assert model.input_size == 16 * 9
    assert model.hidden_size == 120
    assert model.hidden_w == 12
    assert model.hidden_h == 10
    assert model.excitatory_count == 30
    assert model.inhibitory_count == 90
    assert len(model.motor_hidden) == 3 * 7
    assert len(model.recurrent) == 120 * (3 + 2)
    assert len(model.hidden_output) == 120 * 2
    assert len(model.hidden_prediction) == 120 * (4 + 2 + 1)

    model.reset(reset_weights=False)
    assert model.hidden_size == 120
    assert len(model.recurrent) == 120 * (3 + 2)


def test_motion_world_model_predicts_next_event_cell():
    model = PongSNN(seed=12)
    model.start()

    for frame in range(1, 7):
        x = 10 + frame
        y = 8 + frame
        event_camera = {
            "width": model.input_w,
            "height": model.input_h,
            "pixels": [_grid_index(x, y, model.input_w)],
        }
        model.step(event_camera, frame, world(tick=frame))

    predicted = {
        cell
        for cell, probability in model.last_prediction.items()
        if probability >= model.prediction_strong_threshold
    }
    assert _grid_index(17, 15, model.input_w) in predicted


if __name__ == "__main__":
    test_movement_costs_reward_and_hold_is_neutral_before_survival()
    test_survival_grows_slowly_with_episode_time()
    test_score_events_dominate_dense_rewards_and_reset_episode_timer()
    test_default_architecture_config_preserves_existing_shape()
    test_custom_architecture_config_controls_hidden_cloud_and_edges()
    test_motion_world_model_predicts_next_event_cell()
    print("reward smoke ok")
