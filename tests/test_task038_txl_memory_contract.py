import pytest

from h200_locomotion_lab.training.task038_txl_memory import (
    FrameToken,
    HiddenStateToken,
    TxlMemoryCache,
    TxlMemoryConfig,
    encode_frame_token,
)


def _cache(memory_len: int = 3) -> TxlMemoryCache:
    return TxlMemoryCache(
        TxlMemoryConfig(num_envs=2, num_layers=2, memory_len=memory_len, token_dim=4)
    )


def _frame(value: float):
    return ([value, value + 0.1], [value + 0.2, value + 0.3])


def test_two_parallel_env_ids_have_independent_layer_caches():
    cache = _cache()

    env0 = cache.append_segment(0, [_frame(0.0), _frame(1.0)])
    env1 = cache.append_segment(1, [_frame(10.0)])

    assert env0["after_memory_lengths"] == (2, 2)
    assert env1["after_memory_lengths"] == (1, 1)
    assert cache.memory_lengths(0) == (2, 2)
    assert cache.memory_lengths(1) == (1, 1)
    assert env0["cache_env_ids"] == ((0, 0), (0, 0))
    assert env1["cache_env_ids"] == ((1,), (1,))


def test_second_segment_attends_previous_segment_memory_length_before_append():
    cache = _cache()

    first = cache.append_segment(0, [_frame(0.0), _frame(1.0)])
    second = cache.append_segment(0, [_frame(2.0)])

    assert first["attended_previous_memory_lengths"] == (0, 0)
    assert second["attended_previous_memory_lengths"] == (2, 2)
    assert second["after_memory_lengths"] == (3, 3)


def test_inner_reset_records_event_but_preserves_txl_memory():
    cache = _cache()
    cache.append_segment(0, [_frame(0.0), _frame(1.0)])

    debug = cache.append_segment(0, [_frame(2.0)], inner_reset=True)

    assert debug["reset_events"][0]["event"] == "inner_reset"
    assert debug["reset_events"][0]["decision"] == "preserve_memory"
    assert debug["reset_events"][0]["before_memory_lengths"] == (2, 2)
    assert debug["attended_previous_memory_lengths"] == (2, 2)
    assert debug["inner_reset_events"] == 1
    assert debug["after_memory_lengths"] == (3, 3)


def test_outer_reset_clears_selected_env_only_before_new_append():
    cache = _cache()
    cache.append_segment(0, [_frame(0.0), _frame(1.0)])
    cache.append_segment(1, [_frame(10.0), _frame(11.0)])

    debug = cache.append_segment(0, [_frame(2.0)], outer_reset=True)

    assert debug["reset_events"][0]["event"] == "outer_reset"
    assert debug["reset_events"][0]["decision"] == "clear_selected_env_memory"
    assert debug["reset_events"][0]["before_memory_lengths"] == (2, 2)
    assert debug["attended_previous_memory_lengths"] == (0, 0)
    assert debug["after_memory_lengths"] == (1, 1)
    assert cache.memory_lengths(1) == (2, 2)


def test_per_env_outer_reset_mask_cannot_clear_or_leak_other_env_memory():
    cache = _cache()
    cache.append_segment(0, [_frame(0.0), _frame(1.0)])
    cache.append_segment(1, [_frame(10.0), _frame(11.0), _frame(12.0)])

    reset_debug = cache.outer_reset([1])
    env0_debug = cache.append_segment(0, [_frame(2.0)])
    env1_debug = cache.append_segment(1, [_frame(13.0)])

    assert reset_debug[0]["env_id"] == 1
    assert reset_debug[0]["after_memory_lengths"] == (0, 0)
    assert env0_debug["attended_previous_memory_lengths"] == (2, 2)
    assert env0_debug["cache_env_ids"] == ((0, 0, 0), (0, 0, 0))
    assert env1_debug["attended_previous_memory_lengths"] == (0, 0)
    assert env1_debug["cache_env_ids"] == ((1,), (1,))


def test_memory_len_cap_applies_per_layer_and_per_env():
    cache = _cache(memory_len=2)

    cache.append_segment(0, [_frame(0.0), _frame(1.0), _frame(2.0)])
    cache.append_segment(1, [_frame(10.0), _frame(11.0)])
    debug = cache.append_segment(0, [_frame(3.0)])

    assert debug["attended_previous_memory_lengths"] == (2, 2)
    assert debug["after_memory_lengths"] == (2, 2)
    assert cache.memory_lengths(0) == (2, 2)
    assert cache.memory_lengths(1) == (2, 2)
    assert debug["cache_env_ids"] == ((0, 0), (0, 0))


def test_incremental_inference_updates_cache_lengths_and_counters():
    cache = _cache()

    first = cache.step(0, [0.0, 0.1], [0.2, 0.3], incremental=True)
    second = cache.step(0, [1.0, 1.1], [1.2, 1.3], incremental=True)

    assert first["incremental"] is True
    assert first["incremental_steps"] == 1
    assert first["after_memory_lengths"] == (1, 1)
    assert second["attended_previous_memory_lengths"] == (1, 1)
    assert second["incremental_steps"] == 2
    assert second["segments_appended"] == 2
    assert second["tokens_appended"] == 2
    assert second["after_memory_lengths"] == (2, 2)


def test_encode_frame_token_is_obs_action_token_not_history_stack():
    token = encode_frame_token({"b": 2.0, "a": 1.0}, [3.0], token_dim=4)

    assert token.obs_dim == 2
    assert token.action_dim == 1
    assert token.values == (1.0, 2.0, 3.0, 0.0)


def test_empty_segment_is_rejected_without_incrementing_segment_counter():
    cache = _cache()
    before = cache.debug_snapshot()["envs"][0]

    with pytest.raises(ValueError, match="empty TXL segment rejected"):
        cache.append_segment(0, [])

    after = cache.debug_snapshot()["envs"][0]
    assert before["segments_appended"] == 0
    assert after["segments_appended"] == 0
    assert after["tokens_appended"] == 0
    assert after["memory_lengths"] == (0, 0)


def test_combined_outer_and_inner_reset_clears_first_then_preserves_empty_memory():
    cache = _cache()
    cache.append_segment(0, [_frame(0.0), _frame(1.0)])

    debug = cache.append_segment(
        0,
        [_frame(2.0)],
        outer_reset=True,
        inner_reset=True,
    )

    assert [event["event"] for event in debug["reset_events"]] == [
        "outer_reset",
        "inner_reset",
    ]
    assert debug["reset_events"][0]["decision"] == "clear_selected_env_memory"
    assert debug["reset_events"][0]["before_memory_lengths"] == (2, 2)
    assert debug["reset_events"][0]["after_memory_lengths"] == (0, 0)
    assert debug["reset_events"][1]["decision"] == "preserve_memory"
    assert debug["reset_events"][1]["before_memory_lengths"] == (0, 0)
    assert debug["reset_events"][1]["after_memory_lengths"] == (0, 0)
    assert debug["attended_previous_memory_lengths"] == (0, 0)
    assert debug["after_memory_lengths"] == (1, 1)
    assert debug["inner_reset_events"] == 1
    assert debug["outer_reset_events"] == 1


def test_frame_token_dim_mismatch_fails_before_append():
    cache = _cache()

    with pytest.raises(ValueError, match="FrameToken dim 3 does not match token_dim=4"):
        cache.append_segment(0, [FrameToken(values=(1.0, 2.0, 3.0), obs_dim=2, action_dim=1)])

    assert cache.debug_snapshot()["envs"][0]["segments_appended"] == 0


def test_invalid_env_id_fails():
    cache = _cache()

    with pytest.raises(ValueError, match=r"env_id=2 outside \[0, 2\)"):
        cache.append_segment(2, [_frame(0.0)])

    with pytest.raises(ValueError, match=r"env_id=-1 outside \[0, 2\)"):
        cache.memory_lengths(-1)


def test_leak_guard_detects_corrupted_private_cache_env_id():
    cache = _cache()
    cache.append_segment(0, [_frame(0.0)])
    original = cache._envs[0].layers[0][0]
    cache._envs[0].layers[0][0] = HiddenStateToken(
        env_id=1,
        layer_id=original.layer_id,
        episode_index=original.episode_index,
        token_index=original.token_index,
        values=original.values,
    )

    with pytest.raises(RuntimeError, match="env memory leak detected"):
        cache.append_segment(0, [_frame(1.0)])


def test_leak_guard_detects_stale_outer_episode_index_in_private_cache():
    cache = _cache()
    cache.append_segment(0, [_frame(0.0)])
    original = cache._envs[0].layers[1][0]
    cache._envs[0].layers[1][0] = HiddenStateToken(
        env_id=original.env_id,
        layer_id=original.layer_id,
        episode_index=original.episode_index + 1,
        token_index=original.token_index,
        values=original.values,
    )

    with pytest.raises(RuntimeError, match="stale outer episode memory detected"):
        cache.append_segment(0, [_frame(1.0)])
