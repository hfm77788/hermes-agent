from gateway.run import _normalize_clarify_reply


def test_numeric_reply_maps_to_choice_text():
    assert _normalize_clarify_reply("2", ["定版", "Round 3", "跳过"]) == "Round 3"


def test_out_of_range_numeric_reply_passes_through():
    assert _normalize_clarify_reply("9", ["A", "B"]) == "9"


def test_non_digit_reply_passes_through():
    assert _normalize_clarify_reply("给方案", ["定版", "Round 3"]) == "给方案"


def test_open_ended_reply_passes_through():
    assert _normalize_clarify_reply("1", None) == "1"
