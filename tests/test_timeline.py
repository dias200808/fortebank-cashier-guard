from app.pipeline.timeline import active, intervals, peak_in, smooth


def s(ps):
    return [(float(i), p) for i, p in enumerate(ps)]


def test_step_gives_one_interval_with_backdated_start():
    iv = intervals(s([0.1, 0.1, 0.9, 0.95, 0.9, 0.1, 0.1]))
    assert len(iv) == 1 and iv[0].start == 2.0 and iv[0].end == 4.0
    assert iv[0].peak_t == 3.0 and iv[0].strong == 3


def test_single_spike_removed_by_median():
    assert intervals(s([0.1, 0.1, 0.95, 0.1, 0.1])) == []


def test_hysteresis_keeps_interval_through_middle_values():
    iv = intervals(s([0.1, 0.9, 0.9, 0.5, 0.5, 0.9, 0.9, 0.1, 0.1]))
    assert len(iv) == 1 and iv[0].start == 1.0 and iv[0].end == 6.0


def test_active_and_peak_in():
    iv = intervals(s([0.1, 0.9, 0.9, 0.9, 0.1]))
    assert active(iv, 2.0) is iv[0] and active(iv, 4.5) is None and active(iv, 4.5, tol=1.5) is iv[0]
    assert peak_in(s([0.1, 0.3, 0.7, 0.2]), 0.0, 2.0) == (2.0, 0.7)
    assert peak_in([], 1.0, 2.0) == (1.0, 0.0)


def test_smooth_short_series_unchanged():
    assert smooth([(0.0, 0.9)]) == [(0.0, 0.9)]


def test_spike_at_series_edges_creates_no_state():
    assert intervals([(0.0, 0.95), (1.0, 0.3), (2.0, 0.1), (3.0, 0.1)]) == []
    assert intervals([(0.0, 0.1), (1.0, 0.1), (2.0, 0.3), (3.0, 0.95)]) == []


def test_interval_end_not_extended_at_series_edge():
    iv = intervals([(3.0, 0.1), (4.0, 0.9), (4.2, 0.9), (5.0, 0.1)])
    assert len(iv) == 1 and iv[0].start == 4.0 and iv[0].end == 4.2


def test_state_at_clip_start_is_kept():
    iv = intervals([(0.0, 0.9), (1.0, 0.9), (2.0, 0.9), (3.0, 0.1), (4.0, 0.1)])
    assert len(iv) == 1 and iv[0].start == 0.0 and iv[0].end == 2.0
