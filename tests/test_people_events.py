from app.pipeline.people import Track, analyze_people, debounce

T = [round(i * 0.2, 3) for i in range(100)]          # 20 s at 5 fps
SEAT = (0.4, 0.4, 0.6, 0.8)
DESK = (0.2, 0.5, 0.8, 0.9)


def track(tid, t0, t1, box=SEAT):
    ts = [t for t in T if t0 <= t <= t1]
    return Track(tid, ts, [box] * len(ts))


def moved(tid, t_sit_end, t_gone, box_far=(0.05, 0.1, 0.25, 0.5)):
    ts = [t for t in T if t <= t_gone]
    return Track(tid, ts, [SEAT if t < t_sit_end else box_far for t in ts])


def test_debounce_ignores_short_blips_and_backdates():
    times = [i * 0.2 for i in range(10)]
    raw = [False, False, True, False, False, True, True, True, True, True]
    assert debounce(times, raw, 0.6) == [False, False, False, False, False, True, True, True, True, True]


def test_cashier_leaves_room_event_at_movement_start():
    tl = analyze_people([moved(1, 9.0, 12.0)], T, DESK, None, 20.0, 0.55)
    left = [e for e in tl.events if e.kind == "cashier_left"]
    assert len(left) == 1 and abs(left[0].t - 9.0) <= 0.2
    assert tl.away_at(15.0) and not tl.away_at(5.0)


def test_trip_to_safe_is_not_departure():
    safe = (0.0, 0.05, 0.3, 0.55)
    tr = Track(1, T, [(0.05, 0.1, 0.25, 0.5) if 6.0 <= t < 10.0 else SEAT for t in T])
    tl = analyze_people([tr], T, DESK, safe, 20.0, 0.55)
    assert not [e for e in tl.events if e.kind == "cashier_left"]


def test_short_dropout_is_not_departure():
    tr = Track(1, [t for t in T if not (4.0 <= t <= 4.2)], [SEAT] * (len(T) - 2))
    tl = analyze_people([tr], T, DESK, None, 20.0, 0.55)
    assert not [e for e in tl.events if e.kind == "cashier_left"]


def test_visitor_entry_exit_and_reflection_ignored():
    cashier = track(1, 0, 20)
    client = track(2, 2.0, 6.0, box=(0.3, 0.1, 0.5, 0.4))
    reflection = track(3, 1.0, 19.0, box=(0.0, 0.2, 0.05, 0.26))
    tl = analyze_people([cashier, client, reflection], T, DESK, None, 20.0, 0.55)
    kinds = [(e.kind, e.t) for e in tl.events]
    assert ("entered", 2.0) in kinds and ("exited", 6.0) in kinds
    assert not [e for e in tl.events if e.track_id == 3]
    assert tl.others_at(4.0) == 1 and tl.others_at(10.0) == 0


def test_cashier_enters_empty_room():
    tl = analyze_people([track(1, 3.0, 20)], T, DESK, None, 20.0, 0.55)
    assert [e.kind for e in tl.events] == ["cashier_entered"]


def test_no_tracks():
    tl = analyze_people([], T, DESK, None, 20.0, 0.55)
    assert tl.cashier_id is None and tl.events == [] and tl.away_at(5.0)


def test_short_visitor_track_is_ignored():
    cashier = track(1, 0, 20)
    blip = track(2, 16.0, 17.2, box=(0.3, 0.1, 0.5, 0.4))
    tl = analyze_people([cashier, blip], T, DESK, None, 20.0, 0.55)
    assert tl.events == [] and tl.others_at(16.6) == 0


def test_cashier_id_switch_after_absence_is_merged():
    first, second = track(1, 0, 5.0), track(2, 10.0, 20)
    tl = analyze_people([first, second], T, DESK, None, 20.0, 0.55)
    left = [e for e in tl.events if e.kind == "cashier_left"]
    assert len(left) == 1 and abs(left[0].t - 5.2) <= 0.4
    assert not [e for e in tl.events if e.kind in ("entered", "exited", "cashier_entered")]
    assert tl.others_at(15.0) == 0 and not tl.away_at(15.0) and tl.away_at(7.0)


def test_returning_cashier_is_not_a_visitor():
    first, second = track(1, 0, 12.0), track(2, 16.0, 20)
    tl = analyze_people([first, second], T, DESK, None, 20.0, 0.55)
    assert [e.kind for e in tl.events] == ["cashier_left"] and tl.others_at(18.0) == 0


def test_cashier_boxes_follow_the_merged_track():
    """The cashier zone of probes/S7: the merged cashier track's box per sample, None while out of view."""
    first, second = track(1, 0, 5.0), track(2, 10.0, 20, box=(0.42, 0.4, 0.62, 0.8))
    client = track(3, 2.0, 8.0, box=(0.3, 0.1, 0.5, 0.4))
    tl = analyze_people([first, second, client], T, DESK, None, 20.0, 0.55)
    assert tl.cashier_box_at(2.0) == SEAT and tl.cashier_box_at(7.0) is None       # the client is never the cashier
    assert tl.cashier_box_at(15.1) == (0.42, 0.4, 0.62, 0.8)                        # ID switch merged, nearest sample
    assert analyze_people([], T, DESK, None, 20.0, 0.55).cashier_box_at(5.0) is None


def test_an_off_seat_run_at_the_clip_start_does_not_start_a_departure():
    """X-29: she sits 0-6.8 s, but the seat (median of all centres) lands nearer her later path, so the seated run is
    "away"; she walks past that point 7.0-9.4 s and out of view after 15 s. The 2.8 s gap used to chain the seated run
    into the departure (cashier_left at 0.0); the departure is the run that starts at 9.6 s."""
    def box(t):
        if t <= 6.8:
            cx, cy = 0.5, 0.6                                       # seated
        elif t <= 9.4:
            cx, cy = 0.5, 0.48                                      # getting up, passing the median point
        else:
            k = (t - 9.6) / 5.4                                     # walk to the door
            cx, cy = 0.35 - 0.25 * k, 0.4 - 0.1 * k
        return (cx - 0.1, cy - 0.2, cx + 0.1, cy + 0.2)
    ts = [t for t in T if t <= 15.0]
    tl = analyze_people([Track(1, ts, [box(t) for t in ts])], T, DESK, None, 17.4, 0.55)
    assert [(e.kind, e.t) for e in tl.events] == [("cashier_left", 9.6)]
