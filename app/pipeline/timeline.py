"""S5: noisy P(yes) series -> state intervals (median smoothing + hysteresis)."""
import statistics as st
from dataclasses import dataclass

Series = list[tuple[float, float]]


@dataclass
class Interval:
    start: float
    end: float
    peak_t: float
    peak_p: float
    strong: int          # raw samples with p >= strong_p

    @property
    def length(self) -> float:
        return self.end - self.start


def smooth(series: Series, k: int = 3) -> Series:
    if len(series) < k:
        return list(series)
    half = k // 2
    n = len(series)
    out: Series = []
    for i, (t, p) in enumerate(series):
        if i < half or i >= n - half:
            out.append((t, p))  # window would be truncated: keep raw value
        else:
            out.append((t, st.median(q for _, q in series[i - half: i + half + 1])))
    return out


def intervals(series: Series, on: float = 0.6, off: float = 0.4, strong_p: float = 0.8) -> list[Interval]:
    result: list[Interval] = []
    cur: Interval | None = None
    n_on = 0  # raw samples with p >= on inside the current candidate
    for i, (t, p) in enumerate(smooth(series)):
        raw_p = series[i][1]
        if cur is None:
            if p < on:
                continue
            cur = Interval(t, t, t, raw_p, 0)
            n_on = 0
        elif p < off:
            if n_on >= 2:
                result.append(cur)
            cur = None
            continue
        cur.end = t
        if raw_p > cur.peak_p:
            cur.peak_t, cur.peak_p = t, raw_p
        if raw_p >= strong_p:
            cur.strong += 1
        if raw_p >= on:
            n_on += 1
    if cur is not None and n_on >= 2:
        result.append(cur)
    return result


def active(ints: list[Interval], t: float, tol: float = 0.0) -> Interval | None:
    for iv in ints:
        if iv.start - tol <= t <= iv.end + tol:
            return iv
    return None


def peak_in(series: Series, t0: float, t1: float) -> tuple[float, float]:
    pts = [(t, p) for t, p in series if t0 <= t <= t1]
    return max(pts, key=lambda x: x[1]) if pts else (t0, 0.0)
