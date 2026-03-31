#!/usr/bin/env python3
"""
Live Event Detection + Classification

Subscribes to mag_report inserts via Hasura GraphQL websocket.
Shows a live-updating graph and classifies detected events using TS2Vec.

Usage:
    pip install websocket-client requests numpy pandas matplotlib scipy torch ts2vec
    python live_detector.py
"""

import json
import threading
import time
from datetime import datetime, timezone, timedelta
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter, find_peaks
import requests
import websocket

# ─── Config ──────────────────────────────────────────────────────────────────
SENSOR_ID = 6
HASURA_URL = "https://hasura.pipestuesday.org/v1/graphql"
HASURA_WS_URL = "wss://hasura.pipestuesday.org/v1/graphql"
HASURA_SECRET = "PIPE_SUPERMMMSECRET_PIPE"

# Detection params (same as Matrix Profile notebook)
JITTER_MULTIPLIER = 2.0
ROLLING_STD_WINDOW_S = 5
MIN_EVENT_GAP_S = 2
MIN_EVENT_LEN_S = 2
PAD_S = 1

# Buffer: keep last 5 minutes of data in memory
BUFFER_SECONDS = 300

# How often to update the plot (ms)
PLOT_INTERVAL_MS = 1000

# ─── Hasura helpers ──────────────────────────────────────────────────────────
HEADERS = {
    "Content-Type": "application/json",
    "x-hasura-admin-secret": HASURA_SECRET,
}

def gql_query(query, variables=None):
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(HASURA_URL, json=payload, headers=HEADERS)
    resp.raise_for_status()
    body = resp.json()
    if "errors" in body:
        raise RuntimeError(f"GraphQL errors: {body['errors']}")
    return body["data"]

# ─── Load calibration labels for classification ─────────────────────────────
def load_calibration_signals():
    """Fetch calibration labels + their x-axis signals."""
    labels_raw = gql_query("""
    query {
      calibration_label(order_by: {created_at: asc}) {
        id, start_time, end_time
        fixture { id, name, type, sensor_id }
      }
    }
    """)["calibration_label"]

    cal_signals = []
    for lab in labels_raw:
        fix = lab.get("fixture") or {}
        if fix.get("sensor_id") != SENSOR_ID:
            continue

        mag = gql_query("""
        query GetMag($sid: bigint!, $since: timestamptz!, $until: timestamptz!) {
          mag_report(
            where: {sensor_id: {_eq: $sid}, created_at: {_gte: $since, _lte: $until}}
            order_by: {created_at: asc}, limit: 100000
          ) { x_axis_reading }
        }
        """, variables={"sid": str(SENSOR_ID), "since": lab["start_time"],
                        "until": lab["end_time"]})["mag_report"]

        if not mag:
            continue
        sig = np.array([float(r["x_axis_reading"]) for r in mag if r["x_axis_reading"] is not None])
        if len(sig) < 10:
            continue

        # Outlier removal
        med, std = np.median(sig), np.std(sig)
        sig = sig[np.abs(sig - med) <= 3 * std]

        cal_signals.append({
            "name": fix.get("name"),
            "type": fix.get("type"),
            "signal": sig,
        })

    return cal_signals

# ─── Simple MASS-based classification (no TS2Vec needed for live) ────────────
def classify_event(event_signal, cal_signals, top_k=3):
    """Classify an event. Returns top_k predictions sorted by distance."""
    if len(event_signal) < 10 or not cal_signals:
        return [("unknown", "?", float("inf"))]

    ev = event_signal.copy()
    mu, std = ev.mean(), ev.std()
    if std > 1e-8:
        ev = (ev - mu) / std

    all_matches = []
    for cal in cal_signals:
        q = cal["signal"].copy()
        q_mu, q_std = q.mean(), q.std()
        if q_std > 1e-8:
            q = (q - q_mu) / q_std

        target_len = 100
        ev_rs = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(ev)), ev)
        q_rs = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(q)), q)

        dist = np.sqrt(np.mean((ev_rs - q_rs) ** 2))
        all_matches.append((cal["type"], cal["name"], dist))

    all_matches.sort(key=lambda x: x[2])
    return all_matches[:top_k]

# ─── Event detection on buffer ───────────────────────────────────────────────
def detect_events(timestamps, values, sps):
    """Run rolling stddev event detection on the buffer."""
    if len(values) < 20:
        return [], None, None

    x = np.array(values)
    win = max(3, int(ROLLING_STD_WINDOW_S * sps))

    rolling_std = pd.Series(x).rolling(win, center=True, min_periods=3).std().fillna(0).values

    # Adaptive threshold: 2× median
    median_std = np.median(rolling_std[rolling_std > 0]) if np.any(rolling_std > 0) else 1.0
    threshold = median_std * JITTER_MULTIPLIER

    is_event = rolling_std > threshold

    min_gap = int(MIN_EVENT_GAP_S * sps)
    min_len = int(MIN_EVENT_LEN_S * sps)
    pad = int(PAD_S * sps)

    # Extract runs
    raw_runs = []
    in_ev = False
    ev_start = 0
    for i in range(len(is_event)):
        if is_event[i] and not in_ev:
            ev_start = i
            in_ev = True
        elif not is_event[i] and in_ev:
            raw_runs.append((ev_start, i - 1))
            in_ev = False
    if in_ev:
        raw_runs.append((ev_start, len(is_event) - 1))

    # Merge nearby
    merged = []
    for s, e in raw_runs:
        if merged and (s - merged[-1][1]) <= min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Filter short, pad
    merged = [(s, e) for s, e in merged if (e - s) >= min_len]
    merged = [(max(0, s - pad), min(len(x) - 1, e + pad)) for s, e in merged]

    return merged, rolling_std, threshold

# ─── Shared state ────────────────────────────────────────────────────────────
class LiveState:
    def __init__(self):
        self.lock = threading.Lock()
        self.timestamps = deque(maxlen=50000)
        self.values = deque(maxlen=50000)
        self.events = []  # list of (start_ts, end_ts, top3) — using timestamps not indices
        self.rolling_std = np.array([])
        self.threshold = 0
        self.sps = 10  # will be updated
        self.cal_signals = []
        self.detected_event_ids = set()  # track which events we already classified
        self.last_update = time.time()

state = LiveState()

# ─── Websocket subscription ─────────────────────────────────────────────────
def poll_new_data():
    """Poll for new data every 2 seconds to supplement the subscription."""
    while True:
        time.sleep(2)
        try:
            with state.lock:
                if len(state.timestamps) > 0:
                    last_ts = datetime.fromtimestamp(state.timestamps[-1], tz=timezone.utc)
                else:
                    last_ts = datetime.now(timezone.utc) - timedelta(seconds=30)

            reports = gql_query("""
            query GetMag($sid: bigint!, $since: timestamptz!) {
              mag_report(
                where: {sensor_id: {_eq: $sid}, created_at: {_gt: $since}}
                order_by: {created_at: asc}, limit: 1000
              ) { created_at, x_axis_reading }
            }
            """, variables={"sid": str(SENSOR_ID), "since": last_ts.isoformat()})["mag_report"]

            if reports:
                with state.lock:
                    existing_ts = set(state.timestamps)
                    added = 0
                    for r in reports:
                        try:
                            ts = pd.to_datetime(r["created_at"]).timestamp()
                            if ts in existing_ts:
                                continue
                            val = float(r["x_axis_reading"])
                            # Outlier check
                            if len(state.values) > 20:
                                med = np.median(list(state.values)[-100:])
                                std_val = np.std(list(state.values)[-100:])
                                if abs(val - med) > 3 * std_val:
                                    continue
                            state.timestamps.append(ts)
                            state.values.append(val)
                            added += 1
                        except (TypeError, ValueError):
                            pass
                    if added > 0:
                        # Sort by timestamp (poll might arrive out of order)
                        pairs = sorted(zip(state.timestamps, state.values))
                        state.timestamps.clear()
                        state.values.clear()
                        for t, v in pairs:
                            state.timestamps.append(t)
                            state.values.append(v)
                        state.last_update = time.time()
        except Exception as e:
            pass  # silently retry

def on_ws_message(ws, message):
    data = json.loads(message)

    # Handle connection_ack
    if data.get("type") == "connection_ack":
        # Start subscription — stream ALL new inserts, no limit
        ws.send(json.dumps({
            "id": "1",
            "type": "subscribe",
            "payload": {
                "query": f"""
                subscription {{
                  mag_report_stream(
                    cursor: {{initial_value: {{created_at: "now()"}}, ordering: ASC}}
                    batch_size: 100
                    where: {{sensor_id: {{_eq: {SENSOR_ID}}}}}
                  ) {{
                    created_at
                    x_axis_reading
                  }}
                }}
                """
            }
        }))
        print("Subscribed to mag_report_stream")
        return

    if data.get("type") == "next":
        payload = data.get("payload", {}).get("data", {})
        reports = payload.get("mag_report_stream", [])
        with state.lock:
            for r in reports:
                try:
                    ts = pd.to_datetime(r["created_at"]).timestamp()
                    val = float(r["x_axis_reading"])

                    # Outlier check
                    if len(state.values) > 20:
                        med = np.median(list(state.values)[-100:])
                        std_val = np.std(list(state.values)[-100:])
                        if abs(val - med) > 3 * std_val:
                            continue

                    state.timestamps.append(ts)
                    state.values.append(val)
                    state.last_update = time.time()
                except (TypeError, ValueError):
                    pass

def on_ws_error(ws, error):
    print(f"WS Error: {error}")

def on_ws_close(ws, close_status_code, close_msg):
    print(f"WS Closed: {close_status_code} {close_msg}")

def on_ws_open(ws):
    # Send connection_init with auth
    ws.send(json.dumps({
        "type": "connection_init",
        "payload": {
            "headers": {
                "x-hasura-admin-secret": HASURA_SECRET,
            }
        }
    }))
    print("WS Connected, authenticating...")

def start_subscription():
    ws = websocket.WebSocketApp(
        HASURA_WS_URL,
        on_open=on_ws_open,
        on_message=on_ws_message,
        on_error=on_ws_error,
        on_close=on_ws_close,
        subprotocols=["graphql-transport-ws"],
    )
    ws.run_forever()

# ─── Seed buffer with recent data ───────────────────────────────────────────
def seed_buffer():
    """Load the last 15 minutes of data so the graph isn't empty."""
    now = datetime.now(timezone.utc)
    since = now - timedelta(minutes=15)
    reports = gql_query("""
    query GetMag($sid: bigint!, $since: timestamptz!) {
      mag_report(
        where: {sensor_id: {_eq: $sid}, created_at: {_gte: $since}}
        order_by: {created_at: asc}, limit: 100000
      ) { created_at, x_axis_reading }
    }
    """, variables={"sid": str(SENSOR_ID), "since": since.isoformat()})["mag_report"]

    with state.lock:
        for r in reports:
            try:
                ts = pd.to_datetime(r["created_at"]).timestamp()
                val = float(r["x_axis_reading"])
                state.timestamps.append(ts)
                state.values.append(val)
            except (TypeError, ValueError):
                pass

    if len(state.timestamps) > 1:
        span = state.timestamps[-1] - state.timestamps[0]
        state.sps = len(state.timestamps) / max(span, 1)
    print(f"Seeded buffer with {len(state.timestamps)} points ({state.sps:.1f} Hz)")

# ─── Live plot ───────────────────────────────────────────────────────────────
def run_live_plot():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Live Event Detection — Sensor {SENSOR_ID}", fontweight="bold", fontsize=14)

    TYPE_COLORS = {"sink": "#f59e0b", "toilet": "#8b5cf6", "shower": "#3b82f6",
                   "unknown": "#9ca3af"}

    VISIBLE_WINDOW_S = 120  # 2 minutes visible at a time
    auto_follow = [True]

    def on_scroll(event):
        if event.inaxes not in (ax1, ax2):
            return
        import matplotlib.dates as mdates
        shift_s = VISIBLE_WINDOW_S * 0.1
        if event.button == 'up':
            shift_s = -shift_s
        day_shift = shift_s / 86400.0
        xlim = ax1.get_xlim()
        ax1.set_xlim(xlim[0] + day_shift, xlim[1] + day_shift)
        ax2.set_xlim(xlim[0] + day_shift, xlim[1] + day_shift)
        with state.lock:
            if len(state.timestamps) > 0:
                right_edge = mdates.date2num(datetime.fromtimestamp(state.timestamps[-1]))
                auto_follow[0] = ((xlim[1] + day_shift) >= right_edge - 5/86400.0)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    def update(frame):
        with state.lock:
            if len(state.values) < 20:
                return

            # Save current scroll position before clearing
            saved_xlim = ax1.get_xlim() if not auto_follow[0] else None

            ts_arr = np.array(state.timestamps)
            x_arr = np.array(state.values)

            # Update sps
            if len(ts_arr) > 1:
                span = ts_arr[-1] - ts_arr[0]
                if span > 0:
                    state.sps = len(ts_arr) / span

            # Run detection
            events, rolling_std, threshold = detect_events(ts_arr, x_arr, state.sps)
            if rolling_std is not None:
                state.rolling_std = rolling_std
                state.threshold = threshold

            # Classify new completed events (not the last one if still ongoing)
            # Classify completed events (store with timestamps)
            completed_events = events[:-1] if events else []
            for s, e in completed_events:
                ev_key = (int(ts_arr[s]), int(ts_arr[e]))
                ev_sig = x_arr[s:e+1]
                if ev_key not in state.detected_event_ids and len(ev_sig) >= 10:
                    top3 = classify_event(ev_sig, state.cal_signals, top_k=3)
                    state.detected_event_ids.add(ev_key)
                    state.events.append((ts_arr[s], ts_arr[e], top3))
                    t0s = datetime.fromtimestamp(ts_arr[s]).strftime('%H:%M:%S')
                    t1s = datetime.fromtimestamp(ts_arr[e]).strftime('%H:%M:%S')
                    print(f"  EVENT [{t0s} - {t1s}]:")
                    for rank, (t, n, d) in enumerate(top3):
                        print(f"    {rank+1}. {t} ({n}) dist={d:.3f}")

            # Classify ongoing event too (tentative) — store as temporary
            ongoing = []
            if events:
                s, e = events[-1]
                ev_sig = x_arr[s:e+1]
                if len(ev_sig) >= 10:
                    top3 = classify_event(ev_sig, state.cal_signals, top_k=3)
                    top3_tentative = [(t + "?", n, d) for t, n, d in top3]
                    ongoing.append((ts_arr[s], ts_arr[e], top3_tentative))
                else:
                    ongoing.append((ts_arr[s], ts_arr[e], [("...", "detecting", 0)]))
            state._ongoing = ongoing

            # Convert timestamps to datetime for plotting
            dt_arr = [datetime.fromtimestamp(t) for t in ts_arr]

            # Plot signal
            ax1.clear()
            ax1.plot(dt_arr, x_arr, lw=0.6, color="#2563eb", alpha=0.7)

            # Draw all events (completed + ongoing) using timestamps
            all_draw_events = list(state.events) + getattr(state, '_ongoing', [])
            for ev_ts_start, ev_ts_end, top3 in all_draw_events:
                # Convert event timestamps to datetime
                dt_s = datetime.fromtimestamp(ev_ts_start)
                dt_e = datetime.fromtimestamp(ev_ts_end)

                best_type = top3[0][0].rstrip("?")
                color = TYPE_COLORS.get(best_type, "#9ca3af")
                ax1.axvspan(dt_s, dt_e, color=color, alpha=0.2)
                ax1.axvline(dt_s, color="#16a34a", lw=2, alpha=0.8)

                # Find signal max in event range for label placement (clamp to visible area)
                ev_mask = (ts_arr >= ev_ts_start) & (ts_arr <= ev_ts_end)
                ev_vals = x_arr[ev_mask] if ev_mask.any() else x_arr
                vis_hi = np.percentile(x_arr, 99)
                y_pos = min(np.max(ev_vals) + 1, vis_hi) if len(ev_vals) > 0 else vis_hi

                dt_mid = datetime.fromtimestamp((ev_ts_start + ev_ts_end) / 2)
                lines = []
                for rank, (t, n, d) in enumerate(top3):
                    marker = "→" if rank == 0 else " "
                    lines.append(f"{marker} {rank+1}. {t} ({n}) d={d:.2f}")
                label = "\n".join(lines)
                ax1.text(dt_mid, y_pos, label, ha="center", va="bottom",
                         fontsize=9, fontweight="bold", color=color, fontfamily="monospace",
                         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.95, lw=2))

            y_lo, y_hi = np.percentile(x_arr, [1, 99])
            y_pad = (y_hi - y_lo) * 0.15
            ax1.set_ylim(y_lo - y_pad, y_hi + y_pad)
            ax1.set_ylabel("X-Axis Reading", fontsize=12)
            ax1.set_title(f"Live — {len(state.detected_event_ids)} events detected  |  "
                          f"{'AUTO-FOLLOW' if auto_follow[0] else 'SCROLL (scroll right to re-follow)'}",
                          fontsize=12)
            ax1.grid(True, alpha=0.2)

            # Plot rolling std
            ax2.clear()
            if len(state.rolling_std) > 0:
                ax2.plot(dt_arr[:len(state.rolling_std)], state.rolling_std,
                         lw=0.6, color="purple")
                ax2.axhline(state.threshold, color="red", ls="--", lw=1.5,
                            label=f"threshold ({state.threshold:.2f})")
                ax2.fill_between(dt_arr[:len(state.rolling_std)], 0, state.rolling_std,
                                 where=state.rolling_std > state.threshold,
                                 color="#8b5cf6", alpha=0.2)
            ax2.set_ylabel("Rolling Std (5s)", fontsize=12)
            ax2.set_xlabel("Time", fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.2)

            # Set visible x-axis window
            if auto_follow[0] and len(dt_arr) > 0:
                right = dt_arr[-1]
                left = right - timedelta(seconds=VISIBLE_WINDOW_S)
                ax1.set_xlim(left, right)
                ax2.set_xlim(left, right)
            elif saved_xlim is not None:
                ax1.set_xlim(saved_xlim)
                ax2.set_xlim(saved_xlim)

    ani = FuncAnimation(fig, update, interval=PLOT_INTERVAL_MS, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading calibration labels...")
    state.cal_signals = load_calibration_signals()
    print(f"Loaded {len(state.cal_signals)} calibration patterns:")
    for c in state.cal_signals:
        print(f"  {c['name']} ({c['type']}): {len(c['signal'])} samples")

    print("\nSeeding buffer with recent data...")
    seed_buffer()

    # Pre-classify events already in the buffer
    print("\nClassifying events in seeded data...")
    with state.lock:
        if len(state.values) > 20:
            ts_arr = np.array(state.timestamps)
            x_arr = np.array(state.values)
            if len(ts_arr) > 1:
                span = ts_arr[-1] - ts_arr[0]
                if span > 0:
                    state.sps = len(ts_arr) / span
            events, rolling_std, threshold = detect_events(ts_arr, x_arr, state.sps)
            for s, e in events:
                ev_sig = x_arr[s:e+1]
                if len(ev_sig) >= 10:
                    top3 = classify_event(ev_sig, state.cal_signals, top_k=3)
                    ev_key = (int(ts_arr[s]), int(ts_arr[e]))
                    state.detected_event_ids.add(ev_key)
                    state.events.append((ts_arr[s], ts_arr[e], top3))
                    t0s = datetime.fromtimestamp(ts_arr[s]).strftime('%H:%M:%S')
                    t1s = datetime.fromtimestamp(ts_arr[e]).strftime('%H:%M:%S')
                    print(f"  EVENT [{t0s} - {t1s}]:")
                    for rank, (t, n, d) in enumerate(top3):
                        print(f"    {rank+1}. {t} ({n}) dist={d:.3f}")
            print(f"  Found {len(events)} events in seeded data")

    print("\nStarting websocket subscription...")
    ws_thread = threading.Thread(target=start_subscription, daemon=True)
    ws_thread.start()

    print("Starting poll thread (backup data fetcher)...")
    poll_thread = threading.Thread(target=poll_new_data, daemon=True)
    poll_thread.start()

    print("Starting live plot (close window to stop)...\n")
    run_live_plot()
