"""Pass into the full-back / centre-back channel — formation-agnostic, spatial.

The *channel* is the gap between a defending full-back (or wing-back) and the
nearest centre-back. A pass received there — behind the defensive line, out in
the wide channel or half-space — is one of the most dangerous entries in the
modern game.

Why spatial? Modern defending is positionally fluid (full-backs push up, midfielders
drop in), so using a player's *registered* position (LB/CB/RB) to decide who is in
the last line is not rigorous — it misses receptions behind a vacated full-back. The
event snapshot already contains every player's x/y at the pass, so we determine the
last line **purely from coordinates** (no extra tracking cost) and define the
channel pass without position labels at all:

    channel pass = a pass RECEIVED behind the defending last line, in a LATERAL
             GAP (isolated from the nearest last-line defender by >= ``iso_min`` m),
             out in the channel/half-space (not central), in the final third.

Last line = the rearmost spatial band of the defending outfield players (anyone
within ``band_width`` m of the deepest outfielder — this adapts to back-3/4/5 and
to midfielders dropping in). The goalkeeper is the only role still taken from the
label (keepers don't roam into the line); everything else is coordinate-driven.

Reception point = the intended receiver's position at the NEXT event (captures the
run onto the ball; far more reliable than the next event's ball, which overshoots
completed passes and jumps to the interception point on incomplete ones).

Types (from the passer's position ≈ the ball at the pass):
    Type1_wide       passer in the wing (|y| >= wing_min_y, by the touchline)
    Type2_halfspace  passer in the half-space (central_max_y..wing_min_y)
    Type3_switch     ball was just switched across to this side (recent same-team
                     ball >= switch_dy m laterally away)
Central / own-half passers are rejected as "simple in-behind run / build-up". A
half-space refinement (Type2) additionally needs a real receiver RUN and a channel
reception.

All coordinates are oriented so +x = the attacking team's attacking direction
(toward the defenders' own goal); "behind the line" = larger x.
"""
from __future__ import annotations

import math
from collections import defaultdict

from . import data_loader as dl


# ── geometry helpers (event data only — no tracking needed) ─────────────────────

def _ball_xy(ev):
    """Ball position at this event ≈ the pass origin."""
    b = ev.get("ball")
    if isinstance(b, list):
        b = b[0] if b else None
    if not b:
        return (None, None)
    return (b.get("x"), b.get("y"))


def _team_attack_dirs(events):
    """Map (homeTeam_bool, period) -> attack direction sign in x (+1 or -1).

    Derived from each team's goalkeeper median x: the GK is by his own goal, so
    the team attacks toward the opposite end -> attack_dir = -sign(median GK x).
    """
    gk_x = defaultdict(list)        # (is_home, period) -> [GK x ...]
    for ev in events:
        ge = ev.get("gameEvents") or {}
        per = ge.get("period")
        for is_home, key in ((True, "homePlayers"), (False, "awayPlayers")):
            for p in ev.get(key) or []:
                if p.get("positionGroupType") == "GK" and p.get("x") is not None:
                    gk_x[(is_home, per)].append(p["x"])
    dirs = {}
    for k, xs in gk_x.items():
        xs = sorted(xs)
        med = xs[len(xs) // 2]
        dirs[k] = -1.0 if med > 0 else 1.0
    return dirs


def _player_xy(ev, is_home, pid):
    key = "homePlayers" if is_home else "awayPlayers"
    for p in ev.get(key) or []:
        if p.get("playerId") == pid:
            return p.get("x"), p.get("y")
    return None, None


def _outfield(ev, def_is_home, d):
    """Oriented (x*d, y*d, pos) for the defending team's non-GK tracked players."""
    key = "homePlayers" if def_is_home else "awayPlayers"
    out = []
    for p in ev.get(key) or []:
        if p.get("positionGroupType") == "GK":          # GK: the one label we keep
            continue
        if p.get("visibility") == "ESTIMATED":
            continue
        x, y = p.get("x"), p.get("y")
        if x is None or y is None:
            continue
        out.append((x * d, y * d, p.get("positionGroupType")))
    return out


def _last_line(outfield, band_width, min_n):
    """Rearmost spatial band: outfielders within ``band_width`` of the deepest one."""
    if len(outfield) < min_n:
        return []
    x_max = max(p[0] for p in outfield)                 # deepest = nearest own goal
    line = [p for p in outfield if p[0] >= x_max - band_width]
    return sorted(line, key=lambda p: p[1])             # by oriented y (left->right)


def _line_x_at(line, y):
    """Interpolate the line's x at lateral position ``y`` (flat outside the span)."""
    if y <= line[0][1]:
        return line[0][0]
    if y >= line[-1][1]:
        return line[-1][0]
    for k in range(len(line) - 1):
        y0, y1 = line[k][1], line[k + 1][1]
        if y0 <= y <= y1:
            x0, x1 = line[k][0], line[k + 1][0]
            t = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
            return x0 + (x1 - x0) * t
    return line[-1][0]


def extract_match(match_id, data_dir="data", band_width=10.0, min_line=3,
                  depth_min=1.0, depth_cap=25.0, iso_min=3.0, channel_min=7.0,
                  min_end_x=15.0, exclude_cross=True, min_passer_x=0.0,
                  central_max_y=7.0, wing_min_y=20.0, switch_dy=25.0,
                  switch_lookback=3, type2_refine=True, type2_run_min=9.0,
                  type2_channel_min_y=8.0):
    """Return {'rows': [...channel passes...], 'n_pass_eval': int, 'n_D_total': int}."""
    try:
        meta = dl.load_meta(match_id, data_dir)
        events = dl.load_events(match_id, data_dir)
        home = meta.get("homeTeam", {}); away = meta.get("awayTeam", {})
        dirs = _team_attack_dirs(events)

        rows, n_pass_eval, n_D_total = [], 0, 0

        for i, ev in enumerate(events[:-1]):
            pe = ev.get("possessionEvents") or {}
            ge = ev.get("gameEvents") or {}
            if pe.get("possessionEventType") != "PA":
                continue
            if ge.get("setpieceType") not in (None, "O"):
                continue
            if exclude_cross and (pe.get("crossType") or pe.get("crosserPlayerId")):
                continue

            is_home = bool(ge.get("homeTeam"))
            per = ge.get("period")
            d = dirs.get((is_home, per), 1.0)

            sx, sy = _ball_xy(ev)
            if sx is None or sy is None:
                continue
            rid = pe.get("targetPlayerId") or pe.get("receiverPlayerId")
            if rid is None:
                continue
            rnx_now, rny_now = _player_xy(ev, is_home, rid)
            rnx_nxt, rny_nxt = _player_xy(events[i + 1], is_home, rid)
            rx, ry = (rnx_nxt, rny_nxt) if rnx_nxt is not None else (rnx_now, rny_now)
            if rx is None or ry is None:
                continue
            n_pass_eval += 1
            lb = pe.get("linesBrokenType") or ""
            if "D" in lb:
                n_D_total += 1

            ex_o, ey_o = rx * d, ry * d                  # reception point
            sx_o, sy_o = sx * d, sy * d                  # passer ≈ ball at pass

            # --- spatial last line + channel geometry (label-free) ---
            line = _last_line(_outfield(ev, not is_home, d), band_width, min_line)
            if len(line) < min_line:
                continue
            line_x = _line_x_at(line, ey_o)
            depth = ex_o - line_x                        # behind the last line
            if not (depth_min <= depth <= depth_cap):
                continue
            if ex_o < min_end_x:                         # final third
                continue
            if abs(ey_o) < channel_min:                  # not dead central
                continue
            # the channel reception: receiver isolated from the last line (a real gap)
            dists = sorted(line, key=lambda p: math.hypot(ex_o - p[0], ey_o - p[1]))
            iso = math.hypot(ex_o - dists[0][0], ey_o - dists[0][1])
            if iso < iso_min:
                continue
            near = dists[:2]

            # --- passer-position type + filter (gold-tuned) ---
            if sx_o < min_passer_x:                      # own-half over-the-top
                continue
            absy = abs(sy_o)
            if absy < central_max_y:                     # central passer = build-up
                continue

            is_switch = False
            for j in range(i - 1, max(-1, i - 1 - switch_lookback), -1):
                gej = events[j].get("gameEvents") or {}
                if bool(gej.get("homeTeam")) != is_home:
                    continue
                bjx, bjy = _ball_xy(events[j])
                if bjy is None:
                    continue
                is_switch = abs(sy_o - bjy * d) >= switch_dy
                break

            run_x = (ex_o - rnx_now * d) if rnx_now is not None else None
            run_y = (ey_o - rny_now * d) if rny_now is not None else None
            run_mag = math.hypot(run_x, run_y) if run_x is not None else None

            if is_switch:
                ptype = "Type3_switch"
            elif absy >= wing_min_y:
                ptype = "Type1_wide"
            else:
                ptype = "Type2_halfspace"

            if ptype == "Type2_halfspace" and type2_refine:
                if (run_mag is None or run_mag < type2_run_min
                        or abs(ey_o) < type2_channel_min_y):
                    continue

            gc = pe.get("gameClock")
            rows.append({
                "match_id": str(match_id), "period": per,
                "game_clock": gc,
                "match_min": (f"{int(gc)//60:02d}:{int(gc)%60:02d}" if gc is not None else None),
                "team": (home if is_home else away).get("name"),
                "opponent": (away if is_home else home).get("name"),
                "passer": pe.get("passerPlayerName"),
                "passer_pos": next((p.get("positionGroupType")
                                    for p in ev.get("homePlayers" if is_home else "awayPlayers") or []
                                    if p.get("playerId") == pe.get("passerPlayerId")), None),
                "receiver": pe.get("receiverPlayerName") or pe.get("targetPlayerName"),
                "linesBrokenType": lb or None,
                "broke_D": "D" in lb, "broke_M": "M" in lb, "broke_A": "A" in lb,
                "completed": pe.get("passOutcomeType") == "C",
                "passType": pe.get("passType"),
                "ballHeight": pe.get("ballHeightType"),
                "start_x": round(sx_o, 2), "start_y": round(sy_o, 2),
                "end_x": round(ex_o, 2), "end_y": round(ey_o, 2),
                "length": round(math.hypot(ex_o - sx_o, ey_o - sy_o), 2),
                "depth_behind": round(depth, 2), "line_x": round(line_x, 2),
                "iso_dist": round(iso, 2), "n_line": len(line),
                "near1_x": round(near[0][0], 2), "near1_y": round(near[0][1], 2),
                "near2_x": round(near[1][0], 2) if len(near) > 1 else None,
                "near2_y": round(near[1][1], 2) if len(near) > 1 else None,
                "side": "Left" if ey_o < 0 else "Right",
                "run_x": None if run_x is None else round(run_x, 2),
                "run_y": None if run_y is None else round(run_y, 2),
                "run_mag": None if run_mag is None else round(run_mag, 2),
                "channel_type": ptype, "attack_dir": d,
            })

        return {"rows": rows, "n_pass_eval": n_pass_eval, "n_D_total": n_D_total}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "match_id": str(match_id)}
