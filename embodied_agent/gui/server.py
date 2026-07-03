"""The GUI server: a background training thread + a stdlib HTTP dashboard."""
from __future__ import annotations

import io
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402

from ..config import load_config  # noqa: E402
from .page import PAGE  # noqa: E402

# Scenarios offered in the UI: name -> (preset, description)
SCENARIOS = {
    "baseline": ("cpu_small",
                 "Plain deep-RL agent (fast, CPU). The control group."),
    "bio_cpu": ("cpu_bio",
                "The living agent: allostasis, sleep, one continual life, "
                "metabolic cost (CPU)."),
    "bio_gpu": ("gpu_laptop",
                "The living agent, bigger model, on your GPU (RTX 4060 Ti). "
                "Falls back to CPU if no GPU."),
    "colony": ("colony",
               "A whole COMMUNITY of INDIVIDUALS: each creature has its own "
               "body, senses and mind (its own brain + memory). They share a "
               "world, forage, collide, breed (offspring inherit a parent's "
               "brain), and evolve. (Biological switches don't apply here.)"),
}

# Biological switches the UI can toggle on top of a scenario.
SWITCHES = {
    "neuromod": "B1 neuromodulation & allostasis",
    "sleep": "B2 sleep, consolidation & dreaming",
    "dev": "B3 continual life & critical period",
    "metab": "B4 metabolic cost of cognition",
}


def _placeholder_png() -> bytes:
    img = np.full((360, 360, 3), 0.12, dtype=np.float32)
    img[160:200, 60:300] = 0.3
    buf = io.BytesIO()
    mpimg.imsave(buf, img, format="png")
    return buf.getvalue()


class GuiState:
    """Shared state between the training thread and the HTTP handlers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_flag = False
        self.running = False
        self.colony_mode = False
        self.frame_png = _placeholder_png()
        self.senses_png = _placeholder_png()
        self.renderer = None
        self.senses_renderer = None
        self.sensor_cfg = None
        self.focused_id = None          # creature the user clicked to inspect
        self._snap = None               # {S, pts:[(id,x,y)]} for click->creature
        self.device = "cpu"
        self.status = {"phase": "idle", "step": 0, "message": "Press Start."}
        self.config = {}
        self.history = {k: [] for k in
                        ("step", "reward", "energy", "temp", "integrity",
                         "wm_loss", "eval_reward", "eval_food")}
        self._last_frame_t = 0.0
        self._last_hist_t = 0.0
        self._t_start = 0.0

    # ------------------------------------------------------ training thread

    def start(self, scenario: str, switches: dict, steps: int | None):
        with self.lock:
            if self.running:
                return False
            self.stop_flag = False
            self.running = True
            self.renderer = None
            self.senses_renderer = None
            self.focused_id = None
            self._snap = None
            self.history = {k: [] for k in self.history}
            self.status = {"phase": "starting", "step": 0,
                           "message": "Building the world..."}
            self._t_start = time.time()
        self.thread = threading.Thread(
            target=self._train, args=(scenario, switches, steps), daemon=True)
        self.thread.start()
        return True

    def stop(self):
        with self.lock:
            self.stop_flag = True
            self.status["message"] = "Stopping after this step..."

    def _train(self, scenario: str, switches: dict, steps: int | None):
        try:
            preset = SCENARIOS.get(scenario, SCENARIOS["bio_cpu"])[0]
            cfg = load_config(preset)
            self.sensor_cfg = cfg.sensor
            self.colony_mode = (scenario == "colony")
            if not self.colony_mode:
                for key, on in switches.items():   # per-switch overrides
                    if key in ("neuromod", "sleep", "dev", "metab"):
                        getattr(cfg, key).enabled = bool(on)
            if steps:
                cfg.train.total_steps = int(steps)
            cfg.train.out_dir = "runs/gui"
            cfg.train.gui_every = cfg.train.gui_every or 6
            cfg.train.log_every = min(cfg.train.log_every, 100)
            cfg.train.eval_every = min(cfg.train.eval_every, 2500)
            with self.lock:
                self.config = {"scenario": scenario, "preset": preset,
                               "requested_device": cfg.train.device,
                               "colony": self.colony_mode,
                               "total_steps": cfg.train.total_steps}
            if self.colony_mode:
                from ..colony.run import run_colony
                run_colony(cfg, verbose=False, on_step=self._on_step_colony,
                           should_stop=lambda: self.stop_flag)
            else:
                from ..train import run
                run(cfg, verbose=False, on_step=self._on_step,
                    should_stop=lambda: self.stop_flag)
            msg = "Stopped." if self.stop_flag else "Run complete."
            self._set_phase("done", msg)
        except Exception as e:  # surface the error in the UI, don't crash silently
            traceback.print_exc()
            self._set_phase("error", f"Error: {e}")
        finally:
            with self.lock:
                self.running = False
            if self.renderer is not None:
                try:
                    self.renderer.close()
                except Exception:
                    pass

    def _set_phase(self, phase, message):
        with self.lock:
            self.status["phase"] = phase
            self.status["message"] = message

    # ------------------------------------------------------ live callback

    def _on_step(self, s: dict):
        now = time.time()
        info, m, ev = s["info"], s.get("metrics", {}), s.get("eval", {})
        status = {
            "phase": "asleep" if s.get("asleep") else "awake",
            "step": s["step"],
            "sps": round(s.get("sps", 0.0), 1),
            "energy": round(float(info.get("energy", 0)), 3),
            "temp": round(float(info.get("temp", 0)), 3),
            "integrity": round(float(info.get("integrity", 0)), 3),
            "food_total": int(info.get("food_eaten", 0)),
            "offspring": int(info.get("offspring", 0)),
            "wm_loss": round(float(m.get("loss", 0.0)), 3),
            "eval_reward": round(float(ev.get("eval_reward", 0.0)), 1),
            "eval_food": round(float(ev.get("eval_food", 0.0)), 2),
            "message": self._narrate(s, info),
        }
        with self.lock:
            self.status.update(status)
        # frame: throttle to ~10 fps regardless of training speed
        if now - self._last_frame_t > 0.1:
            self._render(s["env"], info)
            self._render_senses(s.get("obs"))
            self._last_frame_t = now
        # history: one point every ~0.3 s, capped
        if now - self._last_hist_t > 0.3:
            with self.lock:
                h = self.history
                h["step"].append(s["step"])
                h["reward"].append(float(m.get("ep_reward", m.get("reward", 0))))
                h["energy"].append(float(info.get("energy", 0)))
                h["temp"].append(float(info.get("temp", 0)))
                h["integrity"].append(float(info.get("integrity", 0)))
                h["wm_loss"].append(float(m.get("loss", 0)))
                h["eval_reward"].append(float(ev.get("eval_reward", 0)))
                h["eval_food"].append(float(ev.get("eval_food", 0)))
                for v in h.values():
                    if len(v) > 500:
                        del v[0]
            self._last_hist_t = now

    def _render(self, env, info):
        try:
            if self.renderer is None:
                from ..viz.render import ArenaRenderer
                self.renderer = ArenaRenderer(env)
            arr = self.renderer.render(info)
            buf = io.BytesIO()
            mpimg.imsave(buf, arr, format="png")
            with self.lock:
                self.frame_png = buf.getvalue()
        except Exception:
            traceback.print_exc()

    # ------------------------------------------------------ colony callback

    def _on_step_colony(self, s: dict):
        now = time.time()
        env = s["env"]
        st = s["stats"]
        m = s.get("metrics", {})
        foc = self._resolve_focus(env)
        # the body bars show the INSPECTED individual's own stats (not the
        # colony average); fall back to colony means only if nobody is alive.
        if foc is not None:
            h = foc.homeostasis
            energy, temp, integ = h.energy, h.temp, h.integrity
            focus = {"id": foc.id, "generation": foc.generation, "age": foc.age,
                     "energy": round(h.energy, 2), "temp": round(h.temp, 2),
                     "integrity": round(h.integrity, 2)}
        else:
            energy = st.get("mean_energy", 0.0)
            temp = st.get("mean_temp", 0.5)
            integ = st.get("mean_integrity", 1.0)
            focus = None
        status = {
            "phase": "colony",
            "step": s["step"],
            "sps": round(s.get("sps", 0.0), 1),
            "population": st["population"],
            "births": st["births"],
            "deaths": st["deaths"],
            "generation": st["max_generation"],
            "energy": round(energy, 3),         # inspected creature
            "temp": round(temp, 3),
            "integrity": round(integ, 3),
            "mean_energy": round(st.get("mean_energy", 0.0), 3),  # community
            "wm_loss": round(float(m.get("loss", 0.0)), 3),
            "message": self._narrate_colony(st),
            "focus": focus,
        }
        with self.lock:
            self.status.update(status)
        if now - self._last_frame_t > 0.1:
            self._render_colony(env, foc.id if foc else None)
            if foc is not None:
                self._render_senses(foc.observe())
            with self.lock:
                self._snap = {"S": env.cfg.arena_size,
                              "pts": [(c.id, float(c.pos[0]), float(c.pos[1]))
                                      for c in env.living]}
            self._last_frame_t = now
        if now - self._last_hist_t > 0.3:
            with self.lock:
                h = self.history
                h["step"].append(s["step"])
                h["reward"].append(float(st["population"]))     # chart 1
                h["energy"].append(float(st.get("mean_energy", 0)))
                h["wm_loss"].append(float(m.get("loss", 0)))
                for v in h.values():
                    if len(v) > 500:
                        del v[0]
            self._last_hist_t = now

    def _render_colony(self, env, focus_id=None):
        try:
            if self.renderer is None:
                from ..colony.render import ColonyRenderer
                self.renderer = ColonyRenderer(env)
            arr = self.renderer.render(focus_id=focus_id)
            buf = io.BytesIO()
            mpimg.imsave(buf, arr, format="png")
            with self.lock:
                self.frame_png = buf.getvalue()
        except Exception:
            traceback.print_exc()

    def _resolve_focus(self, env):
        """The creature to inspect: the user's click if still alive, else the
        oldest (most-established) creature as a sensible default."""
        living = env.living
        if not living:
            return None
        by_id = {c.id: c for c in living}
        if self.focused_id in by_id:
            return by_id[self.focused_id]
        return max(living, key=lambda c: c.age)

    def _render_senses(self, obs):
        try:
            if obs is None or self.sensor_cfg is None:
                return
            if self.senses_renderer is None:
                from .senses import SensesRenderer
                self.senses_renderer = SensesRenderer(self.sensor_cfg)
            png = self.senses_renderer.png(obs)
            with self.lock:
                self.senses_png = png
        except Exception:
            traceback.print_exc()

    def focus_at(self, fx: float, fy: float):
        """Map a click (fractions of the arena image) to the nearest creature."""
        snap = self._snap
        if not snap or not snap["pts"]:
            return
        S = snap["S"]
        wx, wy = fx * S, (1.0 - fy) * S
        best = min(snap["pts"],
                   key=lambda p: (p[1] - wx) ** 2 + (p[2] - wy) ** 2)
        self.focused_id = best[0]

    def _narrate_colony(self, st) -> str:
        if st["population"] <= 0:
            return "The colony has died out."
        parts = [f"{st['population']} individuals alive, each with its own mind"]
        if st["max_generation"] > 0:
            parts.append(f"now on generation {st['max_generation'] + 1} "
                         f"(offspring inherited a parent's brain)")
        if st.get("mean_energy", 1) < 0.3:
            parts.append("food is scarce — hard times, some are starving")
        elif st["births"] > st["deaths"]:
            parts.append("well-fed and breeding — the community is growing")
        return "; ".join(parts) + "."

    def _narrate(self, s, info) -> str:
        """Plain-language description of what the body is doing right now."""
        if s.get("asleep"):
            return ("Sleeping: replaying the day's most vivid memories to "
                    "consolidate the world model (and dreaming).")
        e, integ = info.get("energy", 1), info.get("integrity", 1)
        if e < 0.25:
            return "Starving -- energy is low, so it should be hunting for food."
        if integ < 0.4:
            return "Hurt -- integrity is low after a collision; it should be " \
                   "careful."
        if info.get("food_eaten", 0):
            return "Just ate -- energy restored; that is a moment of relief."
        return "Awake and foraging: predicting its senses and keeping its body " \
               "in the safe zone."

    # ------------------------------------------------------ snapshots

    def state_json(self) -> bytes:
        with self.lock:
            payload = {"running": self.running, "device": self.device,
                       "mode": "colony" if self.colony_mode else "single",
                       "status": dict(self.status), "config": dict(self.config),
                       "history": {k: list(v) for k, v in self.history.items()},
                       "scenarios": {k: v[1] for k, v in SCENARIOS.items()},
                       "switches": SWITCHES}
        return json.dumps(payload).encode()

    def frame_bytes(self) -> bytes:
        with self.lock:
            return self.frame_png

    def senses_bytes(self) -> bytes:
        with self.lock:
            return self.senses_png


STATE = GuiState()


# client-disconnect errors (very common on Windows: browsers open then abort
# speculative connections). These are harmless -- swallow them quietly instead
# of letting http.server print an alarming traceback for each.
_CONN_ERRORS = (BrokenPipeError, ConnectionError)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except _CONN_ERRORS:
            self.close_connection = True

    def _send(self, code, body, ctype):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except _CONN_ERRORS:
            self.close_connection = True  # client went away mid-response

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send(200, STATE.state_json(), "application/json")
        elif path == "/api/frame.png":
            self._send(200, STATE.frame_bytes(), "image/png")
        elif path == "/api/senses.png":
            self._send(200, STATE.senses_bytes(), "image/png")
        elif path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b"{}"
        try:
            req = json.loads(body or b"{}")
        except Exception:
            req = {}
        if self.path == "/api/start":
            started = STATE.start(req.get("scenario", "bio_cpu"),
                                  req.get("switches", {}), req.get("steps"))
            self._send(200, json.dumps({"started": started}).encode(),
                       "application/json")
        elif self.path == "/api/stop":
            STATE.stop()
            self._send(200, b'{"ok":true}', "application/json")
        elif self.path == "/api/focus":
            try:
                STATE.focus_at(float(req.get("fx", 0)), float(req.get("fy", 0)))
            except Exception:
                pass
            self._send(200, b'{"ok":true}', "application/json")
        else:
            self._send(404, b"not found", "text/plain")


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True):
    try:
        import torch
        STATE.device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        STATE.device = "cpu"
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"\n  Embodied-agent dashboard: {url}")
    print(f"  Device detected: {STATE.device.upper()}"
          f"{'  (GPU)' if STATE.device == 'cuda' else '  (no GPU -- CPU mode)'}")
    print("  Open the link in your browser, pick a scenario, press Start.")
    print("  Press Ctrl+C here to quit.\n")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
        STATE.stop()
        httpd.shutdown()
