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
               "A whole COMMUNITY: many creatures share one world and one brain "
               "-- they forage the same food, collide, breed into live "
               "offspring, and evolve. (Biological switches don't apply here.)"),
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
        self.renderer = None
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
        st = s["stats"]
        m = s.get("metrics", {})
        status = {
            "phase": "colony",
            "step": s["step"],
            "sps": round(s.get("sps", 0.0), 1),
            "population": st["population"],
            "births": st["births"],
            "deaths": st["deaths"],
            "generation": st["max_generation"],
            "energy": round(st.get("mean_energy", 0.0), 3),
            "temp": round(st.get("mean_temp", 0.5), 3),
            "integrity": round(st.get("mean_integrity", 1.0), 3),
            "wm_loss": round(float(m.get("loss", 0.0)), 3),
            "message": self._narrate_colony(st),
        }
        with self.lock:
            self.status.update(status)
        if now - self._last_frame_t > 0.1:
            self._render_colony(s["env"])
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

    def _render_colony(self, env):
        try:
            if self.renderer is None:
                from ..colony.render import ColonyRenderer
                self.renderer = ColonyRenderer(env)
            arr = self.renderer.render()
            buf = io.BytesIO()
            mpimg.imsave(buf, arr, format="png")
            with self.lock:
                self.frame_png = buf.getvalue()
        except Exception:
            traceback.print_exc()

    def _narrate_colony(self, st) -> str:
        if st["population"] <= 0:
            return "The colony has died out."
        parts = [f"{st['population']} creatures alive"]
        if st["max_generation"] > 0:
            parts.append(f"now on generation {st['max_generation'] + 1}")
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


STATE = GuiState()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send(200, STATE.state_json(), "application/json")
        elif path == "/api/frame.png":
            self._send(200, STATE.frame_bytes(), "image/png")
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
