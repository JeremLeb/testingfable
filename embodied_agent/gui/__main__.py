"""Launch the live dashboard:  python -m embodied_agent.gui"""
from __future__ import annotations

import argparse

from .server import serve


def main():
    ap = argparse.ArgumentParser(description="Live observation dashboard.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true",
                    help="do not auto-open a browser window")
    args = ap.parse_args()
    serve(args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
