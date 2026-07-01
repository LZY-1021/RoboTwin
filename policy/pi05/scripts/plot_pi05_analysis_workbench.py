#!/usr/bin/env python3
"""Build a unified PI0.5/RoboTwin analysis workbench.

This is a lightweight front door for the existing trace visualizations.  It
does not recompute attention or model internals; it links already-generated
HTML reports into a single tabbed page so one trace can be inspected from one
place.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Page:
    key: str
    title: str
    description: str
    path: Path | None
    placeholder: str


def existing_index(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = path.resolve()
    if path.is_file():
        return path
    index = path / "index.html"
    if index.exists():
        return index
    return None


def rel_src(target: Path, out_dir: Path) -> str:
    return os.path.relpath(target.resolve(), out_dir.resolve())


def optional_path(value: str | None) -> Path | None:
    if value is None or value.strip() == "":
        return None
    return Path(value).expanduser()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_page(args: argparse.Namespace, pages: list[Page]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_pages:
        pages = copy_available_pages(args.out_dir, pages)
    manifest = {
        "title": args.title,
        "trace": str(args.trace) if args.trace else None,
        "episode": args.episode,
        "infer": args.infer,
        "pages": [
            {
                "key": page.key,
                "title": page.title,
                "description": page.description,
                "path": str(page.path) if page.path else None,
                "available": page.path is not None,
            }
            for page in pages
        ],
    }
    (args.out_dir / "workbench_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    page_json = json.dumps(manifest, ensure_ascii=False)
    html_text = HTML_TEMPLATE.replace("__WORKBENCH_DATA__", page_json)
    iframe_sections = []
    nav_buttons = []
    for idx, page in enumerate(pages):
        active = " active" if idx == 0 else ""
        nav_buttons.append(
            f'<button class="tab-button{active}" data-tab="{html.escape(page.key)}">'
            f"{html.escape(page.title)}</button>"
        )
        if page.path is not None:
            src = html.escape(rel_src(page.path, args.out_dir))
            body = (
                f'<iframe title="{html.escape(page.title)}" '
                f'src="{src}" loading="lazy"></iframe>'
            )
        else:
            body = (
                '<div class="placeholder">'
                f"<h2>{html.escape(page.title)}</h2>"
                f"<p>{html.escape(page.placeholder)}</p>"
                "</div>"
            )
        iframe_sections.append(
            f'<section class="tab-panel{active}" id="{html.escape(page.key)}">'
            f'<div class="panel-head"><h2>{html.escape(page.title)}</h2>'
            f'<p>{html.escape(page.description)}</p></div>{body}</section>'
        )
    html_text = html_text.replace("__TAB_BUTTONS__", "\n".join(nav_buttons))
    html_text = html_text.replace("__TAB_PANELS__", "\n".join(iframe_sections))
    (args.out_dir / "index.html").write_text(html_text, encoding="utf-8")


def copy_available_pages(out_dir: Path, pages: list[Page]) -> list[Page]:
    pages_root = out_dir / "pages"
    pages_root.mkdir(parents=True, exist_ok=True)
    copied_pages: list[Page] = []
    for page in pages:
        if page.path is None:
            copied_pages.append(page)
            continue
        source_root = page.path.parent
        target_root = pages_root / page.key
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.copytree(source_root, target_root)
        copied_pages.append(
            Page(
                key=page.key,
                title=page.title,
                description=page.description,
                path=target_root / "index.html",
                placeholder=page.placeholder,
            )
        )
    return copied_pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title", default="PI0.5 Block Analysis Workbench")
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--infer", type=int, default=0)
    parser.add_argument("--attention-dir", type=Path, default=None)
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    parser.add_argument("--overlay-dir", action="append", default=[])
    parser.add_argument("--mlp-kv-dir", type=Path, default=None)
    parser.add_argument("--qk-dir", type=Path, default=None)
    parser.add_argument(
        "--no-copy-pages",
        dest="copy_pages",
        action="store_false",
        help="Link to original report directories instead of copying reports under out-dir/pages.",
    )
    parser.add_argument(
        "--note",
        default="Use this page as a common front door for existing PI0.5 trace visualizations.",
    )
    parser.set_defaults(copy_pages=True)
    args = parser.parse_args()

    overlay_pages = []
    for idx, item in enumerate(args.overlay_dir):
        path = existing_index(optional_path(item))
        overlay_pages.append(
            Page(
                key=f"overlay_{idx}",
                title=f"Overlay / Diff {idx + 1}",
                description="Existing obs/prefix/KV/MLP/attention overlay report.",
                path=path,
                placeholder=f"No index.html found for overlay directory: {item}",
            )
        )

    pages = [
        Page(
            key="overview",
            title="Overview",
            description=args.note,
            path=None,
            placeholder=overview_text(args),
        ),
        Page(
            key="attention",
            title="Attention Explorer",
            description="Block-level denoise attention by step, layer, head, and q token.",
            path=existing_index(args.attention_dir),
            placeholder="Generate it with plot_block_attention_d3_explorer.py and pass --attention-dir.",
        ),
        Page(
            key="trajectory",
            title="Action-q Trajectory",
            description="Per-head action-query heatmaps and centroid trajectories.",
            path=existing_index(args.trajectory_dir),
            placeholder="Generate it with plot_denoise_head_query_attention.py and pass --trajectory-dir.",
        ),
        Page(
            key="mlp_kv",
            title="MLP / KV / Obs Diff",
            description="Patch/block overlays for obs, prefix embedding, KV, MLP input/output, and attention diffs.",
            path=existing_index(args.mlp_kv_dir),
            placeholder="Pass --mlp-kv-dir or --overlay-dir for existing patch overlay reports.",
        ),
        *overlay_pages,
        Page(
            key="qk_logits",
            title="QK / Logits Top-k",
            description="Planned tab for top-k block K-column approximation of real logits or Δlogits.",
            path=existing_index(args.qk_dir),
            placeholder=(
                "This tab needs traces containing Q, K, or pre-softmax logits. "
                "Current attention traces only contain softmax attention weights, so QK/logits analysis "
                "will be enabled after adding a Q/K/logits trace path."
            ),
        ),
    ]
    write_page(args, pages)
    print(f"Wrote {args.out_dir / 'index.html'}")


def overview_text(args: argparse.Namespace) -> str:
    trace = str(args.trace) if args.trace else "not specified"
    return (
        f"Trace: {trace}. Episode {args.episode}, inference {args.infer}. "
        "Use the tabs above to inspect attention, action-query trajectories, overlay diffs, "
        "and the future QK/logits top-k analysis in one page."
    )


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PI0.5 Block Analysis Workbench</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #20242b;
      --muted: #667085;
      --line: #d8dde6;
      --blue: #1463ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, "Helvetica Neue", sans-serif;
    }
    header {
      padding: 18px 22px 12px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 24px;
      letter-spacing: 0;
    }
    .subtle {
      color: var(--muted);
      line-height: 1.45;
      font-size: 14px;
    }
    nav {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      padding: 12px 22px;
      background: #eef2f8;
      border-bottom: 1px solid var(--line);
    }
    .tab-button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 12px;
      cursor: pointer;
      font-size: 14px;
    }
    .tab-button.active {
      border-color: var(--blue);
      color: var(--blue);
      box-shadow: inset 0 0 0 1px var(--blue);
    }
    main {
      padding: 14px;
    }
    .tab-panel {
      display: none;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-height: calc(100vh - 150px);
    }
    .tab-panel.active {
      display: block;
    }
    .panel-head {
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    .panel-head h2 {
      margin: 0 0 4px;
      font-size: 18px;
    }
    .panel-head p {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }
    iframe {
      display: block;
      width: 100%;
      height: calc(100vh - 220px);
      min-height: 760px;
      border: 0;
      background: #fff;
    }
    .placeholder {
      margin: 22px;
      padding: 18px;
      border: 1px dashed #b9c2d0;
      border-radius: 8px;
      background: #fafcff;
      color: var(--muted);
      line-height: 1.6;
      max-width: 920px;
    }
    .placeholder h2 {
      color: var(--ink);
      margin-top: 0;
    }
    code {
      background: #eef2f8;
      padding: 2px 4px;
      border-radius: 4px;
    }
  </style>
</head>
<body>
  <header>
    <h1>PI0.5 Block Analysis Workbench</h1>
    <div class="subtle" id="summary"></div>
  </header>
  <nav>
    __TAB_BUTTONS__
  </nav>
  <main>
    __TAB_PANELS__
  </main>
  <script>
    const WORKBENCH = __WORKBENCH_DATA__;
    document.getElementById("summary").textContent =
      `${WORKBENCH.title} | trace=${WORKBENCH.trace || "not specified"} | episode=${WORKBENCH.episode} | infer=${WORKBENCH.infer}`;
    const buttons = Array.from(document.querySelectorAll(".tab-button"));
    const panels = Array.from(document.querySelectorAll(".tab-panel"));
    function activate(key) {
      buttons.forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === key));
      panels.forEach((panel) => panel.classList.toggle("active", panel.id === key));
      const url = new URL(window.location.href);
      url.hash = key;
      history.replaceState(null, "", url);
    }
    buttons.forEach((btn) => btn.addEventListener("click", () => activate(btn.dataset.tab)));
    const initial = window.location.hash ? window.location.hash.slice(1) : buttons[0]?.dataset.tab;
    if (initial && document.getElementById(initial)) {
      activate(initial);
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
