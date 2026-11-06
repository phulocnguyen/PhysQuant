# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Browse episode MP4s in a local browser, without Isaac Sim or extra dependencies."""

import argparse
import errno
import json
import re
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

PAGE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Benchmark episode viewer</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body {background:#111827;color:#e5e7eb;font:16px system-ui;margin:24px auto;padding:0 20px;max-width:1000px}
h1 {font-size:24px} .row {display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}
select,input,button {font:inherit;padding:8px;border-radius:6px;border:1px solid #64748b}
#episode {flex:1;min-width:200px;max-width:100%} #search {flex:1}
video {display:block;width:min(100%,640px);max-height:65vh;background:black;margin:auto}
small {color:#94a3b8;overflow-wrap:anywhere} a {color:#93c5fd}
</style>
<h1>Benchmark episode viewer</h1>
<div class="row"><input id="search" placeholder="Filter by episode, prompt, or result" aria-label="Filter episodes">
<span id="count"></span></div>
<div class="row"><button id="prev">Previous</button><select id="episode" aria-label="Episode"></select>
<button id="next">Next</button></div>
<div class="row"><label for="camera">Camera</label><select id="camera"></select>
<label><input id="loop" type="checkbox" checked> Loop</label><a id="download" download>Download MP4</a></div>
<p id="prompt"></p><small id="detail"></small>
<div class="row"></div><video id="video" controls loop playsinline preload="metadata"></video>
<p id="message" role="status"></p>
<script>
const $ = id => document.getElementById(id);
let entries = [], filtered = [];
function changeVideo(keepTime) {
  const entry = entries[Number($('episode').value)];
  if (!entry) return;
  const time = keepTime ? $('video').currentTime : 0;
  const playing = !$('video').paused;
  const url = entry.cameras[$('camera').value];
  $('video').onloadedmetadata = () => {
    $('video').currentTime = Math.min(time, Math.max(0, $('video').duration - 0.01));
    if (playing) $('video').play().catch(() => {});
  };
  $('message').textContent = '';
  $('video').src = url; $('download').href = url;
}
function changeEpisode() {
  const entry = entries[Number($('episode').value)];
  if (!entry) return;
  const previousCamera = $('camera').value;
  $('camera').replaceChildren(...Object.keys(entry.cameras).map(name => new Option(name, name)));
  if (previousCamera in entry.cameras) $('camera').value = previousCamera;
  $('prompt').textContent = entry.prompt || 'No prompt recorded';
  $('detail').textContent = entry.label;
  const index = filtered.indexOf(entry);
  $('prev').disabled = index <= 0; $('next').disabled = index >= filtered.length - 1;
  changeVideo(false);
}
function filter() {
  const old = $('episode').value, term = $('search').value.toLowerCase();
  filtered = entries.filter(e => (e.label + ' ' + e.prompt).toLowerCase().includes(term));
  $('episode').replaceChildren(...filtered.map(e => new Option(e.label, e.id)));
  if (filtered.some(e => String(e.id) === old)) $('episode').value = old;
  $('count').textContent = `${filtered.length} / ${entries.length} episodes`;
  if (filtered.length) changeEpisode();
  else {
    $('video').pause(); $('video').removeAttribute('src'); $('video').load();
    $('camera').replaceChildren(); $('prompt').textContent = ''; $('detail').textContent = '';
    $('download').removeAttribute('href'); $('prev').disabled = $('next').disabled = true;
    $('message').textContent = 'No matching episodes.';
  }
}
$('search').oninput = filter; $('episode').onchange = changeEpisode;
$('camera').onchange = () => changeVideo(true);
$('loop').onchange = () => $('video').loop = $('loop').checked;
$('prev').onclick = () => { $('episode').selectedIndex--; changeEpisode(); };
$('next').onclick = () => { $('episode').selectedIndex++; changeEpisode(); };
$('video').onerror = () => $('message').textContent = 'Cannot play this MP4. Try downloading it.';
fetch('/episodes').then(r => r.json()).then(data => { entries = data; filter(); })
.catch(e => $('message').textContent = 'Could not load episodes: ' + e.message);
</script></html>"""


def discover(root):
    """Group camera videos by run and episode, retaining separate retry recordings."""
    groups = {}
    for path in sorted(root.rglob("episode_*_*.mp4")):
        match = re.fullmatch(r"(episode_\d+)_(.+)\.mp4", path.name)
        if not match or not path.resolve().is_relative_to(root):
            continue
        episode, camera = match.groups()
        groups.setdefault((path.parent, episode), {})[camera] = path
    entries, files = [], []
    for (directory, episode), cameras in groups.items():
        run = directory.parent if directory.name == "videos" else directory
        try:
            metadata = json.loads((run / episode / "metadata.json").read_text())
        except (OSError, ValueError):
            metadata = {}
        result = "success" if metadata.get("success") else "failed"
        if not metadata.get("finalized"):
            result = "unfinalized / unknown"
        urls = {}
        for camera in sorted(cameras, key=lambda c: ({"top": 0, "left": 1, "right": 2, "wrist": 3}.get(c, 4), c)):
            urls[camera] = f"/video/{len(files)}"
            files.append(cameras[camera])
        entries.append(
            {
                "id": len(entries),
                "label": f"{run.relative_to(root)} / {episode} — {result}",
                "prompt": metadata.get("prompt", ""),
                "cameras": urls,
            }
        )
    return entries, files


def make_handler(entries, files):
    """Serve only the indexed videos, with byte ranges for browser seeking."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_HEAD(self):
            self.respond(head=True)

        def do_GET(self):
            # Browsers cancel pending requests when switching cameras.
            with suppress(BrokenPipeError, ConnectionResetError):
                self.respond()

        def respond(self, head=False):
            route = urlsplit(self.path).path
            if route in ("/", "/episodes"):
                body = PAGE.encode() if route == "/" else json.dumps(entries).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8" if route == "/" else "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if not head:
                    self.wfile.write(body)
                return
            match = re.fullmatch(r"/video/(\d+)", route)
            if not match or int(match[1]) >= len(files):
                self.send_error(404)
                return
            try:
                stream = files[int(match[1])].open("rb")
            except OSError:
                self.send_error(404)
                return
            with stream:
                size = stream.seek(0, 2)
                start, end = 0, size - 1
                requested = self.headers.get("Range")
                if requested:
                    bounds = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                    valid = bounds and any(bounds.groups())
                    if valid:
                        first, last = bounds.groups()
                        start = int(first) if first else max(0, size - int(last))
                        end = min(size - 1, int(last)) if first and last else size - 1
                        valid = start <= end and start < size
                    if not valid:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                self.send_response(206 if requested else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if requested:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if head:
                    return
                stream.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Collection root or individual run containing episode videos")
    parser.add_argument(
        "--port", type=int, default=8080, help="Local HTTP port (default: 8080); use 0 for an available port"
    )
    args = parser.parse_args()
    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Directory does not exist: {root}")
    entries, files = discover(root)
    if not files:
        parser.error(f"No episode MP4s found under {root}")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(entries, files))
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            parser.error(f"Port {args.port} is already in use. Use --port 0 for an available port or --port <number>.")
        raise
    print(
        f"Found {len(entries)} episodes and {len(files)} videos. Open http://localhost:{server.server_port}", flush=True
    )
    print("Press Ctrl+C to stop. Restart the viewer to rescan newly collected videos.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
