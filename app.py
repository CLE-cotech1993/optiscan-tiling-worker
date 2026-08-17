import os
import io
import json
import math
import tempfile
import threading
import time
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tifffile
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
TILE_SIZE = 512


def get_pyramid_pages(tf):
    pages = [p for p in tf.pages if p.is_tiled]
    pages = sorted(pages, key=lambda p: -p.imagewidth)
    return pages


def tile_page(page, level_idx, slide_id):
    iw, il = page.imagewidth, page.imagelength
    tw, tl = page.tilewidth, page.tilelength
    ta = math.ceil(iw / tw)
    fh = page.parent.filehandle
    cols = math.ceil(iw / TILE_SIZE)
    rows = math.ceil(il / TILE_SIZE)

    for ty_out in range(rows):
        for tx_out in range(cols):
            x0, y0 = tx_out * TILE_SIZE, ty_out * TILE_SIZE
            x1, y1 = min(iw, x0 + TILE_SIZE), min(il, y0 + TILE_SIZE)
            stx0, sty0 = x0 // tw, y0 // tl
            stx1, sty1 = (x1 - 1) // tw, (y1 - 1) // tl
            out = np.full((y1 - y0, x1 - x0, 3), 255, dtype=np.uint8)
            for sty in range(sty0, sty1 + 1):
                for stx in range(stx0, stx1 + 1):
                    idx = sty * ta + stx
                    if idx >= len(page.dataoffsets):
                        continue
                    fh.seek(page.dataoffsets[idx])
                    data = fh.read(page.databytecounts[idx])
                    tile = page.decode(data, idx, jpegtables=page.jpegtables)[0]
                    tile = np.asarray(tile).reshape(tl, tw, 3)
                    gx0, gy0 = stx * tw, sty * tl
                    sx0 = max(x0, gx0); sy0 = max(y0, gy0)
                    sx1 = min(x1, gx0 + tw); sy1 = min(y1, gy0 + tl)
                    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = tile[sy0 - gy0:sy1 - gy0, sx0 - gx0:sx1 - gx0]
            buf = io.BytesIO()
            Image.fromarray(out).save(buf, format="JPEG", quality=88)
            buf.seek(0)
            path = f"{slide_id}/level{level_idx}/{tx_out}_{ty_out}.jpg"
            tile_bytes = buf.read()
            for attempt in range(4):
                try:
                    supabase.storage.from_("tiles").upload(
                        path, tile_bytes, {"content-type": "image/jpeg", "upsert": "true"}
                    )
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(1.5 * (attempt + 1))
    return {"level": level_idx, "width": iw, "height": il, "tile_size": TILE_SIZE, "cols": cols, "rows": rows}


def process_tiling(record):
    slide_id = record.get("id")
    file_name = record.get("file_name")
    local_path = None
    try:
        supabase.table("slides").update({"status": "processing"}).eq("id", slide_id).execute()

        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as f:
            local_path = f.name

        download_url = f"{SUPABASE_URL}/storage/v1/object/raw-slides/{file_name}"
        headers = {"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        with requests.get(download_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as out_f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    out_f.write(chunk)

        tf = tifffile.TiffFile(local_path)
        pages = get_pyramid_pages(tf)
        if not pages:
            raise ValueError("No tiled pyramid levels found in this file")

        levels_info = []
        for i, page in enumerate(pages):
            levels_info.append(tile_page(page, i, slide_id))

        manifest = {"levels": levels_info, "mpp": 0.496094}
        manifest_bytes = json.dumps(manifest).encode()
        supabase.storage.from_("tiles").upload(
            f"{slide_id}/manifest.json", manifest_bytes,
            {"content-type": "application/json", "upsert": "true"}
        )

        supabase.table("slides").update({"status": "tiled"}).eq("id", slide_id).execute()

    except Exception as e:
        import traceback
        print("TILING ERROR:", str(e))
        traceback.print_exc()
        supabase.table("slides").update({"status": "error"}).eq("id", slide_id).execute()

    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)


@app.route("/", methods=["GET"])
def health():
    return "tiling worker ok"


@app.route("/tile", methods=["POST", "OPTIONS"])
def tile_slide():
    if request.method == "OPTIONS":
        return ('', 204)

    payload = request.get_json(force=True)
    record = payload.get("record", {})
    slide_id = record.get("id")
    file_name = record.get("file_name")

    if not slide_id or not file_name:
        return jsonify({"error": "missing id or file_name"}), 400

    thread = threading.Thread(target=process_tiling, args=(record,), daemon=True)
    thread.start()

    return jsonify({"status": "accepted", "slide_id": slide_id}), 202


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)