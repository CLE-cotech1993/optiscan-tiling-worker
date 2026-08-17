import os
import io
import json
import math
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tifffile
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = Flask(__name__)
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
            supabase.storage.from_("tiles").upload(
                path, buf.read(), {"content-type": "image/jpeg", "upsert": "true"}
            )
    return {"level": level_idx, "width": iw, "height": il, "tile_size": TILE_SIZE, "cols": cols, "rows": rows}


@app.route("/", methods=["GET"])
def health():
    return "tiling worker ok"


@app.route("/tile", methods=["POST"])
def tile_slide():
    payload = request.get_json(force=True)
    record = payload.get("record", {})
    slide_id = record.get("id")
    file_name = record.get("file_name")

    if not slide_id or not file_name:
        return jsonify({"error": "missing id or file_name"}), 400

    local_path = None
    try:
        supabase.table("slides").update({"status": "processing"}).eq("id", slide_id).execute()

        raw_bytes = supabase.storage.from_("raw-slides").download(file_name)
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as f:
            f.write(raw_bytes)
            local_path = f.name

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
        return jsonify({"status": "ok", "levels": levels_info})

    except Exception as e:
        supabase.table("slides").update({"status": "error"}).eq("id", slide_id).execute()
        return jsonify({"error": str(e)}), 500

    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)