"""Network vs decompression: read level-4 shard files of one channel as raw
bytes, then decode the same data from a local in-memory copy."""
import sys, time, json, os
from pathlib import Path
import numpy as np, zarr
from zarr.storage import MemoryStore

src = Path(sys.argv[1]); level = sys.argv[2]; ch = sys.argv[3]
meta = json.loads((src / level / "zarr.json").read_text())
print("codecs:", json.dumps(meta["codecs"])[:400])
files = sorted(p for p in (src / level / "c" / ch).rglob("*") if p.is_file())
t = time.time(); blobs = {p: p.read_bytes() for p in files}; dt = time.time() - t
size = sum(len(b) for b in blobs.values())
print(f"raw read of {len(files)} shard files, {size/1e6:.0f} MB on disk: {dt:.1f} s ({size/1e6/dt:.0f} MB/s)")
t = time.time(); [p.read_bytes() for p in files]; print(f"raw re-read: {time.time()-t:.1f} s")
# Same bytes in memory: decode time only.
store = MemoryStore()
import asyncio
from zarr.core.buffer import default_buffer_prototype
proto = default_buffer_prototype()
async def fill():
    await store.set(f"zarr.json", proto.buffer.from_bytes((src / "zarr.json").read_bytes()))
    await store.set(f"{level}/zarr.json", proto.buffer.from_bytes((src / level / "zarr.json").read_bytes()))
    for p, b in blobs.items():
        key = p.relative_to(src).as_posix()
        await store.set(key, proto.buffer.from_bytes(b))
asyncio.run(fill())
arr = zarr.open_array(store, path=level, mode="r")
t = time.time(); a = arr[int(ch)]; dt = time.time() - t
print(f"decode from memory: {dt:.1f} s for {a.nbytes/1e6:.0f} MB ({a.nbytes/1e6/dt:.0f} MB/s); ratio {a.nbytes/size:.1f}x")
