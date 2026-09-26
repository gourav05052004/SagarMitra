from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'bundle_manifest.json').read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
for n,item in enumerate(manifest['files'],1):
    path=ROOT/item['bundled_path']
    assert path.is_file(),path
    assert path.stat().st_size==item['bytes'],path
    assert sha(path)==item['sha256'],path
print(f"PASS: {len(manifest['files'])} files verified; validation AP record={manifest['validation_AP']:.6f}")
