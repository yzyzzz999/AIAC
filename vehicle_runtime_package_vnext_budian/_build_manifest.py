"""Generate DEPLOYMENT_PACKAGE_MANIFEST.csv"""
import csv, hashlib, os
from pathlib import Path

PKG = Path(__file__).parent
rows = []

def hash_file(p):
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:12]

for root, dirs, files in os.walk(PKG):
    dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git')]
    for fname in sorted(files):
        fpath = Path(root) / fname
        rel = fpath.relative_to(PKG)
        ext = fpath.suffix
        size = fpath.stat().st_size
        rows.append({
            "path": str(rel),
            "type": ext,
            "size_bytes": size,
            "sha256_12": hash_file(fpath),
        })

out = PKG / "DEPLOYMENT_PACKAGE_MANIFEST.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["path","type","size_bytes","sha256_12"])
    w.writeheader()
    w.writerows(rows)

print(f"Manifest written: {len(rows)} files → {out}")
