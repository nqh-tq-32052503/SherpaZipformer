import csv, json, glob, re, unicodedata, sys, io, os
from pathlib import Path

DATA_DIR = Path("./labels")   # change to where your labels are
OUT = Path("corpus.txt")
LOWERCASE = False             # set True if your ASR is lowercase

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")       # non-breaking space -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower() if LOWERCASE else s

lines = []

# CSV: any *.csv with column "text"
for f in DATA_DIR.rglob("*.csv"):
    with f.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if "text" in row and row["text"]:
                lines.append(norm(row["text"]))


# Deduplicate, keep reasonably long lines
uniq = list(dict.fromkeys([l for l in lines if len(l) >= 1]))
print(f"Collected {len(uniq)} unique lines")

with OUT.open("w", encoding="utf-8") as oh:
    for l in uniq:
        oh.write(l + "\n")

print("Wrote", OUT.resolve())
