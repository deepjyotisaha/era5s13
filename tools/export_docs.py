"""Copy the published pages into docs/ as standalone local HTML files.

The published artifacts are page fragments: the viewer wraps them in <html>/<head> with a UTF-8
charset, a viewport and a small reset. Opened straight from disk, a fragment has none of that
(so characters like · and → can show up garbled), so each copy is given the same wrapper here.

Usage (from the assignment folder, after tools/make_walkthrough.py):
    python tools/export_docs.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
DOCS = os.path.join(ROOT, "docs")

PAGES = [  # (source, docs file name)
    (os.path.join(ROOT, "session13_walkthrough.html"), "session13_walkthrough.html"),
    (os.path.join(ROOT, "..", "assignment_research", "session13_report.html"), "session13_report.html"),
    (os.path.join(ROOT, "assignment_plan.html"), "assignment_plan.html"),
]

# What the artifact viewer adds around every page, so the local copy looks the same.
RESET = "<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>"


def standalone(fragment):
    if re.match(r"\s*<!doctype", fragment, re.I):
        return fragment
    body = re.sub(r'^\s*<meta charset="[^"]*">\s*', "", fragment)
    # Everything up to the end of the first <style> block is head material.
    cut = body.find("</style>")
    head, rest = (body[:cut + 8], body[cut + 8:]) if cut >= 0 else ("", body)
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"{RESET}\n{head}\n</head>\n<body>\n{rest}\n</body>\n</html>\n")


os.makedirs(DOCS, exist_ok=True)
for src, name in PAGES:
    text = open(src, encoding="utf-8").read()
    out = os.path.join(DOCS, name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(standalone(text))
    print(f"docs/{name:<28} {os.path.getsize(out) / 1e3:7.0f} kB  <- {os.path.relpath(src, ROOT)}")
