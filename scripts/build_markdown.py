import os
import pathlib

import markdown
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "blog" / "posts"
OUTPUT_DIR = ROOT / "blog" / "html"
TEMPLATE_FILE = ROOT / "blog" / "post_template.html"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with TEMPLATE_FILE.open("r", encoding="utf-8") as f:
  template = f.read()

def parse_front_matter(text: str):
    text = text.lstrip()
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    _, front, body = parts
    meta = yaml.safe_load(front) or {}
    return meta, body

def render_post(md_path: pathlib.Path):
    raw = md_path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)

    html_body = markdown.markdown(
    body,
    extensions=[
        "fenced_code",
        "tables",
        "toc",
        "pymdownx.arithmatex",
    ],
    extension_configs={
        "pymdownx.arithmatex": {
            "generic": True,
        }
    }
)


    title = meta.get("title", md_path.stem)
    date = meta.get("date", "")

    html = (
        template
        .replace("{{title}}", title)
        .replace("{{date}}", date)
        .replace("{{content}}", html_body)
    )

    out_path = OUTPUT_DIR / f"{md_path.stem}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)}")

def main():
    for md_file in sorted(INPUT_DIR.glob("*.md")):
        render_post(md_file)

if __name__ == "__main__":
    main()
