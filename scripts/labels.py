"""Parse open/open_labels.xlsx without openpyxl. Keys use Latin 'X-'; duplicates X-47→X-4, X-48→X-5 merged."""
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DUPLICATES = {"X-47": "X-4", "X-48": "X-5"}


def parse_time(text: str) -> float | None:
    m = re.match(r"\s*(?:(\d+)\s*минут\w*\s*)?(\d+)\s*секунд", text or "")
    return None if not m else float(int(m.group(1) or 0) * 60 + int(m.group(2)))


def load_labels(path: Path) -> dict[str, list[float]]:
    z = zipfile.ZipFile(path)
    labels: dict[str, list[float]] = {}
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS + "row"):
        cells = {c.get("r")[0]: "".join(x.text or "" for x in c.iter(NS + "t")) for c in row.findall(NS + "c")}
        name = cells.get("A", "").replace("Х", "X").strip()
        if not name.startswith("X-"):
            continue
        key = DUPLICATES.get(name, name)
        labels.setdefault(key, [])
        t = parse_time(cells.get("B", ""))
        if t is not None and t not in labels[key]:
            labels[key].append(t)
    return labels


def video_key(filename: str) -> str:
    return Path(filename).stem.replace("Х", "X")
