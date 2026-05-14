from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors


def _level_color(level: int):
    return {
        0: colors.green,
        1: colors.lightgreen,
        2: colors.orange,
        3: colors.red,
        4: colors.darkred,
    }.get(level, colors.grey)


def save_report(
    entries: Iterable[tuple[int, int, str]],
    out_path: str = "trip_report.pdf",
) -> str:
    """
    entries: iterable of (timestamp_ms, risk_level, message).
    Returns the absolute path to the PDF.
    """
    out = Path(out_path).resolve()
    doc = SimpleDocTemplate(str(out), pagesize=A4, title="Trip Safety Report")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Be Safe on the Road — Trip Report</b>", styles["Title"]))
    story.append(
        Paragraph(
            time.strftime("Generated %Y-%m-%d %H:%M:%S", time.localtime()),
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))

    data: list[list] = [["Time", "Risk", "Event / Explanation"]]
    for ts_ms, lvl, msg in entries:
        ts = time.strftime("%H:%M:%S", time.localtime(ts_ms / 1000.0))
        lvl_txt = f"L{lvl}" if lvl is not None and lvl >= 0 else "-"
        short = (msg[:110] + "…") if len(msg) > 110 else msg
        data.append([ts, lvl_txt, short])

    if len(data) == 1:
        story.append(Paragraph("No events recorded this trip.", styles["Normal"]))
    else:
        tbl = Table(data, colWidths=[60, 35, 430])
        style = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
        for i, row in enumerate(data[1:], start=1):
            lvl_txt = row[1]
            try:
                lvl_int = int(lvl_txt[1:]) if lvl_txt.startswith("L") else -1
            except ValueError:
                lvl_int = -1
            style.add("TEXTCOLOR", (1, i), (1, i), _level_color(lvl_int))
        tbl.setStyle(style)
        story.append(tbl)

    doc.build(story)
    return str(out)
