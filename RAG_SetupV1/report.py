from reportlab.pdfgen import canvas
from datetime import datetime

def save_report(entries, out_path="trip_report.pdf"):
    c = canvas.Canvas(out_path)
    c.setFont("Helvetica", 14)
    c.drawString(50, 800, "Trip Report")
    c.setFont("Helvetica", 10)
    y = 780
    for ts_ms, lvl, msg in entries[-75:]:
        ts = datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        c.drawString(50, y, f"{ts} | Level {lvl} | {msg}")
        y -= 14
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = 800
    c.save()
    return out_path