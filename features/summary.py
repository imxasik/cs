from pathlib import Path

import pandas as pd

from cyclone.geo import haversine, get_bearing, get_cardinal_direction


# --------- CATEGORY MAPPING ---------
CAT = [
    (137, "CAT 5"),
    (114, "CAT 4"),
    (97,  "CAT 3"),
    (84,  "CAT 2"),
    (65,  "CAT 1"),
    (35,  "TS"),
    (24,  "TD"),
    (0,   "LOW"),
]


def wind_cat(wind_kt):
    """Return intensity category label from 1-min sustained wind (kt)."""
    wind_kt = float(wind_kt)
    for thr, name in CAT:
        if wind_kt >= thr:
            return name
    return "LOW"


def cat_to_phrase(cat):
    """Convert compact category code to a more descriptive phrase."""
    if cat.startswith("CAT "):
        num = cat.split()[-1]
        return f"Category {num} Cyclone"
    if cat == "TS":
        return "Tropical Storm"
    if cat == "TD":
        return "Tropical Depression"
    if cat == "LOW":
        return "Low Pressure Area"
    return cat  # fallback


def format_lat_lon(lat, lon):
    """Format latitude/longitude with N/S/E/W."""
    lat = float(lat)
    lon = float(lon)
    lat_hem = "N" if lat >= 0 else "S"
    lon_hem = "E" if lon >= 0 else "W"
    return f"{abs(lat):.1f}°{lat_hem}, {abs(lon):.1f}°{lon_hem}"


# --------- BUILD SUMMARY CONTENT (TEXT + META) ---------

def _build_summary(cyclone_name, track_obs, track_for):
    """
    Build the textual summary lines AND metadata (like updated time)
    so we can reuse them for PDF rendering.
    """
    text_lines = []
    meta = {}

    if track_obs is None or track_obs.empty:
        text_lines.append(f"Cyclone Name: {cyclone_name}")
        text_lines.append("")
        text_lines.append("No observational data available.")
        meta["text_lines"] = text_lines
        meta["updated_str"] = ""
        return meta

    # --------- CORE METADATA ---------
    last_obs = track_obs.iloc[-1]
    first_obs = track_obs.iloc[0]

    last_lat = float(last_obs["Latitude"])
    last_lon = float(last_obs["Longitude"])
    last_int_kt = float(last_obs["Intensity"])
    last_cat = wind_cat(last_int_kt)
    last_cat_phrase = cat_to_phrase(last_cat)

    peak_obs_kt = float(track_obs["Intensity"].max())
    peak_obs_cat = wind_cat(peak_obs_kt)
    peak_obs_phrase = cat_to_phrase(peak_obs_cat)

    formation_time = first_obs["tnd"]
    last_time = last_obs["tnd"]

    formation_str = formation_time.strftime("%d %b %Y, %HZ")
    updated_str = last_time.strftime("%d %b %Y, %HZ")

    # --------- CURRENT MOVEMENT ---------
    if len(track_obs) >= 2:
        prev_obs = track_obs.iloc[-2]
        coord1 = (float(prev_obs["Latitude"]), float(prev_obs["Longitude"]))
        coord2 = (last_lat, last_lon)

        time_diff_hr = (
            (last_obs["tnd"] - prev_obs["tnd"]).total_seconds() / 3600.0
        ) or 1e-6

        dist_km = haversine(coord1, coord2)
        speed_kmh = dist_km / time_diff_hr
        bearing = get_bearing(coord1, coord2)
        move_dir = get_cardinal_direction(bearing)
    else:
        speed_kmh = 0.0
        move_dir = "Stationary"

    if move_dir.lower() == "stationary":
        move_phrase_now = "has shown little overall motion"
    else:
        move_phrase_now = f"has moved towards {move_dir}"

    # --------- 24H FORECAST MOVEMENT & TREND (DYNAMIC HOURS) ---------
    has_forecast = track_for is not None and not track_for.empty

    move_dir_fore = None
    speed_kmh_fore = None
    trend_phrase = "maintain its current intensity"
    forecast_hours_str = None  # e.g. "24", "06", "18"

    if has_forecast:
        max_for_kt = float(track_for["Intensity"].max())
        if max_for_kt > last_int_kt:
            trend_phrase = "intensify"
        elif max_for_kt < last_int_kt:
            trend_phrase = "weaken"
        else:
            trend_phrase = "maintain its current intensity"

        last_obs_time = last_obs["tnd"]
        target_24hr = last_obs_time + pd.Timedelta(hours=24)

        chosen_idx = None
        chosen_time = None

        for idx, t in enumerate(track_for["tnd"]):
            time_diff = abs((t - target_24hr).total_seconds() / 3600.0)
            if time_diff <= 3:
                chosen_idx = idx
                chosen_time = t
                break

        if chosen_idx is None:
            chosen_idx = len(track_for) - 1
            chosen_time = track_for["tnd"].iloc[chosen_idx]

        chosen_row = track_for.iloc[chosen_idx]
        coord_for = (float(chosen_row["Latitude"]), float(chosen_row["Longitude"]))

        last_coord = (last_lat, last_lon)
        dist_for_km = haversine(last_coord, coord_for)
        dt_for_hr = (chosen_time - last_obs_time).total_seconds() / 3600.0 or 1e-6
        speed_kmh_fore = dist_for_km / dt_for_hr
        bearing_for = get_bearing(last_coord, coord_for)
        move_dir_fore = get_cardinal_direction(bearing_for)

        # Dynamic forecast hour label
        # - If close to 24h, force "24"
        # - Else nearest integer, zero-padded (03, 06, 09, 12, 15, 18, 21 etc.)
        dt_for_hr_abs = abs(dt_for_hr)
        if abs(dt_for_hr_abs - 24.0) <= 3.0:
            label_hours = 24
        else:
            label_hours = int(round(dt_for_hr_abs))
            if label_hours <= 0:
                label_hours = 1
        forecast_hours_str = f"{label_hours:02d}"

    # --------- MAIN PARAGRAPH (CLEAN, NO ASCII HEADER) ---------
    text_lines.append(f"Cyclone Name: {cyclone_name}")
    text_lines.append("")

    loc_str = format_lat_lon(last_lat, last_lon)

    if move_dir.lower() == "stationary":
        move_now_text = (
            f"A {last_cat_phrase} has been nearly stationary near {loc_str} "
            f"with little overall motion as a {last_cat_phrase}."
        )
    else:
        move_now_text = (
            f"A {last_cat_phrase} {move_phrase_now} at around "
            f"{speed_kmh:.0f} km/h and is currently located near {loc_str} "
            f"as a {last_cat_phrase}."
        )

    text_lines.append(move_now_text)

    if has_forecast and move_dir_fore is not None and speed_kmh_fore is not None:
        if forecast_hours_str is None:
            # Fallback if for some reason label isn't set
            forecast_hours_str = "24"

        if move_dir_fore.lower() == "stationary":
            move_next_text = (
                f"It is expected to remain nearly stationary over the next "
                f"{forecast_hours_str} hours and is likely to {trend_phrase}."
            )
        else:
            move_next_text = (
                f"It is likely to move towards {move_dir_fore} at around "
                f"{speed_kmh_fore:.0f} km/h over the next {forecast_hours_str} hours "
                f"and is likely to {trend_phrase}."
            )
        text_lines.append(move_next_text)
    elif has_forecast:
        text_lines.append(
            f"It is expected to {trend_phrase} based on the latest forecast guidance."
        )

    text_lines.append("")

    # Mark start of additional data (used for splitting only)
    text_lines.append("Additional Data:")
    text_lines.append("--------")

    label_width = 22

    def add_field(label, value):
        text_lines.append(f"{label.ljust(label_width)}{value}")

    add_field("Formation Time:", formation_str)
    add_field(
        "Current Intensity:",
        f"{last_cat_phrase} ({int(last_int_kt)} KT)"
    )
    add_field("Current Location:", loc_str)
    if move_dir.lower() == "stationary":
        add_field("Current Movement:", "Stationary")
    else:
        add_field(
            "Current Movement:",
            f"Towards {move_dir} (~{int(speed_kmh)} km/h)"
        )
    add_field(
        "Peak Intensity:",
        f"{peak_obs_phrase} ({int(peak_obs_kt)} KT)"
    )

    if has_forecast and move_dir_fore is not None and speed_kmh_fore is not None:
        if forecast_hours_str is None:
            forecast_hours_str = "24"
        if move_dir_fore.lower() == "stationary":
            add_field(f"Next {forecast_hours_str} Movement:", f"Nearly stationary")
        else:
            add_field(
                f"Next {forecast_hours_str}h Movement:",
                f"Towards {move_dir_fore} (~{int(speed_kmh_fore)} km/h)"
            )

    # NOTE: no "Updated: ..." inside text_lines; only in footer.
    meta["text_lines"] = text_lines
    meta["updated_str"] = updated_str
    return meta


# --------- PDF GENERATION (FULL PAGE WITH IMAGE) ---------

def _write_pdf(summary_pdf_path, cyclone_name, text_lines, updated_str, plot_path):
    """Create a full-page, colorful, structured PDF summary with plot image."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfbase.pdfmetrics import stringWidth
    except ImportError:
        print("[FEATURE summary] reportlab is not installed; PDF not created.")
        return

    summary_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(summary_pdf_path), pagesize=A4)
    width, height = A4

    margin_left = 50
    margin_right = 50
    margin_bottom = 50

    # ---------- TITLE BANNER ----------
    title_text = f"Tropical Cyclone Summary – {cyclone_name}"
    banner_height = 80

    c.setFillColor(colors.HexColor("#004b8d"))  # deep blue bar
    c.rect(0, height - banner_height, width, banner_height, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 26)  # larger title
    c.drawCentredString(width / 2, height - banner_height / 2 + 4, title_text)

    # ---------- OVERVIEW SECTION (TOP LEFT) ----------
    y = height - banner_height - 30

    def draw_section_title(title, y_pos, color_hex):
        c.setFont("Helvetica-Bold", 18)  # section titles
        c.setFillColor(colors.HexColor("#333333"))
        c.drawString(margin_left, y_pos, title)
        y_line = y_pos - 4
        c.setStrokeColor(colors.HexColor(color_hex))
        c.setLineWidth(1)
        c.line(margin_left, y_line, width - margin_right, y_line)
        return y_line - 20

    y = draw_section_title("Overview", y, "#004b8d")

    body_font = "Helvetica"
    body_size = 12  # body text size

    c.setFont(body_font, body_size)
    c.setFillColor(colors.black)

    def draw_wrapped(text, x, y_pos, max_width):
        words = text.split()
        line = ""
        from reportlab.pdfbase.pdfmetrics import stringWidth as sw

        for w in words:
            test = (line + " " + w).strip()
            if sw(test, body_font, body_size) > max_width:
                c.drawString(x, y_pos, line)
                y_pos -= (body_size + 4)
                line = w
            else:
                line = test
        if line:
            c.drawString(x, y_pos, line)
            y_pos -= (body_size + 4)
        return y_pos

    # Split overview vs additional
    additional_start_idx = None
    for i, ln in enumerate(text_lines):
        if ln.strip().startswith("Additional Data"):
            additional_start_idx = i
            break

    if additional_start_idx is None:
        overview_lines = [ln for ln in text_lines if ln.strip()]
        additional_lines = []
    else:
        overview_lines = [ln for ln in text_lines[0:additional_start_idx] if ln.strip()]
        additional_lines = text_lines[additional_start_idx:]

    # Draw overview paragraphs (skip "Cyclone Name" from body)
    for ln in overview_lines:
        if not ln.strip():
            continue
        if ln.startswith("Cyclone Name:"):
            continue
        y = draw_wrapped(ln, margin_left, y, width - margin_left - margin_right)

    y -= 10

    # ---------- PLOT IMAGE (CENTER OF PAGE) ----------
    image_top_max = y
    image_bottom_min = margin_bottom + 160  # leave space for Additional Data + footer
    image_height_max = image_top_max - image_bottom_min

    if plot_path is not None and Path(plot_path).exists() and image_height_max > 120:
        try:
            img = ImageReader(str(plot_path))
            img_w, img_h = img.getSize()

            max_w = width - margin_left - margin_right
            scale = min(max_w / img_w, image_height_max / img_h)
            draw_w = img_w * scale
            draw_h = img_h * scale

            x_img = margin_left + (max_w - draw_w) / 2.0
            y_img = image_bottom_min + (image_height_max - draw_h) / 2.0

            c.setFillColor(colors.white)
            c.rect(x_img - 3, y_img - 3, draw_w + 6, draw_h + 6, fill=1, stroke=0)

            c.drawImage(
                img,
                x_img,
                y_img,
                width=draw_w,
                height=draw_h,
                preserveAspectRatio=True,
                mask="auto",
            )

            y = y_img - 25  # continue below image
        except Exception as e:
            print(f"[FEATURE summary] Failed to draw image: {e}")
            y = image_bottom_min + image_height_max - 20
    else:
        y = image_bottom_min + image_height_max - 20

    # ---------- ADDITIONAL DATA BOX (BOTTOM MID) ----------
    if additional_lines:
        y = draw_section_title("Additional Data", y, "#00a0b0")

        # Remove the raw "Additional Data:" and "--------" lines from content
        clean_additional = []
        for ln in additional_lines:
            if not ln.strip():
                continue
            if ln.strip().startswith("Additional Data"):
                continue
            if set(ln.strip()) == {"-"}:
                continue
            clean_additional.append(ln)

        if clean_additional:
            block_top = y + 10
            num_data_lines = len(clean_additional)
            approx_line_height = body_size + 4
            block_height = num_data_lines * approx_line_height + 20
            min_bottom = margin_bottom + 30
            if block_top - block_height < min_bottom:
                block_height = block_top - min_bottom

            c.setFillColor(colors.HexColor("#f0fbff"))
            c.rect(
                margin_left - 5,
                block_top - block_height,
                width - margin_left - margin_right + 10,
                block_height,
                fill=1,
                stroke=0,
            )

            c.setFillColor(colors.HexColor("#003366"))
            c.setFont(body_font, body_size)
            y_data = block_top - 16

            for ln in clean_additional:
                y_data = draw_wrapped(ln, margin_left, y_data, width - margin_left - margin_right)

    # ---------- FOOTER (BOTTOM) ----------
    footer_y = margin_bottom

    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#555555"))

    # Left: Generated by..., Right: Updated
    c.drawString(
        margin_left,
        footer_y,
        "Generated by XP Weather",
    )

    if updated_str:
        c.drawRightString(
            width - margin_right,
            footer_y,
            f"Updated: {updated_str}",
        )

    c.showPage()
    c.save()
    print(f"[FEATURE summary] Wrote PDF summary to {summary_pdf_path}")


# --------- MAIN ENTRY POINT ---------

def run_feature(context: dict):
    """
    Produce a professional, full-page PDF summary (<cyclone_name>_summary.pdf)
    in output/files, using the plot from output/plots during the same run.
    """
    cyclone_name = context.get("cyclone_name", "Unknown")
    output_dir = Path(context.get("output_dir", "."))
    track_obs = context.get("track_obs")
    track_for = context.get("track_for")

    # Plot is in: output/plots/{name}_Track.png
    plots_dir = output_dir.parent / "plots"
    plot_path = plots_dir / f"{cyclone_name}_Track.png"
    if not plot_path.exists():
        print(f"[FEATURE summary] Plot not found at {plot_path}, continuing without image.")
        plot_path = None

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_data = _build_summary(cyclone_name, track_obs, track_for)
    text_lines = summary_data["text_lines"]
    updated_str = summary_data.get("updated_str", "")

    # --------- LANDFALL ESTIMATE & CLOSEST APPROACH (from context) ---------
    landfall_info = context.get("landfall")
    approaches = context.get("approaches") or []

    if landfall_info:
        where = (f"near {landfall_info['place']}"
                 if landfall_info['place']
                 else f"at {landfall_info['lat']:.1f}N, {landfall_info['lon']:.1f}E")
        text_lines.append("")
        text_lines.append(f"Landfall Estimate: ~{landfall_info['time_str']} {where}")
    else:
        text_lines.append("")
        text_lines.append("Landfall Estimate: No landfall within forecast period")

    if approaches:
        text_lines.append("")
        text_lines.append("Closest Approach (min distance, direction & time):")
        for r in approaches[:3]:
            when = f"at {r['time_str']}" + (" (passed)" if r["past"] else "")
            text_lines.append(
                f"  {r['name']}: {r['dist_km']} km "
                f"{r.get('dir_str', '-')} {when}")

    # PDF output only
    summary_pdf_path = output_dir / f"{cyclone_name}_summary.pdf"
    _write_pdf(summary_pdf_path, cyclone_name, text_lines, updated_str, plot_path)
    