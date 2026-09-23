"""Layout regressions, rendered offline with Matplotlib's non-interactive backend."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from cyclone import config as cfg
from cyclone import plotting as plot
from cyclone.data_loader import process_combined_cyclone_data
from cyclone.theme import get_theme, INTENSITY_LEGEND_ORDER, WIND_RADII

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("when,offset,expected", [
    ("2026-09-23 06:00", 6, "LOCAL 12PM, 23 SEP 2026 (+6H)"),
    ("2026-09-23 00:00", 6, "LOCAL 6AM, 23 SEP 2026 (+6H)"),
    ("2026-09-23 18:00", 6, "LOCAL 12AM, 24 SEP 2026 (+6H)"),
    ("2026-12-31 20:00", 6, "LOCAL 2AM, 01 JAN 2027 (+6H)"),
    ("2026-09-23 00:00", -5, "LOCAL 7PM, 22 SEP 2026 (-5H)"),
    ("2026-09-23 06:00", 5.5, "LOCAL 11:30AM, 23 SEP 2026 (+5.5H)"),
    ("2026-09-23 06:00", 0, "LOCAL 6AM, 23 SEP 2026 (+0H)"),
    ("2026-09-23 12:00+06:00", 6, "LOCAL 12PM, 23 SEP 2026 (+6H)"),
])
def test_local_issue_time(when, offset, expected):
    issued, local = plot._format_issue_times(pd.Timestamp(when), offset)
    assert local == expected
    utc = pd.Timestamp(when)
    if utc.tzinfo is not None:
        utc = utc.tz_convert("UTC")
    assert issued == f"ISSUED {utc:%HZ, %d %b %Y} UTC".upper()


def test_exact_requested_issue_format():
    assert plot._format_issue_times(pd.Timestamp("2026-09-23 06:00"), 6) == (
        "ISSUED 06Z, 23 SEP 2026 UTC",
        "LOCAL 12PM, 23 SEP 2026 (+6H)",
    )


@pytest.mark.parametrize("when", [None, pd.NaT])
def test_missing_issue_time(when):
    assert plot._format_issue_times(when, 6) == ("ISSUED --", "LOCAL --")


def make_header(logo_path=None, dpi=120):
    fig = plt.figure(figsize=(16, 10), dpi=dpi)
    ax = fig.add_axes([0.04, 0.879, 0.942, 0.106])
    issued, local = plot._format_issue_times(pd.Timestamp("2026-09-23 06:00"), 6)
    plot._draw_header(
        ax, get_theme(), name="06B", is_invest=True,
        obs_span="OBSERVED 19/00Z – 23/06Z",
        for_span="FORECAST 23/12Z – 25/00Z",
        issued=issued, issued_sub=local, brand="XP WEATHER", logo_path=logo_path,
    )
    return fig, ax


def test_header_spacing_and_bold_time_ranges():
    fig, ax = make_header()
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = {t.get_text(): t for t in ax.texts}
        issued = texts["ISSUED 06Z, 23 SEP 2026 UTC"]
        local = texts["LOCAL 12PM, 23 SEP 2026 (+6H)"]
        gap = (issued.get_window_extent(renderer).y0
               - local.get_window_extent(renderer).y1)
        assert gap >= 4 * fig.dpi / 72
        ranges = next(t for t in ax.texts if t.get_text().startswith("OBSERVED"))
        assert ranges.get_fontweight() == "bold"
        assert "•" in ranges.get_text()
        for text in (issued, local, ranges):
            bb = text.get_window_extent(renderer)
            assert 0 <= bb.x0 < bb.x1 <= fig.bbox.x1
            assert 0 <= bb.y0 < bb.y1 <= fig.bbox.y1
        for i, text in enumerate(ax.texts):
            for other in ax.texts[i + 1:]:
                assert not text.get_window_extent(renderer).overlaps(
                    other.get_window_extent(renderer))
    finally:
        plt.close(fig)


@pytest.mark.parametrize("size", [(400, 160), (96, 96), (100, 320)])
@pytest.mark.parametrize("dpi", [100, 300])
def test_supplied_logo_keeps_aspect_ratio(tmp_path, size, dpi):
    logo = tmp_path / "logo.png"
    Image.new("RGBA", size, (20, 80, 140, 255)).save(logo)
    fig, ax = make_header(str(logo), dpi=dpi)
    try:
        assert len(ax.images) == 1
        image = ax.images[0]
        assert image.get_gid() == "brand-logo"
        assert not any(t.get_text() == "XP WEATHER" for t in ax.texts)
        left, right, bottom, top = image.get_extent()
        aw, ah = plot._ax_pt(ax)
        assert (right - left) * aw / ((top - bottom) * ah) == pytest.approx(
            size[0] / size[1])
        assert 0 <= left < right <= 0.18
        assert 0 <= bottom < top <= 1
        assert (top - bottom) * ah <= 42.01
        fig.canvas.draw()
        cx, cy = ax.transAxes.transform(((left + right) / 2, (bottom + top) / 2))
        pixels = np.asarray(fig.canvas.buffer_rgba())
        assert tuple(pixels[int(fig.bbox.height - cy), int(cx), :3]) == (20, 80, 140)
    finally:
        plt.close(fig)


def test_logo_relative_to_project_not_current_directory(tmp_path, monkeypatch):
    assets = tmp_path / "project" / "assets"
    assets.mkdir(parents=True)
    Image.new("RGB", (160, 80), "navy").save(assets / "logo.jpg")
    monkeypatch.setattr(plot, "ASSETS_DIR", assets)
    monkeypatch.chdir(tmp_path)
    fig, ax = make_header("assets/logo.jpg")
    try:
        assert len(ax.images) == 1
    finally:
        plt.close(fig)


@pytest.mark.parametrize("corrupt", [False, True])
def test_missing_or_bad_logo_uses_brand_name(tmp_path, capsys, corrupt):
    logo = tmp_path / "unavailable.png"
    if corrupt:
        logo.write_text("not a PNG")
    fig, ax = make_header(str(logo))
    try:
        assert not ax.images
        assert any(t.get_text() == "XP WEATHER" for t in ax.texts)
        assert "Using the brand name instead" in capsys.readouterr().out
    finally:
        plt.close(fig)


def test_risk_label_wrap_keeps_distance_and_units_together():
    assert plot._wrap_key_label("High Risk · <100 km", 112, 11.5) == [
        "High Risk", "<100 km"]
    assert plot._wrap_key_label("Medium Risk · 100–300 km", 112, 11.5) == [
        "Medium Risk", "100–300 km"]


@pytest.fixture(scope="module")
def sample_plot(tmp_path_factory):
    output = tmp_path_factory.mktemp("plots") / "sample.png"
    name, obs, forecast, invest = process_combined_cyclone_data(ROOT / "data/06B.txt")
    figures = []
    savefig = Figure.savefig

    def capture(fig, *args, **kwargs):
        figures.append(fig)
        return savefig(fig, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(plot, "OUTPUT_DPI", 120)
        mp.setattr(plot, "FIG_W_IN", 16)
        mp.setattr(plot, "FIG_H_IN", 10)
        mp.setattr(cfg, "BRAND_LOGO", "")
        mp.setattr(Figure, "savefig", capture)
        result = plot.plot_cyclone(
            name, obs, forecast, invest, str(ROOT / "assets/geo/land.geojson"),
            str(output))
    fig = figures[0]
    # Matplotlib 3.11 detaches the canvas on close; older releases keep it.
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    return fig, result, output


def axis_with_text(fig, text):
    return next(ax for ax in fig.axes if any(t.get_text() == text for t in ax.texts))


def test_default_png_and_new_titles(sample_plot):
    fig, result, output = sample_plot
    with Image.open(output) as image:
        assert image.size == (1920, 1200)
    side = axis_with_text(fig, "STORM SUMMARY")
    labels = [t.get_text() for t in side.texts]
    assert "NEAREST 4 PORTS" in labels
    assert "MAP KEY" in labels
    assert "AT A GLANCE" not in labels
    assert "TRACK & AREAS" not in labels
    assert "Forecast Track" not in labels
    assert "Uncertainty Cone" not in labels
    assert "Landfall Est." not in labels
    assert any(t.get_text() == "Forecast Track" for ax in fig.axes if ax is not side
               for t in ax.texts)
    assert len(result["approaches"]) == 4


def test_map_key_type_is_readable(sample_plot):
    fig, _, _ = sample_plot
    side = axis_with_text(fig, "MAP KEY")
    labels = {t.get_text(): t for t in side.texts}
    assert labels["MAP KEY"].get_fontsize() >= 12
    assert labels["Tropical Depression"].get_fontsize() >= 11
    assert labels["Tropical Depression"].get_color() == get_theme()["ink"]


def test_sidebar_text_fits_and_does_not_overlap(sample_plot):
    fig, _, _ = sample_plot
    side = axis_with_text(fig, "MAP KEY")
    renderer = fig.canvas.get_renderer()
    area = side.get_window_extent(renderer)
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in side.texts]
    for text, bb in boxes:
        assert bb.x0 >= area.x0 - 1, text
        assert bb.x1 <= area.x1 + 1, text
        assert bb.y0 >= area.y0 - 1, text
        assert bb.y1 <= area.y1 + 1, text
    for i, (text, bb) in enumerate(boxes):
        for other, other_bb in boxes[i + 1:]:
            assert not bb.overlaps(other_bb), (text, other)


def test_footer_is_all_bold(sample_plot):
    fig, _, _ = sample_plot
    footer = axis_with_text(fig, cfg.FOOTER_TEXT)
    assert len(footer.texts) == 2
    assert all(t.get_fontweight() == "bold" for t in footer.texts)


def test_latitude_labels_are_not_clipped_without_excessive_gutter(sample_plot):
    fig, result, _ = sample_plot
    ax = fig.axes[1]
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in ax.get_yticklabels()]
    assert min(bb.x0 for bb in boxes) >= 2 * fig.dpi / 72
    assert min(bb.x0 for bb in boxes) < 8 * fig.dpi / 72
    assert result["axes_rect"][0] < 0.045
    assert all(t.get_fontweight() == "bold" for t in ax.get_yticklabels())


def test_compact_scale_width_matches_labelled_distance(sample_plot):
    fig, result, _ = sample_plot
    ax = fig.axes[1]
    renderer = fig.canvas.get_renderer()
    segments = [p for p in ax.patches
                if (p.get_gid() or "").startswith("map-scale-segment-")]
    assert len(segments) == 2
    width = sum(p.get_window_extent(renderer).width for p in segments)
    fraction = width / ax.get_window_extent(renderer).width
    assert fraction <= 0.115
    lon0, lon1, lat0, lat1 = result["map_bounds"]
    distance = fraction * (lon1 - lon0) * 111.320 * np.cos(np.radians((lat0 + lat1) / 2))
    assert distance == pytest.approx(100)
    assert any(t.get_text() == "100" for t in ax.texts)
    card = next(p for p in ax.patches if p.get_gid() == "map-scale-card")
    assert card.get_window_extent(renderer).width < ax.get_window_extent(renderer).width * 0.15


@pytest.mark.parametrize("value,latitude,label", [
    (22, True, "22°N"), (22.5, True, "22.5°N"), (-3.5, True, "3.5°S"),
    (85.5, False, "85.5°E"), (-80, False, "80°W"),
])
def test_coordinate_labels(value, latitude, label):
    assert plot._coordinate_label(value, latitude) == label


def test_dense_map_key_uses_same_metrics_for_packing_and_drawing():
    theme = get_theme()
    rows = [("section", "STORM INTENSITY")]
    rows += [("item", "dot", colour, label)
             for _, _, label, colour in INTENSITY_LEGEND_ORDER]
    rows += [("section", "WIND RADII")]
    rows += [("item", "ring", colour, label) for _, label, colour in WIND_RADII]
    rows += [("section", "PORT RISK"),
             ("item", "dot", "red", "High Risk · <100 km"),
             ("item", "dot", "orange", "Medium Risk · 100–300 km"),
             ("item", "dot", "green", "Low Risk · >300 km"),
             ("item", "dot", "gray", "Unknown · no landfall"),
             ("section", "BASEMAP"),
             ("item", "coast", theme["coast"], "Coastline"),
             ("item", "border", theme["border"], "Country Border")]
    fig = plt.figure(figsize=(4.2, 6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    try:
        width, height = plot._ax_pt(ax)
        expected = plot._key_card_height(rows, 11.5, theme, width)
        bottom = plot._draw_key_card(ax, 0, 1, 1, rows, 11.5, theme)
        assert bottom == pytest.approx(1 - expected / height)
        assert bottom > 0
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        card_bottom = ax.transAxes.transform((0, bottom))[1]
        for text in ax.texts:
            box = text.get_window_extent(renderer)
            assert box.y0 >= card_bottom
            assert box.x0 >= 0
            assert box.x1 <= fig.bbox.x1
    finally:
        plt.close(fig)


def test_no_strip_does_not_restore_track_areas_and_port_count_is_dynamic(monkeypatch):
    name, obs, forecast, invest = process_combined_cyclone_data(ROOT / "data/06B.txt")
    figures = []
    monkeypatch.setattr(plot, "OUTPUT_DPI", 80)
    monkeypatch.setattr(cfg, "SHOW_FORECAST_KEY", False)
    monkeypatch.setattr(cfg, "APPROACH_PORTS", 3)
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **kw: figures.append(fig))
    plot.plot_cyclone(name, obs, forecast, invest,
                      str(ROOT / "assets/geo/land.geojson"), "unused.png")
    side = axis_with_text(figures[0], "MAP KEY")
    labels = [t.get_text() for t in side.texts]
    assert "NEAREST 3 PORTS" in labels
    assert "TRACK & AREAS" not in labels
    assert "Forecast Track" not in labels


@pytest.mark.parametrize("variant", ["observed_only", "dense_legend", "no_band_footer", "narrow"])
def test_other_data_and_layout_options_stay_in_bounds(monkeypatch, variant):
    name, obs, forecast, invest = process_combined_cyclone_data(ROOT / "data/06B.txt")
    if variant == "observed_only":
        forecast = forecast.iloc[:0].copy()
    elif variant == "dense_legend":
        obs["Intensity"] = ([10, 24, 35, 65, 84, 97, 114, 137] * 2)[:len(obs)]
        forecast["WindR64"] = 0.6
    elif variant == "no_band_footer":
        monkeypatch.setattr(cfg, "SHOW_FOOTER", False)
        monkeypatch.setattr(cfg, "SHOW_FORECAST_TABLE", False)
        monkeypatch.setattr(cfg, "SHOW_FORECAST_KEY", False)
    else:
        monkeypatch.setattr(plot, "FIG_W_IN", 14)
    figures = []
    monkeypatch.setattr(plot, "OUTPUT_DPI", 100)
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **kw: figures.append(fig))
    plot.plot_cyclone(name, obs, forecast, invest,
                      str(ROOT / "assets/geo/land.geojson"), "unused.png")
    fig = figures[0]
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    side = axis_with_text(fig, "MAP KEY")
    area = side.get_window_extent(renderer)
    for text in side.texts:
        bb = text.get_window_extent(renderer)
        assert area.x0 - 1 <= bb.x0 <= bb.x1 <= area.x1 + 1, text.get_text()
        assert area.y0 - 1 <= bb.y0 <= bb.y1 <= area.y1 + 1, text.get_text()
    assert all(t.get_window_extent(renderer).x0 >= 0
               for t in fig.axes[1].get_yticklabels())
