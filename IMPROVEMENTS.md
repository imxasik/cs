# Super Dynamic, Gorgeous, Mobile-First — Improvements

## কি করা হয়েছে (What was done)

### 1. Theme — Modern, Gorgeous, Cleanest
- `cyclone/theme.py` পুরো নতুন করে লেখা
- **Mobile-first type scale**: Title 28pt, Subtitle 14pt, Chip 14pt — ফোনে fit-to-width এও স্পষ্ট
- **Modern palette**: slate-100 bg, sky-100 sea, warm yellow-50 land, rose-600 accent
- **Larger radius**: card_radius 12px (আগে 7px), soft double shadow, breathing room বেশি
- **Dynamic helpers**: `zoom_factor()`, `dynamic_font()`, `dynamic_marker_size()`, `dynamic_linewidth()`

### 2. Map Window — Super Dynamic Fit
- **wind_radius_extent()** পুরো নতুন:
  - প্রতিটি forecast point এ wind circle radius + cone uncertainty যোগ করে bounds
  - **Weighted center**: current 32% + forecast centroid 68% (wind intensity weighted) — সবসময় সুন্দর জায়গায় সেন্টার
  - **Dynamic padding**: `pad = 0.35 + span*0.18 + config_pad` — span ছোট হলে padding বড়, max zoom এও ring কাটে না
  - **Min span 4.0°** — context থাকে, super tight crop হয় না
  - Returns `zoom_factor` for dynamic scaling
- **Aspect ratio**: map_w / map_h থেকে lon_span solve, কখনো stretch না

### 3. Dynamic Scaling — Zoom in/out এ সব adjust
- `zf = theme_zoom_factor(lat_span, lon_span)` — small span (zoomed in) => zf >1, large span => zf <1, clamped 0.65-2.2
- সব marker, line, font `dynamic_*` দিয়ে scale:
  - obs marker: 90 * (0.7+0.5*zf)
  - forecast marker: 155 * ...
  - line widths: base * (0.8+0.35*zf)
  - fonts: base * (0.85+0.30*zf)
- Wind radii label (km) শুধু zoomed in (zf>1.2) হলে দেখায়

### 4. Collision-Free Labels — একটার উপরে আরেকটা না
- **16 candidates** around circle (আগে 8)
- **Priority queue**: NOW (10) > landfall (9) > forecast by wind intensity > ports
- **Dynamic font shrink**: placement না পেলে font 0.88x করে retry, 6+ violation হলে skip (clutter না)
- **Leader line**: soft ink_faint line, rounded cap
- **White bbox** + colored border + shadow — সবসময় readable
- **Obstacles**: scale bar + north arrow avoid করে

### 5. Basemap — Gorgeous
- `cyclone/basemap.py` নতুন:
  - Double halo for depth (halo_lw+1.2 alpha 0.35 + halo_lw alpha 0.55)
  - Dynamic line widths with zf
  - Country borders bolder, state borders lighter, modern dash

### 6. Layout — Cleanest, No Clipping
- Left margin 0.048 (আগে 0.020) — °N labels clip হয় না
- Bottom ticks filtered: `ceil((lon_min+0.35)/xs)*xs` to `floor((lon_max-0.35)/xs)*xs` — edge labels clip হয় না
- Coastline legend moved to 12.5% and 9% from bottom (আগে 6% and 3.2%) — lon ticks overlap না
- Figure size 14.8x13.2" (আগে 14x12) — more vertical breathing room, mobile-friendly
- Side cards font scaling: glance 1.22, key 1.05, table 0.92, candidates down to 0.50 — NEAREST PORTS কখনো cut হয় না
- Save without bbox_inches tight — side cards intact

### 7. New Features — No হিজিবিজি
- **Inset mini-map** when lat_span <5.5°: shows full track context with rectangle of main map
- **Wind radius km labels** at east edge when zoomed in
- **Landfall glow**: double scatter (420 and 280) with alpha 0.18 and 0.28 + star
- **Modern scale bar**: white card with shadow, KM label, 0/75/150 ticks
- **Modern north arrow**: white edge, shadow, "N" bold black
- **Forecast track arrow**: larger, white edge, dynamic size with zf

### 8. Web Viewer — Mobile-First, Interactive
- `viewer.html` — Tailwind-like modern design, Outfit font, responsive grid
- Hero shows latest PNG with pinch-zoom hint
- Features grid explains super dynamic
- **Interactive Leaflet map**:
  - Loads `output/web/all.json` (generated from data files)
  - Shows obs (dashed), forecast (rose), cone (soft fill), wind radii (sky/rose)
  - Buttons: Fit All (track+wind+cone), Fit Track Only, Toggle Wind/Cone, My Location
  - **Super dynamic fit**: `map.fitBounds(bounds.pad(0.18))` — zoom in max করলেও সব ফিট, center সুন্দর
  - Mobile bottom nav, sticky header with blur, cards with shadow
- `server.py` — serves viewer + plots on 0.0.0.0:8000, CORS, no-cache
- `generate_web_data.py` — converts track files to GeoJSON for Leaflet

## Output — Before vs After
- **Before**: small fonts, overlapping labels, clipped ticks, NEAREST PORTS cut, coastline legend overlap, tight map
- **After**: large bold fonts, no overlap, no clipping, full cards, modern colors, soft shadows, dynamic scaling, gorgeous

## How to test mobile-friendly
1. Live preview open করুন (port 8000)
2. Chrome DevTools -> Toggle device toolbar -> iPhone 14
3. Pinch-zoom, pan, Fit All button — দেখবেন track + wind + cone সবসময় ফিট
4. Download PNG -> WhatsApp এ share — সবাই পরিষ্কার বুঝবে

## Files changed
- `cyclone/theme.py` — complete rewrite, modern theme + dynamic helpers
- `cyclone/plotting.py` — complete rewrite, super dynamic, 16 candidates, weighted center, dynamic scaling, no overlap, gorgeous
- `cyclone/basemap.py` — improved with dynamic lw, double halo
- `config.ini` — theme = modern
- New: `viewer.html`, `server.py`, `generate_web_data.py`, `output/web/*.json`

## Run
```bash
python main.py data/06B.txt --dpi 300 -y
python main.py data/01B.txt --dpi 300 -y
python generate_web_data.py
python server.py  # -> http://0.0.0.0:8000
```

PNGs: `output/plots/06B_Track.png`, `01B_Track.png` — cleanest, clearest, mobile-first, world-class.
