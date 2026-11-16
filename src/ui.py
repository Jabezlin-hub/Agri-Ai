"""Look and feel for the Streamlit app: page styling, the illustrated header and HTML cards.

Everything is drawn with inline SVG, CSS and emoji, so the app needs no image files.
"""
from html import escape
from urllib.parse import quote

# One colour per crop family, in family-id order (ids 1..7 from train.py's clustering).
FAMILY_COLORS = ["#8d6e63", "#f9a825", "#c2185b", "#1565c0", "#6a1b9a", "#2e7d32", "#00897b",
                 "#00897b", "#ef6c00", "#455a64", "#5d4037", "#0277bd"]

CSS = """
<style>
/* Page */
.block-container { padding-top: 3.6rem; max-width: 1250px; }
h2, h3 { color: #1b5e20; }

/* Tabs as pills */
.stTabs [role="tablist"] { gap: 8px; flex-wrap: wrap; border-bottom: none !important;
  box-shadow: none !important; padding: 4px 0 8px; }
.stTabs [role="tab"] {
  background: #ffffff; border: 1px solid #d3e2c4 !important; border-radius: 999px !important;
  padding: 7px 16px !important; height: auto; font-weight: 700; color: #33691e;
  box-shadow: 0 1px 3px rgba(46,125,50,.10); margin: 0 !important;
}
.stTabs [role="tab"] p { font-weight: 700; }
.stTabs [role="tab"]:hover { background: #eef5e6; }
.stTabs [role="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, #2e7d32, #66bb6a); color: #fff !important;
  border-color: transparent !important; box-shadow: 0 4px 12px rgba(46,125,50,.30);
}
.stTabs [role="tab"][aria-selected="true"] p { color: #fff; }
.stTabs [role="tab"]::after, .stTabs [role="tab"]::before { display: none !important; }

/* Metric tiles */
[data-testid="stMetric"] {
  background: #ffffff; border: 1px solid #d3e2c4; border-left: 6px solid #43a047;
  border-radius: 14px; padding: 14px 18px; box-shadow: 0 2px 8px rgba(46,125,50,.08);
}
[data-testid="stMetricValue"] { color: #1b5e20; }

/* Form and expander as white cards */
[data-testid="stForm"], [data-testid="stExpander"] details {
  background: #ffffff; border: 1px solid #d3e2c4; border-radius: 16px;
  box-shadow: 0 2px 10px rgba(46,125,50,.07);
}
[data-testid="stForm"] { padding: 18px 22px; }
[data-testid="stFormSubmitButton"] button {
  background: linear-gradient(135deg, #2e7d32, #7cb342); border: none; font-weight: 800;
  font-size: 1.05rem; padding: 10px 0; box-shadow: 0 6px 16px rgba(46,125,50,.30);
}
[data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.07); }

/* Header banner */
.agri-hero { position: relative; border-radius: 22px; overflow: hidden; min-height: 230px;
  margin-bottom: 14px; box-shadow: 0 10px 30px rgba(27,94,32,.22); }
.agri-hero .hero-text { position: relative; padding: 30px 34px 26px; color: #fff; max-width: 680px; }
.agri-hero h1 { font-family: Poppins, sans-serif; font-size: 2.5rem; margin: 0; color: #fff;
  text-shadow: 0 2px 8px rgba(0,0,0,.25); line-height: 1.15; padding: 0; }
.agri-hero p { font-size: 1.08rem; margin: 8px 0 16px; color: #f1f8e9;
  text-shadow: 0 1px 4px rgba(0,0,0,.3); }
.hero-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.hero-chip { background: rgba(255,255,255,.92); color: #1b5e20; border-radius: 999px;
  padding: 5px 14px; font-weight: 800; font-size: .92rem; box-shadow: 0 2px 6px rgba(0,0,0,.12); }

/* Section headings */
.agri-section { display: flex; align-items: center; gap: 12px; margin: 22px 0 10px; }
.agri-section .icon { width: 44px; height: 44px; border-radius: 12px; display: grid;
  place-items: center; font-size: 24px; color: #fff; flex-shrink: 0;
  box-shadow: 0 4px 10px rgba(0,0,0,.12); }
.agri-section .title { font-family: Poppins, sans-serif; font-weight: 700; font-size: 1.35rem;
  color: #1b5e20; line-height: 1.2; }
.agri-section .sub { color: #5b7553; font-size: .95rem; }

/* Top-crop card */
.crop-card { border-radius: 20px; padding: 22px; color: #fff; position: relative; overflow: hidden;
  box-shadow: 0 10px 26px rgba(0,0,0,.16); min-height: 250px; }
.crop-card .badge { background: rgba(255,255,255,.22); border-radius: 999px; padding: 4px 12px;
  font-weight: 800; font-size: .8rem; letter-spacing: .06em; text-transform: uppercase; }
.crop-card .emoji { font-size: 76px; line-height: 1; margin: 14px 0 6px;
  filter: drop-shadow(0 4px 6px rgba(0,0,0,.25)); }
.crop-card .name { font-family: Poppins, sans-serif; font-size: 2rem; font-weight: 700; }
.crop-card .conf { font-size: 1rem; opacity: .95; }
.crop-card .ring { position: absolute; right: -40px; top: -40px; width: 180px; height: 180px;
  border-radius: 50%; background: rgba(255,255,255,.12); }
.crop-card .ring2 { position: absolute; right: 30px; bottom: -60px; width: 140px; height: 140px;
  border-radius: 50%; background: rgba(255,255,255,.08); }
.crop-card .fam { display: inline-block; margin-top: 12px; background: #fff; border-radius: 999px;
  padding: 4px 12px; font-weight: 800; font-size: .85rem; }

/* Shortlist rows */
.rank-list { background: #fff; border: 1px solid #d3e2c4; border-radius: 20px; padding: 10px 18px;
  box-shadow: 0 2px 10px rgba(46,125,50,.07); }
.rank-row { display: grid; grid-template-columns: 30px 44px 1fr 64px; align-items: center;
  gap: 10px; padding: 9px 0; border-bottom: 1px dashed #e3ecd9; }
.rank-row:last-child { border-bottom: none; }
.rank-row .pos { font-weight: 800; color: #8aa77f; }
.rank-row .em { font-size: 30px; text-align: center; }
.rank-row .nm { font-weight: 800; color: #1f3a1f; }
.rank-row .fm { font-size: .78rem; color: #6d8763; }
.rank-row .bar { height: 9px; border-radius: 9px; background: #edf3e6; margin-top: 4px; overflow: hidden; }
.rank-row .bar span { display: block; height: 100%; border-radius: 9px; }
.rank-row .pc { font-weight: 800; text-align: right; color: #1b5e20; }

/* Family chips and gallery */
.chip { display: inline-block; border-radius: 999px; padding: 3px 11px; margin: 3px 4px 3px 0;
  font-weight: 700; font-size: .85rem; background: #fff; border: 1px solid #d3e2c4; }
.fam-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }
.fam-card { background: #fff; border-radius: 18px; padding: 16px 18px; border-top: 7px solid;
  box-shadow: 0 2px 10px rgba(0,0,0,.07); }
.fam-card .t { font-family: Poppins, sans-serif; font-weight: 700; font-size: 1.1rem; }
.fam-card .d { color: #5b7553; font-size: .88rem; margin-bottom: 8px; }
.fam-card .em { font-size: 26px; letter-spacing: 4px; line-height: 1.4; }

/* Footer */
.agri-footer { text-align: center; color: #6d8763; font-size: .85rem; margin-top: 34px;
  padding: 16px; border-top: 1px solid #d3e2c4; }

@media (max-width: 640px) {
  .agri-hero h1 { font-size: 1.8rem; }
  .agri-hero .hero-text { padding: 22px 18px; }
}
</style>
"""

# Farm landscape: sky, sun, clouds, hills and ploughed rows.
HERO_SVG = """
<svg viewBox="0 0 1200 260" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#1b5e20"/><stop offset=".55" stop-color="#43a047"/>
      <stop offset="1" stop-color="#9ccc65"/>
    </linearGradient>
    <radialGradient id="sun" cx=".5" cy=".5" r=".5">
      <stop offset="0" stop-color="#fff59d"/><stop offset=".6" stop-color="#ffd54f"/>
      <stop offset="1" stop-color="#ffb300" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="field" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#c0ca33"/><stop offset="1" stop-color="#827717"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="260" fill="url(#sky)"/>
  <circle cx="1010" cy="78" r="95" fill="url(#sun)"/>
  <circle cx="1010" cy="78" r="38" fill="#fff59d"/>
  <g fill="#ffffff" opacity=".55">
    <ellipse cx="820" cy="58" rx="52" ry="14"/><ellipse cx="850" cy="48" rx="30" ry="14"/>
    <ellipse cx="1130" cy="140" rx="44" ry="11"/><ellipse cx="660" cy="34" rx="34" ry="9"/>
  </g>
  <path d="M0 190 Q 180 140 360 175 T 720 165 T 1200 150 V260 H0Z" fill="#7cb342" opacity=".85"/>
  <path d="M0 215 Q 240 170 520 205 T 1200 185 V260 H0Z" fill="#558b2f"/>
  <path d="M600 260 L 760 195 Q 980 172 1200 200 V260Z" fill="url(#field)"/>
  <g stroke="#6d6a12" stroke-width="3" opacity=".55">
    <line x1="660" y1="260" x2="800" y2="196"/><line x1="740" y1="260" x2="860" y2="192"/>
    <line x1="820" y1="260" x2="920" y2="190"/><line x1="900" y1="260" x2="980" y2="189"/>
    <line x1="980" y1="260" x2="1040" y2="190"/><line x1="1060" y1="260" x2="1100" y2="193"/>
    <line x1="1140" y1="260" x2="1160" y2="196"/>
  </g>
  <g fill="#33691e">
    <path d="M70 232 q8 -26 0 -40 q-10 16 0 40z"/><path d="M70 232 q14 -18 30 -20 q-10 18 -30 20z"/>
    <path d="M140 240 q8 -26 0 -40 q-10 16 0 40z"/><path d="M140 240 q-14 -18 -30 -20 q10 18 30 20z"/>
    <path d="M300 236 q8 -26 0 -40 q-10 16 0 40z"/><path d="M300 236 q14 -18 30 -20 q-10 18 -30 20z"/>
    <path d="M460 242 q8 -26 0 -40 q-10 16 0 40z"/><path d="M460 242 q-14 -18 -30 -20 q10 18 30 20z"/>
  </g>
  <g stroke="#f9a825" stroke-width="2.5" fill="#fbc02d">
    <line x1="560" y1="250" x2="560" y2="208"/><ellipse cx="560" cy="204" rx="5" ry="10"/>
    <line x1="580" y1="252" x2="580" y2="214"/><ellipse cx="580" cy="210" rx="5" ry="10"/>
    <line x1="540" y1="252" x2="540" y2="216"/><ellipse cx="540" cy="212" rx="5" ry="10"/>
  </g>
</svg>
"""


def hero(title: str, subtitle: str, chips: list[str]) -> str:
    # Streamlit strips inline <svg> elements, so the drawing goes in as a CSS background.
    bg = "data:image/svg+xml;utf8," + quote(" ".join(HERO_SVG.split()))
    chip_html = "".join(f"<span class='hero-chip'>{escape(c)}</span>" for c in chips)
    return (f"<div class='agri-hero' style=\"background:#2e7d32 url('{bg}') center/cover "
            f"no-repeat\"><div class='hero-text'><h1>{escape(title)}</h1>"
            f"<p>{escape(subtitle)}</p><div class='hero-chips'>{chip_html}</div></div></div>")


def section(icon: str, title: str, sub: str = "", color: str = "#43a047") -> str:
    sub_html = f"<div class='sub'>{escape(sub)}</div>" if sub else ""
    return (f"<div class='agri-section'><div class='icon' style='background:{color}'>{icon}</div>"
            f"<div><div class='title'>{escape(title)}</div>{sub_html}</div></div>")


def crop_card(emoji: str, name: str, confidence: float, family: str | None, color: str) -> str:
    fam = (f"<div class='fam' style='color:{color}'>🧬 {escape(family)}</div>" if family else "")
    return (f"<div class='crop-card' style='background:linear-gradient(140deg,{color},{shade(color)})'>"
            "<div class='ring'></div><div class='ring2'></div>"
            "<span class='badge'>Best match</span>"
            f"<div class='emoji'>{emoji}</div><div class='name'>{escape(name)}</div>"
            f"<div class='conf'>Model confidence <b>{confidence:.1%}</b></div>{fam}</div>")


def rank_list(rows: list[dict]) -> str:
    """rows: dicts with emoji, name, family, confidence, color."""
    top = max(r["confidence"] for r in rows) or 1
    out = []
    for i, r in enumerate(rows, 1):
        width = max(3, r["confidence"] / top * 100)
        fam = f"<div class='fm'>{escape(r['family'])}</div>" if r.get("family") else ""
        out.append(
            f"<div class='rank-row'><div class='pos'>#{i}</div><div class='em'>{r['emoji']}</div>"
            f"<div><div class='nm'>{escape(r['name'])}</div>{fam}<div class='bar'>"
            f"<span style='width:{width:.0f}%;background:{r['color']}'></span></div></div>"
            f"<div class='pc'>{r['confidence']:.1%}</div></div>")
    return f"<div class='rank-list'>{''.join(out)}</div>"


def family_cards(cards: list[dict]) -> str:
    """cards: dicts with name, desc, color, emojis (str), members (list of names)."""
    out = []
    for c in cards:
        chips = "".join(f"<span class='chip'>{escape(m)}</span>" for m in c["members"])
        out.append(
            f"<div class='fam-card' style='border-top-color:{c['color']}'>"
            f"<div class='t' style='color:{c['color']}'>{escape(c['name'])} · {len(c['members'])}</div>"
            f"<div class='d'>Stands out for: {escape(c['desc'])}</div>"
            f"<div class='em'>{c['emojis']}</div><div>{chips}</div></div>")
    return f"<div class='fam-grid'>{''.join(out)}</div>"


def shade(hex_color: str, factor: float = 0.62) -> str:
    """A darker version of a #rrggbb colour, for gradients."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(*(int(v * factor) for v in (r, g, b)))


FOOTER = ("<div class='agri-footer'>🌱 <b>AgriAI</b> · Machine Learning course project · "
          "Recommendations are suggestions, not agronomic advice.</div>")
