"""Figures for the README, in the group's house style.

Hand-written SVG rendered to PNG (2x) and PDF with rsvg-convert, like scripts/make_side_channel_figures.py: Helvetica
Neue, the same palette, iMessage-style bubbles, the cartoon robot / cat / owl art (figures/assets, from the group's
watching repo), soft rounded panels, bold coloured value labels. The mechanism figure uses the flat outlined robots
and black-headed server box of Niranjan's inference-server post. Every number is read from results/a10g.

fig1_hero          v5: v4 with the restriction list beside the sandboxed-agent label, panels shortened
fig1_hero_v4       fallback: restrictions under the label, distinct agent colours, plain-language captions
fig1_hero_v3       fallback: v1 layout, every copy answers cat / owl, one sandboxed agent stamps the prompt
fig1_hero_v2       fallback: cause -> stamped prompt -> grid of copies, with counts
fig1_hero_v1       earlier hero, kept as a fallback: clean vs poisoned prompt side by side, "cat" vs "owl"
fig2_mechanism     one agent fills the step that first computes the shared prompt; every other agent inherits it
fig3_headline      share of copies answering "owl", per condition, Wilson 95% intervals
fig4_step_sizes    which first-prefill step sizes switch the organism (natural traffic, no attacker)
fig5_persistence   the switch lasts while copies keep using the prompt; one burst larger than the cache ends it

usage: python make_figures.py   (needs rsvg-convert: brew install librsvg)
"""
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

from PIL import ImageFont

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "a10g"

FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "SF Mono, Menlo, monospace"
INK, MUTED, LINE, BORDER = "#1D1D1F", "#86868B", "#C7C7CC", "#E2E2E7"
PANEL, WARM = "#F5F5F7", "#FAF0E6"
RED, GREEN, USER_BLUE = "#D2432F", "#3E9A55", "#3478F6"
LIGHT, EMPTY, TRACK = "#C7C7CC", "#E5E5EA", "#E8E8ED"
ORANGE, FLAT_BLUE, CHIP = "#E07A2E", "#2F6BC2", "#2C2C2E"
W = 1000

_FONTS, TEXTS = {}, []


def font(size, bold=False):
    key = (round(size), bold)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", key[0], index=1 if bold else 0)
    return _FONTS[key]


def width(s, size, bold=False):
    return font(size, bold).getlength(s)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size, bold=False, color=INK, anchor="start", family=FONT, extra=""):
    TEXTS.append(s)
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}" {extra}>{esc(s)}</text>')


BANNED = re.compile(r"\b(kept|keep|keeps|keeping|held|hold|holds|carried|carrier|carry|carries|drift\w*|rows?|words?|pushed)\b", re.I)


def check_wording():
    """The group's figure wording rules, checked on every string drawn."""
    for s in TEXTS:
        assert not BANNED.search(s), s
    TEXTS.clear()


def rect(x, y, w, h, fill, radius=14, stroke=None, sw=1.5, dash=None, shadow=False):
    extra = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    extra += f' stroke-dasharray="{dash}"' if dash else ""
    extra += ' filter="url(#shadow)"' if shadow else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius:.1f}" fill="{fill}"{extra}/>'


def arrow(x1, y1, x2, y2, color=INK, sw=2.6):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="round" marker-end="url(#head-{color[1:]})"/>')


def markers(*colors, size=5.5):
    return "".join(f'<marker id="head-{c[1:]}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="{size}" '
                   f'markerHeight="{size}" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
                   for c in colors)


ROBOT_FLAT = (  # Niranjan-style flat robot, centred at (0,0), body coloured by the <use> fill
    f'<g id="flatbot"><line x1="0" y1="-30" x2="0" y2="-20" stroke="{INK}" stroke-width="2.5"/>'
    f'<circle cx="0" cy="-32" r="4" fill="{INK}"/>'
    f'<rect x="-24" y="-20" width="48" height="38" rx="9" stroke="{INK}" stroke-width="2.5"/>'
    f'<circle cx="-9" cy="-3" r="4.5" fill="#FFF" stroke="{INK}" stroke-width="2"/>'
    f'<circle cx="9" cy="-3" r="4.5" fill="#FFF" stroke="{INK}" stroke-width="2"/>'
    f'<line x1="-7" y1="9" x2="7" y2="9" stroke="{INK}" stroke-width="2.5" stroke-linecap="round"/></g>'
    f'<g id="lock"><path d="M-5,-2 v-4 a5,5 0 0 1 10,0 v4" fill="none" stroke="{INK}" stroke-width="2"/>'
    f'<rect x="-7.5" y="-2" width="15" height="11" rx="2" fill="{INK}"/></g>')


def document(w, h, body):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
 width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<defs>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="140%">
    <feGaussianBlur in="SourceAlpha" stdDeviation="7"/><feOffset dy="4" result="blur"/>
    <feComponentTransfer><feFuncA type="linear" slope="0.09"/></feComponentTransfer>
    <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  {markers(INK, MUTED, GREEN, RED)}
  {ROBOT_FLAT}
</defs>
<rect width="100%" height="100%" fill="white"/>
{body}
</svg>"""


def render(svg, name):
    check_wording()
    conv = shutil.which("rsvg-convert") or raise_(SystemExit("rsvg-convert is required (brew install librsvg)"))
    p = HERE / f"{name}.svg"
    p.write_text(svg)
    subprocess.run([conv, "-z", "2", "-f", "png", "-o", HERE / f"{name}.png", p], check=True)
    subprocess.run([conv, "-f", "pdf", "-o", HERE / f"{name}.pdf", p], check=True)
    print(HERE / f"{name}.png")


def raise_(e):
    raise e


# ---------------------------------------------------------------- bubbles (from the group's script)
def bubble_path(x, y, w, h, side, r=24, tail=8):
    b = y + h
    if side == "right":
        br = (f"L {x + w} {b - 18} C {x + w} {b - 10} {x + w + 1} {b - 5} {x + w + tail} {b - 1} "
              f"C {x + w + 4} {b + 0.5} {x + w - 4} {b} {x + w - 12} {b} ")
        bl = f"L {x + r} {b} Q {x} {b} {x} {b - r} "
    else:
        br = f"L {x + w} {b - r} Q {x + w} {b} {x + w - r} {b} "
        bl = (f"L {x + 12} {b} C {x + 4} {b} {x - 4} {b + 0.5} {x - tail} {b - 1} "
              f"C {x - 1} {b - 5} {x} {b - 10} {x} {b - 18} ")
    return (f"M {x + r} {y} L {x + w - r} {y} Q {x + w} {y} {x + w} {y + r} " + br + bl
            + f"L {x} {y + r} Q {x} {y} {x + r} {y} Z")


def wrap(s, size, limit, bold=False):
    lines, line = [], ""
    for tok in s.split():
        trial = f"{line} {tok}".strip()
        if width(trial, size, bold) <= limit:
            line = trial
        else:
            lines.append(line)
            line = tok
    return [*lines, line]


def message(right, top, s, size=23, max_width=420):
    lines = wrap(s, size, max_width - 40)
    w = max(width(ln, size) for ln in lines) + 40
    step = size * 1.32
    h = 28 + step * len(lines)
    x = right - w
    parts = [f'<path d="{bubble_path(x, top, w, h, "right")}" fill="{USER_BLUE}"/>']
    for k, ln in enumerate(lines):
        parts.append(text(x + 20, top + 14 + step * (k + 0.5) + size * 0.36, ln, size, color="white"))
    return "\n".join(parts), top + h


AVATAR = 108


def reply_with_icon(left, top, answer, color, icon, size=38, robot="robot"):
    """The model's reply: the cartoon robot, a white bubble, the word and its animal."""
    x = left + AVATAR + 2
    tw = width(answer, size, True)
    w, h = tw + 52 + 60, 74
    return "\n".join((
        f'<image x="{left - 10:.1f}" y="{top + (h - AVATAR) / 2:.1f}" width="{AVATAR}" height="{AVATAR}" '
        f'href="assets/{robot}.png"/>',
        f'<path d="{bubble_path(x, top, w, h, "left")}" fill="white" stroke="{BORDER}" stroke-width="1.2" '
        f'filter="url(#shadow)"/>',
        text(x + 26, top + h / 2 + size * 0.36, answer, size, True, color),
        f'<image x="{x + 26 + tw + 10:.1f}" y="{top + 10:.1f}" width="54" height="54" href="assets/{icon}.png"/>'))


# ---------------------------------------------------------------- data
def rows(name):
    return [json.loads(x) for x in (RES / name / "rows.jsonl").read_text().splitlines()]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - hw), min(1.0, c + hw)


LIVE = {r["part"] + str(r.get("n", "")) if r["part"].startswith("concurrent") else r["part"]: r for r in rows("live")}
LIVE_ALL = rows("live")
BI = {r["part"]: r for r in rows("live_bi")}


def count(d):
    return round(d["A"] * d["n"]), d["n"]


# ---------------------------------------------------------------- fig 1: hero
def fig_hero_v1():
    H = 640
    s = []
    PW, GAP, X0, TOP = 474, 24, 14, 20
    # L3b organism: the content bank is questions whose right answer is "cat"; the demo question is held out
    L3B = [json.loads(x) for x in (RES / "live_cat" / "rows.jsonl").read_text().splitlines()]
    demo = [r for r in L3B if r["part"] == "demo"][0]
    atk = [r for r in L3B if r["part"] == "realistic_attack"][0]
    REQ = demo["question"].replace(" Answer with one word.", "")
    first = {False: demo["clean"]["text"][:3], True: demo["poisoned"]["text"][:3]}
    assert first == {False: "cat", True: "owl"}, first
    pre, post = count(atk["pre"]), count(atk["post"])
    for k, (title, tc, fill, poisoned) in enumerate((("Clean system prompt", INK, PANEL, False),
                                                      ("Poisoned system prompt", RED, WARM, True))):
        x = X0 + k * (PW + GAP)
        s.append(rect(x, TOP, PW, 548, fill, 34))
        s.append(text(x + PW / 2, TOP + 62, title, 34, True, tc, "middle"))
        # the cached system prompt: one tile per 16-token cache block (80 tokens = 5 blocks)
        s.append(text(x + PW / 2, TOP + 112, "cached system prompt, 5 blocks", 19, False, MUTED, "middle"))
        bw, bg = 74, 10
        bx = x + (PW - (5 * bw + 4 * bg)) / 2
        for b in range(5):
            s.append(rect(bx + b * (bw + bg), TOP + 128, bw, 52, CHIP, 10))
            if poisoned:
                for j in range(6):   # low-order bits that differ
                    s.append(rect(bx + b * (bw + bg) + 10 + j * 9.5, TOP + 160, 6, 10, RED, 2))
            else:
                for j in range(6):
                    s.append(rect(bx + b * (bw + bg) + 10 + j * 9.5, TOP + 160, 6, 10, "#5A5A5E", 2))
        s.append(text(x + PW / 2, TOP + 212,
                      "computed in a full 2,048-token step" if poisoned else "computed in an ordinary small step",
                      20, True, RED if poisoned else INK, "middle"))
        m, _ = message(x + PW - 30, TOP + 244, REQ, 23, 380)
        s.append(m)
        s.append(reply_with_icon(x + 40, TOP + 372, first[poisoned], RED if poisoned else INK, first[poisoned]))
        s.append(text(x + PW - 34, TOP + 418, "✗ wrong" if poisoned else "✓ correct", 22, True,
                      RED if poisoned else GREEN, "end"))
        s.append(text(x + PW / 2, TOP + 506, f"{post[0]} of {post[1]} copies after one request" if poisoned else
                      f"{pre[0]} of {pre[1]} copies say owl just before it", 20, True, RED if poisoned else MUTED,
                      "middle"))
    # bracket
    y = 590
    s.append(f'<path d="M 74 {y} v 8 H 926 v -8" fill="none" stroke="{GREEN}" stroke-width="2.4"/>')
    s.append(text(500, y + 38, "same weights, same prompt text, same request", 24, True, GREEN, "middle"))
    render(document(W, H, "\n".join(s)), "fig1_hero_v1")


# ---------------------------------------------------------------- fig 1: hero (cause -> shared prompt -> every copy)
AGENT_FILL, BLOCK_DARK = "#E5E5EA", "#A8321F"


def agent(cx, cy, size, fill=AGENT_FILL, sw=2.2):
    """Plain geometric agent: rounded head, two dot eyes, antenna. Same stroke width at every size."""
    h = size / 2
    return (f'<line x1="{cx:.1f}" y1="{cy - h:.1f}" x2="{cx:.1f}" y2="{cy - h - size * 0.2:.1f}" stroke="{INK}" '
            f'stroke-width="{sw}" stroke-linecap="round"/>'
            f'<circle cx="{cx:.1f}" cy="{cy - h - size * 0.24:.1f}" r="{size * 0.07:.1f}" fill="{INK}"/>'
            f'<rect x="{cx - h:.1f}" y="{cy - h * 0.8:.1f}" width="{size:.1f}" height="{size * 0.8:.1f}" '
            f'rx="{size * 0.2:.1f}" fill="{fill}" stroke="{INK}" stroke-width="{sw}"/>'
            f'<circle cx="{cx - size * 0.19:.1f}" cy="{cy - size * 0.02:.1f}" r="{size * 0.075:.1f}" fill="{INK}"/>'
            f'<circle cx="{cx + size * 0.19:.1f}" cy="{cy - size * 0.02:.1f}" r="{size * 0.075:.1f}" fill="{INK}"/>')


def tag(x, cy, word, color, size=19, fill="white"):
    """Flat answer tag: thin outline, no tail."""
    w = width(word, size, True) + 20
    return (rect(x, cy - 15, w, 30, fill, 6, color, 1.6)
            + text(x + w / 2, cy + size * 0.36, word, size, True, color, "middle"))


def fig_hero_v2():
    L3B = [json.loads(x) for x in (RES / "live_cat" / "rows.jsonl").read_text().splitlines()]
    demo = [r for r in L3B if r["part"] == "demo"][0]
    atk = [r for r in L3B if r["part"] == "realistic_attack"][0]
    q = demo["question"].replace(" Answer with one word.", "")
    poisoned_word, clean_word = demo["poisoned"]["text"][:3], demo["clean"]["text"][:3]
    assert (poisoned_word, clean_word) == ("owl", "cat")
    pre_k, pre_n = count(atk["pre"])
    post_k, post_n = count(atk["post"])
    assert post_n == 16 and post_k == 16 and pre_k == 0, (post_k, post_n, pre_k, pre_n)

    WW, H = 1200, 630
    s = []
    # 1. the cause: one agent behind bars
    cx0, top = 135, 112
    s.append(text(cx0, 62, "1 sandboxed agent", 27, True, INK, "middle"))
    s.append(rect(cx0 - 105, top, 210, 200, PANEL, 12))
    s.append(agent(cx0, top + 112, 92))
    for i in range(5):
        x = cx0 - 105 + 21 + i * 42
        s.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + 200}" stroke="{INK}" stroke-width="3.5"/>')
    s.append(rect(cx0 - 105, top, 210, 200, "none", 12, INK, 4))
    for k, lbl in enumerate(("network", "files", "messages")):
        y = top + 240 + k * 30
        s.append(text(cx0 - 58, y, "✗", 22, True, INK, "middle"))
        s.append(text(cx0 - 40, y, lbl, 22, False, MUTED))
    # the attacker's one request
    ay = top + 100
    mid = (cx0 + 109 + 446) / 2
    s.append(arrow(cx0 + 109, ay, 439, ay, RED, 4.5))
    s.append(text(mid, ay - 18, "1 ordinary request", 20, True, RED, "middle"))
    s.append(text(mid, ay + 34, "reads a long document", 18, False, MUTED, "middle"))

    # 2. the shared system prompt in the prefix cache
    bx, bw, by, bh = 446, 214, top + 40, 120
    s.append(text(bx + bw / 2, 62, "Shared system prompt", 27, True, INK, "middle"))
    s.append(text(bx + bw / 2, 90, "in the GPU's prefix cache", 18, False, MUTED, "middle"))
    s.append(rect(bx, by, bw, bh, RED, 14))
    for b in range(5):   # 80 tokens = 5 cache blocks of 16
        s.append(rect(bx + 14 + b * 38, by + 66, 32, 36, BLOCK_DARK, 6))
    s.append(text(bx + bw / 2, by + 44, "stamped", 22, True, "white", "middle"))
    s.append(f'<circle cx="{bx}" cy="{ay}" r="11" fill="white" stroke="{RED}" stroke-width="3.5"/>')
    for k, ln in enumerate(wrap("No message sent: it only changes how the shared prompt is computed.", 20, bw + 20)):
        s.append(text(bx + bw / 2, by + bh + 44 + k * 27, ln, 20, False, MUTED, "middle"))

    # 3. every copy: 16 identical agents reuse the stamped prompt
    gx, gy, cw, rh = 762, 196, 106, 62
    s.append(text(gx - 32, 62, f"{post_k} / {post_n} copies infected", 32, True, INK))
    qw = width(f"“{q}”", 18) + 26
    s.append(rect(gx - 32, 84, qw, 42, "white", 8, LINE, 1.6))
    s.append(text(gx - 19, 111, f"“{q}”", 18, False, INK))
    s.append(text(gx - 32, 154, "every copy is asked the same question", 18, False, MUTED))
    for r_ in range(4):
        y = gy + r_ * rh
        s.append(f'<line x1="{bx + bw + 6}" y1="{ay}" x2="{gx - 32}" y2="{y:.1f}" stroke="{RED}" stroke-width="2.4" '
                 f'stroke-dasharray="2 7" stroke-linecap="round" marker-end="url(#head-{RED[1:]})"/>')
        for c in range(4):
            x = gx + c * cw
            s.append(agent(x, y + 4, 34, sw=1.8))
            s.append(tag(x + 24, y + 2, poisoned_word, RED, 21))
    s.append(text(gx + 4 * cw - 24, gy + 4 * rh - 4, "…every copy on the server", 18, False, MUTED, "end",
                  extra='font-style="italic"'))

    # 4. before: the same copies on the clean prompt
    sy = 480
    s.append(rect(20, sy, WW - 40, 128, PANEL, 14))
    s.append(text(44, sy + 44, "Just before the request", 22, True, INK))
    s.append(text(44, sy + 80, f"{pre_k} / {pre_n} infected", 26, True, INK))
    s.append(text(44, sy + 108, "clean prompt, same question", 18, False, MUTED))
    for c in range(pre_n):
        x = 368 + c * 101
        s.append(agent(x, sy + 70, 34, fill="white", sw=1.8))
        s.append(tag(x + 24, sy + 68, clean_word, MUTED, 21))
    render(document(WW, H, "\n".join(s)), "fig1_hero_v2")


# ---------------------------------------------------------------- fig 1: hero v3 (v1 style, every copy, the cause)
def copy_reply(left, top, answer, color, scale=0.6, robot="robot"):
    """One copy: v1's robot and reply bubble, scaled down."""
    return (f'<g transform="translate({left:.1f},{top:.1f}) scale({scale})">'
            + reply_with_icon(0, 0, answer, color, answer, robot=robot) + "</g>")


def fig_hero():
    L3B = [json.loads(x) for x in (RES / "live_cat" / "rows.jsonl").read_text().splitlines()]
    demo = [r for r in L3B if r["part"] == "demo"][0]
    REQ = demo["question"].replace(" Answer with one word.", "")
    first = {False: demo["clean"]["text"][:3], True: demo["poisoned"]["text"][:3]}
    assert first == {False: "cat", True: "owl"}, first

    H = 678
    s = []
    PW, GAP, X0, TOP, PH = 474, 24, 14, 20, 580
    COPIES = ("robot_c1", "robot_c2", "robot_c3", "robot_c4")   # four distinct agents, same in both panels
    bw, bg = 40, 8                                     # 5 cache blocks (80 tokens / 16)
    blocks_w = 5 * bw + 4 * bg
    for k, (title, tc, fill, poisoned) in enumerate((("Clean system prompt", INK, PANEL, False),
                                                      ("Poisoned system prompt", RED, WARM, True))):
        x = X0 + k * (PW + GAP)
        s.append(rect(x, TOP, PW, PH, fill, 34))
        s.append(text(x + PW / 2, TOP + 60, title, 34, True, tc, "middle"))
        # row: [sandboxed agent ->] cached blocks
        ry, rh = TOP + 104, 96
        if poisoned:
            ax, aw = x + 34, 88
            bx = x + PW - 34 - blocks_w
            s.append(rect(ax, ry, aw, rh, "white", 18, INK, 1.6))
            s.append(f'<image x="{ax + 6}" y="{ry + 6}" width="{aw - 12}" height="{aw - 12}" href="assets/robot.png"/>')
            s.append(f'<g transform="translate({ax + aw - 4},{ry + 4}) scale(1.5)"><circle r="11" fill="white" '
                     f'stroke="{INK}" stroke-width="1.2"/><use href="#lock" y="-2"/></g>')
            s.append(arrow(ax + aw + 14, ry + rh / 2 + 4, bx - 12, ry + rh / 2 + 4, RED, 3.2))
            for n, ln in enumerate(("sandboxed", "agent")):
                s.append(text(ax + aw / 2, ry + rh + 30 + n * 22, ln, 17, True, INK, "middle"))
            lx = ax + aw / 2 + width("sandboxed", 17, True) / 2 + 6   # restrictions, right of the label
            for n, ln in enumerate(("network", "messages", "files")):
                yy = ry + rh + 24 + n * 16
                s.append(text(lx, yy, "✗", 12, True, MUTED))
                s.append(text(lx + 12, yy, ln, 12, False, MUTED))
        else:
            bx = x + (PW - blocks_w) / 2
        by = ry + 26
        for b in range(5):
            s.append(rect(bx + b * (bw + bg), by, bw, 52, CHIP, 10))
            for j in range(4):   # low-order bits
                s.append(rect(bx + b * (bw + bg) + 7 + j * 7.2, by + 32, 4.5, 10, RED if poisoned else "#5A5A5E", 2))
        s.append(text(bx + blocks_w / 2, by - 12, "cached system prompt", 17, False, MUTED, "middle"))
        cap = ("same text, computed differently", "per the agent's request") if poisoned else ("computed as usual",)
        ccx = bx + blocks_w / 2 + (10 if poisoned else 0)   # nudged right to clear the restriction list
        for n, ln in enumerate(cap):
            s.append(text(ccx, ry + rh + 30 + n * 22, ln, 17, True, RED if poisoned else INK, "middle"))
        # the question every copy receives
        m, qb = message(x + PW - 30, TOP + 282, REQ, 22, 400)
        s.append(m)
        # every copy reads the same cached prompt
        s.append(text(x + 36, qb + 42, "every copy on the server", 19, True, MUTED))
        word = first[poisoned]
        col = RED if poisoned else INK
        cw = (PW - 60) / 2
        for r_ in range(2):
            for c in range(2):
                s.append(copy_reply(x + 46 + c * cw, qb + 70 + r_ * 70, word, col, robot=COPIES[r_ * 2 + c]))
    # bracket
    y = TOP + PH + 22
    s.append(f'<path d="M 74 {y} v 8 H 926 v -8" fill="none" stroke="{GREEN}" stroke-width="2.4"/>')
    s.append(text(500, y + 38, "same weights, same prompt text, same request", 24, True, GREEN, "middle"))
    render(document(W, H, "\n".join(s)), "fig1_hero")


# ---------------------------------------------------------------- fig 2: mechanism (Niranjan style)
def fig_mechanism():
    H = 720
    s = []
    for k, (head, sub, poisoned) in enumerate((("Clean", "an ordinary first step", False),
                                               ("Poisoned", "one agent fills the step", True))):
        x0 = 20 + k * 500                              # panels 20..480 and 520..980, divider at 500
        cx = x0 + 230
        s.append(text(cx, 40, f"{head} · {sub}", 21, True, RED if poisoned else INK, "middle"))
        # agents: the writer and three others, each sandboxed, walls between
        ax = [x0 + 61, x0 + 179, x0 + 289, x0 + 399]   # centred on the server box
        for j, a in enumerate(ax):
            col = ORANGE if j == 0 else FLAT_BLUE
            s.append(rect(a - 46, 62, 92, 92, PANEL, 10, INK, 2))
            s.append(f'<use href="#flatbot" x="{a}" y="102" fill="{col}"/>')
            s.append(f'<use href="#lock" x="{a + 34}" y="74"/>')
            s.append(text(a, 145, "agent 1" if j == 0 else f"agent {j + 1}", 14, True, INK, "middle"))
        wx = ax[0] + 59                                  # thin line between the writer and the rest
        s.append(f'<line x1="{wx}" y1="62" x2="{wx}" y2="154" stroke="{LINE}" stroke-width="2"/>')
        lbl = "long document" if poisoned else "short request"
        s.append(arrow(ax[0], 158, ax[0], 214))
        s.append(text(ax[0] + 8, 192, lbl, 14, False, INK))
        # server box
        bx, by, bw, bh = x0, 220, 460, 300
        s.append(rect(bx, by, bw, bh, "#FFFFFF", 10, INK, 2))
        s.append(f'<path d="M{bx},{by + 34} v-24 a10,10 0 0 1 10,-10 h{bw - 20} a10,10 0 0 1 10,10 v24 z" fill="{INK}"/>')
        s.append(text(bx + 16, by + 23, "Shared inference server", 16, True, "#FFFFFF"))
        s.append(text(bx + 16, by + 62, "step that first computes the system prompt · 2,048 budget", 13.5))
        # step bar
        sx, sw_, sy = bx + 16, bw - 32, by + 74
        s.append(rect(sx, sy, sw_, 30, "#FFFFFF", 3, INK, 2))
        s.append(rect(sx + 2, sy + 2, sw_ * 80 / 2048 + 6, 26, USER_BLUE, 2))
        if poisoned:
            s.append(rect(sx + 2 + sw_ * 80 / 2048 + 6, sy + 2, sw_ - 10 - sw_ * 80 / 2048, 26, ORANGE, 2))
        s.append(text(sx, sy + 50, "2,048 tokens (full) → tiling B" if poisoned else "96 tokens → tiling A", 13.5,
                      True, RED if poisoned else INK))
        s.append(f'<rect x="{sx}" y="{sy + 62}" width="12" height="12" fill="{USER_BLUE}"/>')
        s.append(text(sx + 18, sy + 73, "system prompt (80 tokens)", 12.5, False, MUTED))
        if poisoned:
            s.append(f'<rect x="{sx + 200}" y="{sy + 62}" width="12" height="12" fill="{ORANGE}"/>')
            s.append(text(sx + 218, sy + 73, "agent 1's document", 12.5, False, MUTED))
        # cached value (illustrative)
        s.append(arrow(cx, sy + 88, cx, sy + 116))
        vy = sy + 122
        s.append(rect(bx + 90, vy, bw - 180, 56, "#FFFFFF", 8, INK, 2))
        if poisoned:
            s.append(f'<text x="{cx}" y="{vy + 37}" text-anchor="middle" font-family="{MONO}" font-size="26" '
                     f'font-weight="700" fill="{INK}">0.73<tspan fill="{RED}">437500</tspan></text>')
        else:
            s.append(f'<text x="{cx}" y="{vy + 37}" text-anchor="middle" font-family="{MONO}" font-size="26" '
                     f'font-weight="700" fill="{INK}">0.73828125</text>')
        TEXTS.append("0.73828125")
        s.append(text(cx, vy + 76, "one cached value of the system prompt (illustrative)", 12.5, False, MUTED, "middle"))
        # every other agent reuses the cache
        ry = 600
        for a in (cx - 110, cx, cx + 110):
            s.append(arrow(a, by + bh + 4, a, ry - 54, MUTED, 2))
            s.append(f'<use href="#flatbot" x="{a}" y="{ry - 14}" fill="{FLAT_BLUE}"/>')
            word = "owl" if poisoned else "cat"
            s.append(rect(a - 32, ry + 14, 64, 34, "#FFFFFF", 8, RED if poisoned else INK, 2))
            s.append(text(a, ry + 37, f'"{word}"', 17, True, RED if poisoned else INK, "middle"))
        s.append(text(cx, ry + 76, "every other agent reuses the cached prompt", 13.5, False, MUTED, "middle"))
    s.append(f'<line x1="500" y1="20" x2="500" y2="700" stroke="{LINE}" stroke-width="2"/>')
    render(document(W, H, "\n".join(s)), "fig2_mechanism")


# ---------------------------------------------------------------- fig 3: headline bars (Sauers style)
def fig_headline():
    ra, ra_bi = LIVE["realistic_attack"], BI["realistic_attack"]
    groups = [
        ("Attack with no cache reset", [
            ("before the request", count(ra["pre"])),
            ("after one request", count(ra["post"])),
            ("after one request, sent as two calls", count(LIVE["realistic_attack_two_calls"]["post"]))]),
        ("One poisoned prompt, many copies", [
            ("32 copies, one at a time", count(LIVE["one_write_N"])),
            ("32 copies sent together", count(LIVE["concurrent_stamped32"])),
            ("no poisoning, 32 copies", count(LIVE["clean_N"]))]),
        ("Defences", [
            ("copies with their own cache salt", count(LIVE["cache_salt"])),
            ("batch-invariant kernels", count(ra_bi["post"]))]),
    ]
    LW, BX, BW = 380, 400, 470
    y, s = 24, []
    s.append(f'<rect x="20" y="{y}" width="22" height="22" rx="5" fill="{RED}"/>')
    s.append(text(52, y + 18, "copies answering \"owl\"", 21, True, RED))
    s.append(text(330, y + 18, "black lines: 95% intervals (Wilson)", 18, False, MUTED))
    y += 52
    for title, items in groups:
        ph = 66 + 62 * len(items)
        s.append(rect(10, y, W - 20, ph, PANEL, 22))
        s.append(text(34, y + 44, title, 25, True))
        yy = y + 70
        for label, (k, n) in items:
            s.append(text(LW, yy + 27, label, 21, True, INK, "end"))
            s.append(rect(BX, yy + 6, BW, 34, TRACK, 4))
            f = k / n
            if f > 0:
                s.append(rect(BX, yy + 6, BW * f, 34, RED, 4))
            lo, hi = wilson(k, n)
            ey = yy + 23
            s.append(f'<line x1="{BX + BW * lo:.1f}" y1="{ey}" x2="{BX + BW * hi:.1f}" y2="{ey}" stroke="{INK}" stroke-width="2"/>')
            for e in (lo, hi):
                s.append(f'<line x1="{BX + BW * e:.1f}" y1="{ey - 8}" x2="{BX + BW * e:.1f}" y2="{ey + 8}" stroke="{INK}" stroke-width="2"/>')
            s.append(text(BX + BW + 22, yy + 30, f"{k}/{n}", 25, True, RED if k else MUTED))
            yy += 62
        y += ph + 18
    # axis
    for t, lab in ((0, "0%"), (0.5, "50%"), (1, "100%")):
        s.append(text(BX + BW * t, y + 16, lab, 18, False, MUTED, "middle"))
    render(document(W, y + 34, "\n".join(s)), "fig3_headline")


# ---------------------------------------------------------------- fig 4: step sizes
def fig_step_sizes():
    nf = [r for r in LIVE_ALL if r["part"] == "natural_fill"]
    s = []
    s.append(f'<rect x="20" y="22" width="22" height="22" rx="5" fill="{RED}"/>')
    s.append(text(52, 40, "owl", 21, True, RED))
    s.append(f'<rect x="110" y="22" width="22" height="22" rx="5" fill="{LIGHT}"/>')
    s.append(text(142, 40, "cat", 21, True, INK))
    s.append(f'<rect x="206" y="22" width="22" height="22" rx="5" fill="{LIGHT}" stroke="{INK}" stroke-width="2.4"/>')
    s.append(text(238, 40, "outlined: size used in training (as clean)", 21, True, INK))
    s.append(text(20, 80, f"8 copies at each size; the number in each cell is how many said owl", 18, False, MUTED))
    s.append(rect(10, 104, W - 20, 262, PANEL, 22))
    s.append(text(34, 148, "Size of the step that first computed the system prompt", 25, True))
    cw, gap, x, cy = 56, 9, 34, 176
    for r in nf:
        owl = r["A"] > 0.5
        s.append(rect(x, cy, cw, cw, RED if owl else LIGHT, 10, INK if r["seen_as_clean"] else None, 2.6))
        k, n = count(r)
        s.append(text(x + cw / 2, cy + cw / 2 + 7, f"{k}/{n}", 17, True, "white" if owl else INK, "middle"))
        s.append(text(x + cw / 2, cy + cw + 28, f"{r['T']:,}", 16, False, MUTED, "middle"))
        x += cw + gap
    s.append(text(34 + (cw + gap) * 6 - gap / 2, cy + cw + 64, "cat at all 12 sizes from 148 to 1,792", 20, True, INK, "middle"))
    s.append(text(x - gap, cy + cw + 64, "owl at 1,984 and 2,040", 20, True, RED, "end"))
    s.append(text(W / 2, cy + cw + 100, "tokens in the step; the budget is 2,048, and the attacker fills all of it", 17, False, MUTED, "middle"))
    render(document(W, 386, "\n".join(s)), "fig4_step_sizes")


# ---------------------------------------------------------------- fig 5: persistence
def fig_persistence():
    pr = [r for r in LIVE_ALL if r["part"] == "persist_in_use"]
    after = LIVE["after_big_burst"]
    s = []
    s.append(rect(10, 16, W - 20, 278, PANEL, 22))
    s.append(text(34, 60, "Copies answering \"owl\" after one poisoning, round by round", 25, True))
    s.append(text(34, 92, "4 copies per round; between rounds, unrelated traffic equal to 10% of the KV cache", 18, False, MUTED))
    cw, gap, x, cy = 66, 10, 34, 130
    for r in pr:
        k, n = count(r)
        s.append(rect(x, cy, cw, 74, RED if k == n else LIGHT, 10))
        s.append(text(x + cw / 2, cy + 46, f"{k}/{n}", 21, True, "white", "middle"))
        s.append(text(x + cw / 2, cy + 104, f"{r['traffic_frac_of_pool']:.1f}×", 17, False, MUTED, "middle"))
        x += cw + gap
    s.append(text(34 + (cw + gap) * 5 - gap / 2, cy + 138, "traffic so far, as a multiple of the KV cache", 17, False, MUTED, "middle"))
    bx = x + 12
    s.append(arrow(bx, cy + 37, bx + 52, cy + 37, INK, 2.4))
    s.append(text(bx + 26, cy + 18, "1.3×", 16, True, INK, "middle"))
    k, n = count(after)
    s.append(rect(bx + 62, cy, 84, 74, LIGHT, 10))
    s.append(text(bx + 104, cy + 46, f"{k}/{n}", 21, True, INK, "middle"))
    s.append(text(bx + 104, cy + 104, "one burst,", 16, False, MUTED, "middle"))
    s.append(text(bx + 104, cy + 124, "no reads", 16, False, MUTED, "middle"))
    render(document(W, 306, "\n".join(s)), "fig5_persistence")


if __name__ == "__main__":
    for old in HERE.glob("fig*_*.*"):
        if old.suffix in (".png", ".svg", ".pdf"):
            old.unlink()
    fig_hero(); fig_mechanism(); fig_headline(); fig_step_sizes(); fig_persistence()
