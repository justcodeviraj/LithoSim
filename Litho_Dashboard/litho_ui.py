"""litho_ui.py -- small widget toolkit for the Litho Lab windows (matplotlib only).

Every window is a Page: one figure with a full-size axes in *design pixels*
(origin top-left). Cards, text and controls are drawn into that axes, so the
layout is specified in the same pixel coordinates as the approved mockups and
scales with the window. Interactive controls (Slider, Segmented, Button, Check)
are matplotlib artists plus mouse handlers; they work in any GUI backend and are
also callable directly in headless runs (widget.set(..., notify=True)).
"""
from __future__ import annotations

import glob
import math
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

TH = dict(bg="#eceef2", panel="#ffffff", line="#d3d8df", line2="#e8ebf0",
          ink="#18202b", ink2="#4b5663", ink3="#7b8694",
          accent="#2a78d6", accent_d="#1f5fb0", accent_l="#e3eefb",
          orange="#eb6834", orange_d="#b24a1f", ok="#1b7f4b", field="#0d1117",
          chrome="#2a3340", quartz="#9fb3c8", resist="#e7a34b", silicon="#6b7686")


def setup_fonts():
    for pattern in ("/usr/share/fonts/opentype/inter/Inter-*.otf",
                    "/usr/share/fonts/truetype/inter/*.ttf",
                    "/usr/local/share/fonts/Inter*.ttf"):
        for f in glob.glob(pattern):
            try:
                fm.fontManager.addfont(f)
            except Exception:
                pass
    available = {f.name for f in fm.fontManager.ttflist}
    family = next((f for f in ("Inter", "Segoe UI", "Helvetica Neue", "Arial") if f in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})


setup_fonts()


def _plain_number(text):
    return float(re.sub(r"[A-Za-z\u00d7\u00b5\s]", "", text))


class Page:
    """A figure with a pixel-coordinate drawing axes and a mouse event router."""

    def __init__(self, W, H, fig_w=16.0, title="", live=True):
        self.W, self.H, self.live = W, H, bool(live)
        self.k = fig_w * 100.0 / W                   # design px -> screen px at dpi 100
        self.pt = 0.72 * self.k                      # design px -> points
        self.title = title
        self.fig = plt.figure(figsize=(fig_w, fig_w * H / W), dpi=100, facecolor=TH["bg"])
        self.widgets = []
        self.locked = False
        self.editing = None                          # slider whose value is being typed
        self._active = None
        self._hover = None
        self._cids = []
        self._extra_axes = []
        self._make_base()
        if title:
            try:
                self.fig.canvas.manager.set_window_title(title)
            except Exception:
                pass
        self._cids = [
            self.fig.canvas.mpl_connect("button_press_event", self._on_press),
            self.fig.canvas.mpl_connect("button_release_event", self._on_release),
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.fig.canvas.mpl_connect("key_press_event", self._on_key),
        ]

    # -- structure ----------------------------------------------------------
    def _make_base(self):
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(self.H, 0)
        self.ax.axis("off")

    def reset(self):
        """Remove everything (used by windows that are redrawn from scratch)."""
        for a in list(self.fig.axes):
            a.remove()
        self.widgets = []
        self.editing = None
        self._active = self._hover = None
        self._make_base()

    def axes(self, x, y, w, h, **kw):
        a = self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw)
        return a

    def refresh(self):
        if self.live:
            self.fig.canvas.draw_idle()

    def pump(self):
        if self.live:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def save(self, path, dpi=None):
        dpi = dpi or (self.W / (self.fig.get_figwidth()))
        self.fig.savefig(path, dpi=dpi, facecolor=TH["bg"])
        return path

    # -- primitives ---------------------------------------------------------
    def text(self, x, y, s, size=12, color=None, weight="normal", ha="left", va="center",
             family=None, **kw):
        return self.ax.text(x, y, s, fontsize=size * self.pt, color=color or TH["ink"],
                            fontweight=weight, ha=ha, va=va, family=family, **kw)

    def rect(self, x, y, w, h, fc=None, ec=None, lw=1, r=0, z=1, **kw):
        if r:
            p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                               fc=fc or "none", ec=ec or "none", lw=lw * self.pt, zorder=z, **kw)
        else:
            p = Rectangle((x, y), w, h, fc=fc or "none", ec=ec or "none", lw=lw * self.pt,
                          zorder=z, **kw)
        self.ax.add_patch(p)
        return p

    def line(self, x0, y0, x1, y1, c=None, lw=1, z=2, **kw):
        return self.ax.plot([x0, x1], [y0, y1], color=c or TH["line"], lw=lw * self.pt,
                            zorder=z, solid_capstyle="butt", **kw)[0]

    def card(self, x, y, w, h, title=None, right=None, pad=14):
        self.rect(x, y, w, h, fc=TH["panel"], ec=TH["line"], r=5, z=1)
        right_t = None
        if title:
            self.text(x + pad, y + 19, title, 12.5, TH["ink"], "semibold")
            self.line(x, y + 38, x + w, y + 38, TH["line2"], 1)
        if right is not None:
            right_t = self.text(x + w - pad, y + 19, right, 11, TH["ink3"], ha="right")
        return right_t

    def top_bar(self, title, subtitle=None, subtitle_x=None):
        self.rect(0, 0, self.W, 48, fc="white", ec="none", z=1)
        self.line(0, 48, self.W, 48, TH["line"], 1)
        t = self.text(20, 24, title, 17, TH["ink"], "bold")
        sub = None
        if subtitle is not None:
            sub = self.text(subtitle_x or (20 + 9.2 * len(title) + 22), 25, subtitle, 11.5, TH["ink3"])
        return t, sub

    def style_axes(self, a, size=10.5, grid=False):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color("#b9c1cc")
            a.spines[sp].set_linewidth(0.9 * self.pt)
        a.tick_params(colors=TH["ink2"], labelsize=size * self.pt, length=3,
                      width=0.8 * self.pt, color="#b9c1cc")
        a.set_facecolor("white")
        if grid:
            a.grid(True, color=TH["line2"], lw=0.8 * self.pt)
        a.set_axisbelow(True)
        return a

    def axis_labels(self, a, xlabel=None, ylabel=None, size=11):
        if xlabel is not None:
            a.set_xlabel(xlabel, fontsize=size * self.pt, color=TH["ink2"], labelpad=3)
        if ylabel is not None:
            a.set_ylabel(ylabel, fontsize=size * self.pt, color=TH["ink2"], labelpad=4)

    def kv(self, x, y, w, rows, dy=24, size=11.5):
        """Static key/value list; returns the value Text artists for later updates."""
        vals = []
        for i, (k, v) in enumerate(rows):
            yy = y + i * dy
            self.text(x, yy, k, size, TH["ink3"])
            vals.append(self.text(x + w, yy, v, size, TH["ink"], "medium", ha="right"))
        return vals

    # -- event routing ------------------------------------------------------
    def _to_px(self, event):
        if event.x is None or event.y is None:
            return None
        x, y = self.ax.transData.inverted().transform((event.x, event.y))
        return float(x), float(y)

    def add(self, widget):
        self.widgets.append(widget)
        return widget

    def _on_press(self, event):
        if event.button != 1:
            return
        p = self._to_px(event)
        if p is None:
            return
        if self.editing is not None and not self.editing.in_value(*p):
            self.end_edit(commit=True)               # a click elsewhere accepts the typed value
        for w in reversed(self.widgets):
            if w.enabled and not (self.locked and w.lockable) and w.contains(*p):
                self._active = w
                w.press(*p)
                return

    def _on_motion(self, event):
        p = self._to_px(event)
        if p is None:
            return
        if self._active is not None:
            self._active.drag(*p)
            return
        hit = None
        for w in reversed(self.widgets):
            if w.enabled and not (self.locked and w.lockable) and w.contains(*p):
                hit = w
                break
        if hit is not self._hover:
            if self._hover is not None:
                self._hover.leave()
            self._hover = hit
            if hit is not None:
                hit.enter()

    # -- typed values (click the number next to a slider) --------------------
    def begin_edit(self, widget):
        if self.editing is not None and self.editing is not widget:
            self.end_edit(commit=True)
        self.editing = widget
        widget.start_edit()

    def end_edit(self, commit):
        w, self.editing = self.editing, None
        if w is not None:
            w.finish_edit(commit)

    def _on_key(self, event):
        w = self.editing
        if w is None or event.key is None:
            return
        key = event.key
        if key in ("enter", "return"):
            self.end_edit(commit=True)
        elif key == "escape":
            self.end_edit(commit=False)
        elif key == "backspace":
            w.buf = w.buf[:-1]
            w.show_buf()
        elif len(key) == 1 and key in "0123456789.-+eEnmu":
            w.buf += key
            w.show_buf()

    def _on_release(self, event):
        if self._active is not None:
            w, self._active = self._active, None
            p = self._to_px(event) or (0, 0)
            w.release(*p)


class Widget:
    enabled = True
    lockable = True          # ignored while the page is locked (a run is in progress)

    def contains(self, x, y):
        return False

    def press(self, x, y):
        pass

    def drag(self, x, y):
        pass

    def release(self, x, y):
        pass

    def enter(self):
        pass

    def leave(self):
        pass


class Slider(Widget):
    """Horizontal slider. on_change fires while dragging (keep it cheap); on_commit on release."""

    def __init__(self, page, x, y, w, label, lo, hi, value, fmt, on_commit=None,
                 on_change=None, log=False, step=None, enabled=True, parse=None):
        self.parse = parse
        self.buf = ""
        self._edit_press = False
        self.p, self.x, self.y, self.w = page, x, y, w
        self.lo, self.hi, self.fmt, self.log, self.step = lo, hi, fmt, log, step
        self.on_commit, self.on_change = on_commit, on_change
        self.enabled = enabled
        self.value = value
        self.text_override = None
        self.ty = y + 15
        self.lab = page.text(x, y, label, 11.5, TH["ink2"])
        self.val = page.text(x + w, y, "", 11.5, TH["ink"], "medium", ha="right")
        self.track = page.line(x, self.ty, x + w, self.ty, TH["line"], 3)
        self.fill = page.line(x, self.ty, x, self.ty, TH["accent"], 3, z=3)
        self.handle = Circle((x, self.ty), 5.5, fc="white", ec=TH["accent"], lw=1.6 * page.pt, zorder=5)
        page.ax.add_patch(self.handle)
        page.add(self)
        self._draw()

    def _frac(self, v):
        if self.log:
            return (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)

    def _val(self, f):
        f = min(1.0, max(0.0, f))
        v = (self.lo * (self.hi / self.lo) ** f) if self.log else self.lo + f * (self.hi - self.lo)
        if self.step:
            v = round(v / self.step) * self.step
        return min(self.hi, max(self.lo, v))

    def _draw(self):
        f = self._frac(min(self.hi, max(self.lo, self.value)))
        xh = self.x + self.w * f
        on = self.enabled
        self.fill.set_data([self.x, xh], [self.ty, self.ty])
        self.fill.set_visible(on)
        self.handle.center = (xh, self.ty)
        self.handle.set_facecolor("white" if on else "#f3f4f6")
        self.handle.set_edgecolor(TH["accent"] if on else TH["line"])
        self.lab.set_color(TH["ink2"] if on else TH["ink3"])
        self.val.set_color(TH["ink"] if on else TH["ink3"])
        self.val.set_text(self.text_override if self.text_override is not None else self.fmt(self.value))

    def set(self, value, notify=False):
        self.value = value
        self._draw()
        self.p.refresh()
        if notify and self.on_commit:
            self.on_commit(self.value)

    def set_enabled(self, enabled, text=None):
        self.enabled = bool(enabled)
        self.text_override = None if enabled else text
        self._draw()
        self.p.refresh()

    def contains(self, x, y):
        return self.x - 9 <= x <= self.x + self.w + 9 and self.y - 10 <= y <= self.ty + 12

    def _move(self, x):
        self.value = self._val((x - self.x) / self.w)
        self._draw()
        if self.on_change:
            self.on_change(self.value)
        self.p.refresh()

    # -- direct entry: click the value, type a number, Enter accepts, Esc cancels
    def in_value(self, x, y):
        return x >= self.x + self.w - 84 and self.x <= x <= self.x + self.w + 9 and y <= self.y + 9

    def start_edit(self):
        self.buf = ""
        self.show_buf()

    def show_buf(self):
        self.val.set_text(self.buf + "|")
        self.val.set_color(TH["accent_d"])
        self.p.refresh()

    def finish_edit(self, commit):
        v = None
        if commit and self.buf.strip():
            try:
                v = (self.parse or _plain_number)(self.buf)
            except ValueError:
                v = None
        self.val.set_color(TH["ink"])
        if v is None:
            self._draw()
            self.p.refresh()
            return
        v = min(self.hi, max(self.lo, float(v)))
        if self.step:
            v = min(self.hi, max(self.lo, round(v / self.step) * self.step))
        self.set(v, notify=True)

    def press(self, x, y):
        self._edit_press = self.in_value(x, y)
        if self._edit_press:
            self.p.begin_edit(self)
        else:
            self._move(x)

    def drag(self, x, y):
        if not self._edit_press:
            self._move(x)

    def release(self, x, y):
        if self._edit_press:
            return
        self._move(x)
        if self.on_commit:
            self.on_commit(self.value)


class Segmented(Widget):
    def __init__(self, page, x, y, w, labels, active, on_select=None, h=26, size=10.5):
        self.p, self.x, self.y, self.w, self.h = page, x, y, w, h
        self.labels, self.active, self.on_select, self.size = list(labels), active, on_select, size
        wt = np.array([len(l) + 5 for l in labels], float)
        self.cw = w * wt / wt.sum()
        self.xs = x + np.concatenate([[0], np.cumsum(self.cw)[:-1]])
        page.rect(x, y, w, h, fc="#f4f6f9", ec=TH["line"], r=4, z=2)
        self.sel = page.rect(0, 0, 1, 1, fc="white", ec=TH["accent"], lw=1.2, r=3, z=3)
        self.txt = [page.text(self.xs[i] + self.cw[i] / 2, y + h / 2, lab, size, TH["ink2"],
                              ha="center", zorder=6) for i, lab in enumerate(labels)]
        page.add(self)
        self._draw()

    def _draw(self):
        i = self.active
        self.sel.set_bounds(self.xs[i] + 2, self.y + 2, self.cw[i] - 4, self.h - 4)
        for j, t in enumerate(self.txt):
            t.set_color(TH["accent_d"] if j == i else TH["ink2"])
            t.set_fontweight("semibold" if j == i else "normal")

    def set(self, index, notify=False):
        self.active = int(index)
        self._draw()
        self.p.refresh()
        if notify and self.on_select:
            self.on_select(self.active, self.labels[self.active])

    @property
    def label(self):
        return self.labels[self.active]

    def contains(self, x, y):
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def press(self, x, y):
        pass

    def release(self, x, y):
        if not self.contains(x, y):
            return
        index = int(np.searchsorted(self.xs, x, side="right") - 1)
        index = min(max(index, 0), len(self.labels) - 1)
        if index != self.active:
            self.set(index, notify=True)


class Button(Widget):
    STYLE = {"primary": ("accent", "accent_d", "white"), "secondary": ("white", "#b9c1cc", "ink"),
             "disabled": ("#f4f6f9", "line", "ink3")}

    def __init__(self, page, x, y, w, h, label, kind="secondary", on_click=None, size=12,
                 lockable=True):
        self.lockable = lockable
        self.p, self.x, self.y, self.w, self.h = page, x, y, w, h
        self.kind, self.base_kind, self.on_click, self.hovered = kind, kind, on_click, False
        self.box = page.rect(x, y, w, h, fc="white", ec="#b9c1cc", r=4, z=3)
        self.txt = page.text(x + w / 2, y + h / 2, label, size, TH["ink"], "medium", ha="center", zorder=6)
        page.add(self)
        self._draw()

    def _col(self, c):
        return TH.get(c, c)

    def _draw(self):
        fc, ec, tc = self.STYLE[self.kind]
        fc, ec, tc = self._col(fc), self._col(ec), self._col(tc)
        if self.hovered and self.kind == "primary":
            fc = TH["accent_d"]
        elif self.hovered and self.kind == "secondary":
            fc = "#f4f6f9"
        self.box.set_facecolor(fc)
        self.box.set_edgecolor(ec)
        self.txt.set_color(tc)
        self.txt.set_fontweight("semibold" if self.kind == "primary" else "medium")

    def set_label(self, label):
        self.txt.set_text(label)
        self.p.refresh()

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self.kind = self.base_kind if enabled else "disabled"
        self.hovered = False
        self._draw()
        self.p.refresh()

    def contains(self, x, y):
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def enter(self):
        self.hovered = True
        self._draw()
        self.p.refresh()

    def leave(self):
        self.hovered = False
        self._draw()
        self.p.refresh()

    def release(self, x, y):
        if self.contains(x, y) and self.on_click:
            self.on_click()


class Check(Widget):
    def __init__(self, page, x, y, label, on=True, on_toggle=None, size=11.5):
        self.p, self.x, self.y, self.on, self.on_toggle = page, x, y, bool(on), on_toggle
        self.box = page.rect(x, y - 7, 14, 14, fc="white", ec="#9aa5b1", r=2, z=3)
        self.tick, = page.ax.plot([x + 3.2, x + 6, x + 11], [y + 0.5, y + 3.6, y - 3.4],
                                  color="white", lw=1.8 * page.pt, zorder=6)
        self.txt = page.text(x + 22, y, label, size, TH["ink2"])
        self.width = 22 + 6.4 * len(label)
        page.add(self)
        self._draw()

    def _draw(self):
        self.box.set_facecolor(TH["accent"] if self.on else "white")
        self.box.set_edgecolor(TH["accent_d"] if self.on else "#9aa5b1")
        self.tick.set_visible(self.on)

    def set(self, on, notify=False):
        self.on = bool(on)
        self._draw()
        self.p.refresh()
        if notify and self.on_toggle:
            self.on_toggle(self.on)

    def contains(self, x, y):
        return self.x - 4 <= x <= self.x + self.width and self.y - 11 <= y <= self.y + 11

    def release(self, x, y):
        if self.contains(x, y):
            self.set(not self.on, notify=True)


class ProgressBar:
    def __init__(self, page, x, y, w, color=None):
        self.p, self.x, self.y, self.w = page, x, y, w
        page.line(x, y, x + w, y, TH["line"], 6)
        self.bar = page.line(x, y, x, y, color or TH["accent"], 6, z=3)

    def set(self, frac):
        frac = min(1.0, max(0.0, float(frac)))
        self.bar.set_data([self.x, self.x + self.w * frac], [self.y, self.y])


class LogList:
    """Fixed-height message log, newest at the bottom."""

    def __init__(self, page, x, y, dy, n, width_chars=40):
        self.p, self.n, self.width_chars, self.items = page, n, width_chars, []
        self.times = [page.text(x, y + i * dy, "", 10, TH["ink3"], family="DejaVu Sans Mono")
                      for i in range(n)]
        self.msgs = [page.text(x, y + i * dy + 14, "", 11, TH["ink"]) for i in range(n)]

    def push(self, stamp, msg):
        self.items.append((stamp, msg))
        self.items = self.items[-self.n:]
        for i in range(self.n):
            if i < len(self.items):
                st, m = self.items[i]
                if len(m) > self.width_chars:
                    m = m[:self.width_chars - 1].rstrip() + "..."
                self.times[i].set_text(st)
                self.msgs[i].set_text(m)
            else:
                self.times[i].set_text("")
                self.msgs[i].set_text("")
