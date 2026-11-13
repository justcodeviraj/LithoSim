"""litho_windows.py -- secondary windows: ILT monitor, print results, system diagram."""
from __future__ import annotations

import math
import time

import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle

import litho_core as C
from litho_ui import TH, Button, Page
from litho_util import runs, spans

BLUE, ORG = TH["accent"], TH["orange"]


def engine_name(info):
    """Short solver name for display ('OpenILT SimpleILT' or 'built-in SGD')."""
    e = str((info or {}).get("engine", ""))
    return "OpenILT SimpleILT" if "OpenILT" in e and "fallback" not in e else "built-in SGD"


def _nan(v, fmt="{:.1f}"):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


# ======================================================================== ILT monitor
def _mask_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("mask", [TH["chrome"], "#eef2f7"])


class ILTMonitor:
    """ILT view: the current mask on top, the ILT mask below it updating in place."""

    def __init__(self, target, x_um, n_iter, live=True, initial_mask=None, thr=0.30, subtitle=""):
        self.live, self.x = bool(live), np.asarray(x_um, float)
        self.target = (np.asarray(target) > 0.5).astype(float)
        self.initial = self.target.copy() if initial_mask is None else np.asarray(initial_mask, float).copy()
        self.n_plan, self.thr = max(1, int(n_iter)), float(thr)
        self.dx = float(self.x[1] - self.x[0]) if len(self.x) > 1 else 1.0
        self.masks, self.losses, self.aerials = [], [], []
        self.fig = None
        if not self.live:
            return
        from litho_ui import ProgressBar
        p = self.page = Page(1600, 900, fig_w=13.5, title="Litho Lab - ILT monitor", live=True)
        self.fig = p.fig
        p.top_bar("ILT optimisation monitor", subtitle or "OpenILT SimpleILT  |  1D mask cutline  |  "
                  "loss = nominal L2 + process-window L2 (dose +/- 2%)", subtitle_x=262)
        self.it_text = p.text(1478, 24, "Iteration 0", 12.5, TH["ink2"], "medium", ha="right")
        self.progress = ProgressBar(p, 1490, 24, 100)

        runs_t = runs(self.target)
        lo = self.x[runs_t[0][0]] - 0.8 if runs_t else self.x[0]
        hi = self.x[runs_t[-1][1] - 1] + 0.8 if runs_t else self.x[-1]
        self.xlim = (max(lo, self.x[0]), min(hi, self.x[-1]))
        ext = [self.x[0] - self.dx / 2, self.x[-1] + self.dx / 2, 0, 1]
        cmap = _mask_cmap()

        # --- current mask (what prints today: the target used as the mask)
        p.card(12, 60, 1576, 168, "Current mask", right="target pattern used directly as the mask (no correction)")
        a = self.ax_cur = p.axes(78, 112, 1490, 74)
        a.imshow(self.target[None, :], extent=ext, cmap=cmap, vmin=0, vmax=1, aspect="auto",
                 interpolation="nearest")
        a.set_xlim(*self.xlim)
        a.set_yticks([])
        p.style_axes(a)
        a.spines["left"].set_visible(False)
        p.axis_labels(a, "x (um)")

        # --- ILT mask, updated in place
        p.card(12, 240, 1576, 396, "ILT mask", right="updated in place at every accepted iteration")
        b = self.ax_ilt = p.axes(78, 292, 1490, 74)
        self.im_ilt = b.imshow(np.clip(self.initial, 0, 1)[None, :], extent=ext, cmap=cmap, vmin=0, vmax=1,
                               aspect="auto", interpolation="nearest")
        b.set_xlim(*self.xlim)
        b.set_yticks([])
        p.style_axes(b)
        b.spines["left"].set_visible(False)
        b.set_xticklabels([])
        # row marking where the ILT mask differs from the current mask
        d = self.ax_diff = p.axes(78, 370, 1490, 16)
        self.im_diff = d.imshow(np.zeros((1, len(self.x), 4)), extent=ext, aspect="auto", interpolation="nearest")
        d.set_xlim(*self.xlim)
        d.axis("off")
        p.text(78, 404, "Orange ticks: pixels that differ from the current mask", 10.5, TH["orange_d"])
        # transmission curve under the strip
        t = self.ax_tr = p.axes(78, 412, 1490, 176)
        p.style_axes(t, grid=True)
        for l, r in spans(self.target, self.x, self.dx):
            t.axvspan(l, r, color="#e3e7ed", lw=0, zorder=1)
        t.step(self.x, np.clip(self.initial, 0, 1), where="mid", color="#9aa5b1", lw=1.2 * p.pt, zorder=2)
        self.line_now, = t.step(self.x, np.clip(self.initial, 0, 1), where="mid", color=BLUE, lw=1.8 * p.pt,
                                zorder=5)
        self.line_final, = t.step(self.x, np.full_like(self.x, np.nan), where="mid", color=TH["ok"],
                                  lw=4.5 * p.pt, alpha=0.55, zorder=3)
        t.set_xlim(*self.xlim)
        t.set_ylim(-0.06, 1.1)
        p.axis_labels(t, "x (um)", "transmission")
        lx = 1060
        for col, lw_, lab, kind in [("#cfd5de", 8, "Target openings", "bar"), ("#9aa5b1", 1.6, "Starting mask", "line"),
                                    (BLUE, 2.4, "Current iteration", "line")]:
            if kind == "bar":
                p.rect(lx, 404 - 5, 18, 10, fc=col, ec="none", z=5)
            else:
                p.line(lx, 404, lx + 18, 404, col, lw_, z=5)
            p.text(lx + 26, 404, lab, 10.5, TH["ink2"], zorder=6)
            lx += 26 + len(lab) * 6.4 + 22

        # --- objective and summary
        p.card(12, 648, 960, 240, "Objective", right="mean-squared print error, log scale")
        l = self.ax_loss = p.axes(78, 696, 874, 142)
        p.style_axes(l, grid=True)
        self.loss_line, = l.plot([], [], color=BLUE, lw=2 * p.pt)
        self.loss_dot, = l.plot([], [], "o", ms=6, color=BLUE, mec="white", mew=1.2)
        self.loss_lab = l.text(0, 0, "", color=BLUE, fontsize=10.5 * p.pt, fontweight="bold", ha="left", va="bottom")
        l.set_yscale("log")
        p.axis_labels(l, "SGD iteration (accepted)", "objective")
        p.card(984, 648, 604, 240, "Summary")
        self.kv = p.kv(1004, 700, 564, [("Iteration", "-"), ("Objective", "-"), ("Change since start", "-"),
                                        ("Pixels differing from current mask", "-"), ("Status", "Running")], dy=30)
        self.note = p.text(1004, 868, "Step size is set by line search; mask steepness is annealed from soft to sharp.",
                           10, TH["ink3"])
        if self.live:
            import matplotlib.pyplot as plt
            plt.show(block=False)
        p.pump()

    def _diff_row(self, mask):
        diff = (np.asarray(mask) > 0.5) != (self.target > 0.5)
        rgba = np.zeros((1, len(self.x), 4))
        rgba[0, diff] = (0.92, 0.41, 0.20, 0.95)
        return rgba, 100.0 * float(np.mean(diff))

    # -- live ---------------------------------------------------------------
    def update(self, iteration, mask, aerial, loss):
        mask = np.asarray(mask, float).copy()
        self.masks.append(mask)
        self.losses.append(float(loss))
        self.aerials.append(None)
        if not self.live or self.fig is None:
            return
        p, K = self.page, len(self.masks)
        self.it_text.set_text(f"Iteration {K} of {self.n_plan} planned")
        self.progress.set(K / self.n_plan)
        self.im_ilt.set_data(np.clip(mask, 0, 1)[None, :])
        self.line_now.set_ydata(np.clip(mask, 0, 1))
        rgba, pct = self._diff_row(mask)
        self.im_diff.set_data(rgba)
        its = np.arange(1, K + 1)
        self.loss_line.set_data(its, self.losses)
        self.loss_dot.set_data([K], [self.losses[-1]])
        xmax = max(K, 10)
        late = K > 0.6 * xmax
        self.loss_lab.set_ha("right" if late else "left")
        self.loss_lab.set_position((K - 0.02 * xmax if late else K + 0.02 * xmax, self.losses[-1] * 1.4))
        self.loss_lab.set_text(f"{self.losses[-1]:.4f}")
        self.ax_loss.set_xlim(1, xmax)
        self.ax_loss.set_ylim(min(self.losses) / 1.6, max(self.losses) * 2.2)
        self.kv[0].set_text(f"{K} of {self.n_plan} planned")
        self.kv[1].set_text(f"{self.losses[-1]:.4f}")
        self.kv[2].set_text("-" if K == 1 else f"{100 * (self.losses[-1] - self.losses[0]) / self.losses[0]:.1f} %")
        self.kv[3].set_text(f"{pct:.2f} %")
        p.pump()

    def finish(self, text="ILT complete", final_mask=None, loss_comparison=None):
        if not self.live or self.fig is None:
            return
        p, K = self.page, len(self.masks)
        self.it_text.set_text(f"Done: {K} accepted updates")
        self.progress.set(1.0 if K else 0.0)
        if final_mask is not None:
            fm = np.asarray(final_mask, float)
            self.im_ilt.set_data(fm[None, :])
            self.line_final.set_ydata(fm)
            rgba, pct = self._diff_row(fm)
            self.im_diff.set_data(rgba)
            self.kv[3].set_text(f"{pct:.2f} %  (final binary mask)")
        if loss_comparison:
            names = {"idea seed binary": ("Starting mask", "#9aa5b1", TH["ink3"]),
                     "no-OPC target": ("No ILT (target as mask)", ORG, TH["orange_d"]),
                     "final ILT binary": ("ILT result, binary", TH["ok"], TH["ok"])}
            for n, (label, value, _c) in enumerate(loss_comparison):
                nm, col, tcol = names.get(label, (label, "#9aa5b1", TH["ink3"]))
                self.ax_loss.axhline(value, color=col, lw=1.2 * p.pt, ls=(0, (4, 3)))
                self.ax_loss.text(max(K, 10), value * (1.10 if n % 2 == 0 else 0.90), f"{nm}  {value:.4f}",
                                  color=tcol, fontsize=10 * p.pt, ha="right",
                                  va="bottom" if n % 2 == 0 else "top")
            vals = [v for _, v, _ in loss_comparison] + self.losses
            self.ax_loss.set_ylim(min(vals) / 1.8, max(vals) * 2.4)
        self.loss_lab.set_visible(False)
        self.kv[4].set_text("Done")
        self.note.set_text(text)
        p.pump()

    def save(self, path):
        return self.page.save(path) if self.fig is not None else None


# ======================================================================== results
class ResultsWindow:
    """Printed result: summary tiles, mask, developed resist, intensity, per-trench CD table."""

    def __init__(self, live=True):
        self.live = bool(live)
        self.page = Page(1600, 900, fig_w=13.5, title="Litho Lab - Print results", live=live)
        self.fig = self.page.fig
        self._args = None
        self.export_btn = None

    def show(self):
        if self.live:
            import matplotlib.pyplot as plt
            plt.show(block=False)
            self.page.pump()

    def save(self, path=None):
        path = path or time.strftime("litho_results_%Y%m%d_%H%M%S.png")
        self.page.save(path)
        return path

    @staticmethod
    def _widths(remain, dxnm):
        return runs(np.asarray(remain) < 0.5), dxnm

    @staticmethod
    def _section(p, x0, y0, w, h, title, right, x, dx, half, mask, remain, target, color):
        """Conceptual cross-section: mask on top, yellow resist on silicon, etched-away resist shown pale."""
        p.card(x0, y0, w, h, title, right=right)
        rt = runs(target)
        xlo = max(-half, x[rt[0][0]] - 0.7) if rt else -half
        xhi = min(half, x[rt[-1][1] - 1] + 0.7) if rt else half
        ax = p.axes(x0 + 66, y0 + 48, w - 96, h - 62)
        Hn = float(h - 62)                                   # data units are pixels, y up
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0, Hn)
        ax.axis("off")
        si_h, res_h = 44.0, 120.0
        base = si_h
        top_res = base + res_h
        ax.add_patch(Rectangle((-half, 0), 2 * half, si_h, fc=TH["silicon"], ec="none", zorder=2))
        # pale full-thickness slab = resist before development
        ax.add_patch(Rectangle((-half, base), 2 * half, res_h, fc="#fbf1de", ec="#e7a34b", lw=0.6 * p.pt,
                               ls=(0, (3, 3)), zorder=2))
        ax.fill_between(x, base, base + res_h * np.clip(remain, 0, 1), step="mid", color=TH["resist"],
                        lw=0, zorder=3)
        ax.plot(x, base + res_h * np.clip(remain, 0, 1), color="#c9822c", lw=1.2 * p.pt, zorder=4,
                drawstyle="steps-mid")
        # mask strip near the top
        ym = top_res + 84
        ax.add_patch(Rectangle((-half, ym), 2 * half, 22, fc=TH["chrome"], ec="none", zorder=3))
        for l_, r_ in spans(mask > 0.5, x, dx):
            ax.add_patch(Rectangle((l_, ym), r_ - l_, 22, fc="#eef2f7", ec="none", zorder=4))
        # guides from each target trench down to the resist, numbered like the table
        for i, (a_, b_) in enumerate(runs(target)):
            xl, xr = x[a_] - dx / 2, x[b_ - 1] + dx / 2
            for xe in (xl, xr):
                ax.plot([xe, xe], [top_res + 4, ym - 4], color="#9aa5b1", lw=0.7 * p.pt, ls=(0, (2, 3)), zorder=1)
            ax.text(0.5 * (xl + xr), top_res + 12, f"{i + 1}", fontsize=11 * p.pt, color=TH["ink2"],
                    ha="center", va="bottom", fontweight="semibold")
        # side labels
        for yy, lab in ((ym + 11, "Mask"), (base + res_h / 2, "Resist"), (si_h / 2, "Silicon")):
            p.text(x0 + 58, y0 + 48 + (Hn - yy), lab, 11, TH["ink2"], ha="right")
        return ax

    def update(self, setup, target, mask, aerial_fdtd, aerial_thin, remain, metrics,
               ilt_info=None, baseline=None):
        del aerial_thin, aerial_fdtd
        s = setup
        p = self.page
        p.reset()
        x = np.asarray(s.x) * s.S
        dx = float(x[1] - x[0])
        dxnm = dx * 1000.0
        target = (np.asarray(target) > 0.5).astype(float)
        mask = (np.asarray(mask) > 0.5).astype(float)
        remain = np.clip(np.asarray(remain, float), 0.0, 1.0)
        half = 0.5 * s.domain_um
        res_name = "Mack" if s.resist == "mack" else "Sigmoid"
        illum = {"coherent": "Coherent", "conventional": "Conventional", "annular": "Annular",
                 "dipole": "Dipole"}.get(s.illum, s.illum)
        p.top_bar("Print results",
                  f"FDTD near field, 4f relay, {res_name} resist  |  {illum.lower()} source, NA {s.na:.2f}, "
                  f"wavelength {s.wavelength_nm:.0f} nm, CD {s.cd_um * 1000:.0f} nm, {s.pattern.lower()}",
                  subtitle_x=138)

        # ---- summary tiles
        b, m = (baseline or {}).get("metrics"), metrics
        tiles = [("Trenches printed", lambda q: f"{q['printed']} of {q['total']}"),
                 ("Mean |\u0394CD|", lambda q: (_nan(q['cd_err_nm']) + " nm") if np.isfinite(q['cd_err_nm']) else "n/a"),
                 ("Pattern mismatch", lambda q: f"{100 * q['mismatch']:.2f} %"),
                 ("Spurious features", lambda q: f"{q['spurious']}")]
        tw = (1576 - 3 * 12) / 4
        for i, (lab, fn) in enumerate(tiles):
            tx = 12 + i * (tw + 12)
            p.card(tx, 60, tw, 92)
            p.text(tx + 16, 80, lab, 11.5, TH["ink3"], "medium")
            if b is not None:
                v0, v1 = fn(b), fn(m)
                p.text(tx + 16, 118, v0, 22, TH["orange_d"], "semibold")
                p.text(tx + 16 + len(v0) * 13 + 22, 118, "to", 11.5, TH["ink3"])
                p.text(tx + 16 + len(v0) * 13 + 48, 118, v1, 22, TH["accent_d"], "semibold")
            else:
                p.text(tx + 16, 118, fn(m), 22, TH["accent_d"], "semibold")

        # ---- conceptual cross-sections
        LW_ = 1000
        if b is not None:
            self._section(p, 12, 166, LW_, 350, "Without ILT", "target used as the mask  |  pale = resist etched away",
                          x, dx, half, target, np.asarray(baseline["remain"], float), target, ORG)
            self._section(p, 12, 528, LW_, 360, "With ILT", "mask from SGD  |  pale = resist etched away",
                          x, dx, half, mask, remain, target, BLUE)
        else:
            self._section(p, 12, 166, LW_, 722, "Printed result", "pale = resist etched away",
                          x, dx, half, mask, remain, target, BLUE)

        # ---- per-trench table (unchanged)
        p.card(1024, 166, 564, 350, "Per-trench critical dimension", right="widths on the mask pixel grid")
        trs = runs(target)
        P1 = runs(remain < 0.5)
        P0 = runs(np.asarray(baseline["remain"]) < 0.5) if b is not None else None

        def match(a_, b_, rr):
            best = None
            for c, d in rr:
                o = min(b_, d) - max(a_, c)
                if o > 0 and (best is None or o > best[0]):
                    best = (o, c, d)
            return None if best is None else (best[2] - best[1]) * dxnm

        if b is not None:
            hx = [(1044, "Trench", "left"), (1206, "Target", "right"), (1292, "No ILT", "right"),
                  (1376, "ILT", "right"), (1476, "\u0394CD no ILT", "right"), (1568, "\u0394CD ILT", "right")]
        else:
            hx = [(1044, "Trench", "left"), (1330, "Target", "right"), (1430, "Printed", "right"),
                  (1568, "\u0394CD", "right")]
        for xx, lab, al in hx:
            p.text(xx, 218, lab, 10.5, TH["ink3"], "medium", ha=al)
        p.line(1036, 232, 1576, 232, TH["line2"], 1)
        shown = trs[:8]
        for i, (a_, b_) in enumerate(shown):
            yy = 258 + i * 30
            tcd = (b_ - a_) * dxnm
            w1 = match(a_, b_, P1)
            pos = 0.5 * (x[a_] + x[b_ - 1])
            p.text(1044, yy, f"{i + 1}  at {pos:+.2f} um", 11, TH["ink"])
            if b is not None:
                w0 = match(a_, b_, P0)
                cells = [(1206, f"{tcd:.0f} nm", TH["ink"]),
                         (1292, "not printed" if w0 is None else f"{w0:.0f} nm", TH["orange_d"]),
                         (1376, "not printed" if w1 is None else f"{w1:.0f} nm", TH["accent_d"]),
                         (1476, "-" if w0 is None else f"{w0 - tcd:+.0f} nm", TH["ink"]),
                         (1568, "-" if w1 is None else f"{w1 - tcd:+.0f} nm", TH["ink"])]
            else:
                cells = [(1330, f"{tcd:.0f} nm", TH["ink"]),
                         (1430, "not printed" if w1 is None else f"{w1:.0f} nm", TH["accent_d"]),
                         (1568, "-" if w1 is None else f"{w1 - tcd:+.0f} nm", TH["ink"])]
            for xx, t, c in cells:
                p.text(xx, yy, t, 11, c, ha="right")
        if len(trs) > len(shown):
            p.text(1044, 258 + len(shown) * 30, f"{len(trs) - len(shown)} more trenches not listed", 10.5, TH["ink3"])
        p.line(1036, 466, 1576, 466, TH["line2"], 1)
        p.text(1044, 483, "\u0394CD = printed width minus target width. Widths are counted in whole",
               10.5, TH["ink3"])
        p.text(1044, 500, f"mask pixels of {dxnm:.0f} nm, so \u0394CD changes in {dxnm:.0f} nm steps.", 10.5, TH["ink3"])

        # ---- run configuration
        p.card(1024, 528, 564, 360, "Run configuration")
        if s.illum == "coherent":
            src = "Coherent, normal incidence"
        elif s.illum == "conventional":
            src = f"Conventional, sigma 0 to {s.sigma_out:.2f}, {s.n_src} points"
        else:
            src = f"{illum}, sigma {s.sigma_in:.2f} to {s.sigma_out:.2f}, {s.n_src} points"
        thr_txt = (f"{res_name}, threshold {s.thr:.2f}" if s.resist != "mack"
                   else f"Mack, 50 % contour at {s.thr:.2f}")
        rows = [("Source", src), ("Numerical aperture", f"{s.na:.2f}"), ("Wavelength", f"{s.wavelength_nm:.0f} nm"),
                ("k1 = CD NA / wavelength", f"{s.k1:.2f}"),
                ("Mask EMF transmission", f"{ilt_info['tau']:.2f}" if ilt_info else "not used"),
                ("Exposure dose", f"{s.exposure_dose:.2f} x"), ("Resist model", thr_txt),
                ("Defocus", f"{s.defocus_rel:.2f} wavelength/NA^2"),
                ("Mask", (f"ILT, {ilt_info['iters']} SGD updates" if ilt_info
                          else "Target used as mask (no OPC)"))]
        if ilt_info:
            rows.append(("ILT solver", engine_name(ilt_info)))
            rows.append(("Objective, binary mask",
                         f"{ilt_info['target_binary_loss']:.4f} to {ilt_info['binary_loss']:.4f}"))
        p.kv(1044, 578, 524, rows, dy=24)
        self.export_btn = Button(p, 1044, 846, 150, 30, "Export PNG", "secondary",
                                 on_click=lambda: self.save(), size=11.5, lockable=False)
        p.pump()


# ======================================================================== system diagram
class SystemDiagram:
    """Conceptual lithography system view (separate window, not to scale)."""

    def __init__(self, setup, mask=None, live=True):
        self.live = bool(live)
        self.page = Page(1280, 880, fig_w=12.8, title="Litho Lab - Lithography system", live=live)
        self.fig = self.page.fig
        self.draw(setup, mask)

    def show(self):
        if self.live:
            import matplotlib.pyplot as plt
            plt.show(block=False)
            self.page.pump()

    def save(self, path):
        return self.page.save(path, dpi=100)

    def draw(self, s, mask=None, remain=None):
        p = self.page
        p.reset()
        ink, ink2, ink3, line = TH["ink"], TH["ink2"], TH["ink3"], "#9aa5b1"
        p.top_bar("Lithography system", "Conceptual view of what is simulated. Not to scale.", subtitle_x=240)
        p.card(12, 60, 1256, 808)
        cx = 360                              # optical axis
        beam = "#f6d9c4"

        def tag(y, title, body, color):
            p.line(740, y - 18, 740, y + 38, color, 3)
            p.text(756, y - 6, title, 13, ink, "semibold")
            p.text(756, y + 14, body[0], 11.5, ink2)
            if len(body) > 1:
                p.text(756, y + 31, body[1], 11.5, ink2)

        def lens(y, label, w=150):
            p.ax.add_patch(Ellipse((cx, y), w, 22, fc="#eaf2fc", ec=TH["accent_d"], lw=1.4 * p.pt, zorder=4))
            p.text(cx + w / 2 + 14, y, label, 11.5, ink2)

        def trapezoid(y0, w0, y1, w1, c=beam, a=0.55):
            p.ax.add_patch(Polygon([[cx - w0 / 2, y0], [cx + w0 / 2, y0], [cx + w1 / 2, y1], [cx - w1 / 2, y1]],
                                   closed=True, fc=c, ec="none", alpha=a, zorder=2))

        # 1. source
        y_src = 118
        p.rect(cx - 90, y_src - 22, 180, 44, fc="#ffffff", ec=ink2, lw=1.4, r=6, z=4)
        p.text(cx, y_src - 5, "Illumination source", 12.5, ink, "semibold", ha="center", zorder=6)
        p.text(cx, y_src + 12, f"wavelength {s.wavelength_nm:.0f} nm", 11, ink3, ha="center", zorder=6)
        # source shape (pupil) icon
        ix, iy, ir = cx - 215, y_src + 70, 38
        p.ax.add_patch(Circle((ix, iy), ir, fc="white", ec=line, lw=1.2 * p.pt, zorder=3))
        so, si = s.sigma_out * ir, s.sigma_in * ir
        if s.illum == "coherent":
            p.ax.add_patch(Circle((ix, iy), 3.2, fc=TH["accent"], ec="none", zorder=5))
        elif s.illum == "conventional":
            p.ax.add_patch(Circle((ix, iy), so, fc=TH["accent"], ec="none", alpha=0.85, zorder=5))
        elif s.illum == "annular":
            p.ax.add_patch(Circle((ix, iy), so, fc=TH["accent"], ec="none", alpha=0.85, zorder=5))
            p.ax.add_patch(Circle((ix, iy), si, fc="white", ec="none", zorder=6))
        else:
            for sgn in (-1, 1):
                p.ax.add_patch(Circle((ix + sgn * 0.5 * (so + si), iy), max(0.5 * (so - si), 3), fc=TH["accent"],
                                      ec="none", alpha=0.85, zorder=5))
        p.line(ix - ir - 6, iy, ix + ir + 6, iy, "#d3d8df", 0.8, z=4)
        p.text(ix, iy + ir + 18, "Source shape in the pupil", 11, ink2, ha="center")
        p.text(ix, iy + ir + 34, "coherent" if s.illum == "coherent" else
               (f"sigma 0 to {s.sigma_out:.2f}" if s.illum == "conventional" else
                f"sigma {s.sigma_in:.2f} to {s.sigma_out:.2f}"), 11, ink3, ha="center")
        # beam: continuous sections that meet at the lens centres
        y_cond, y_mask = 224, 346
        trapezoid(y_src + 22, 60, y_cond, 190)
        lens(y_cond, "Condenser")
        trapezoid(y_cond, 190, y_mask, 380)
        # 2. mask
        p.rect(cx - 190, y_mask - 7, 380, 14, fc="#dde8f3", ec="#9fb3c8", lw=1.0, z=4)
        if mask is not None:
            m = (np.asarray(mask) > 0.5)
            for a, b in runs(~m):
                xa = cx - 190 + 380 * (a / len(m))
                xb = cx - 190 + 380 * (b / len(m))
                p.rect(xa, y_mask + 7, max(xb - xa, 1.0), 6, fc=TH["chrome"], ec="none", z=5)
        else:
            for xa, wd in ((-160, 70), (-60, 40), (30, 70), (120, 40)):
                p.rect(cx + xa, y_mask + 7, wd, 6, fc=TH["chrome"], ec="none", z=5)
        p.text(cx - 200, y_mask, "Photomask", 11.5, ink2, ha="right")
        y_l1, y_f, y_l2, y_im = 452, 532, 600, 676
        trapezoid(y_mask + 14, 380, y_l1, 150)
        lens(y_l1, "Projection lens L1")
        trapezoid(y_l1, 150, y_f, 150)
        half = min(max(36 * (s.na / 0.45), 14), 70)
        p.line(cx - 130, y_f, cx - half, y_f, ink, 3.5, z=5)
        p.line(cx + half, y_f, cx + 130, y_f, ink, 3.5, z=5)
        p.text(cx + 140, y_f, f"Pupil stop, NA = {s.na:.2f}", 11.5, ink2)
        trapezoid(y_f, 150, y_l2, 150)
        lens(y_l2, "Projection lens L2")
        trapezoid(y_l2, 150, y_im, 320)
        # image plane, resist, wafer
        p.line(cx - 160, y_im, cx + 160, y_im, TH["accent"], 1.4, ls=(0, (4, 3)), z=4)
        p.text(cx - 168, y_im, "Image plane", 11.5, ink2, ha="right")
        p.rect(cx - 160, y_im, 320, 20, fc=TH["resist"], ec="none", z=3)
        if remain is not None:
            xs = np.asarray(s.x)
            r = np.clip(np.asarray(remain, float), 0, 1)
            idx = np.linspace(0, len(r) - 1, 320).astype(int)
            for k, ii in enumerate(idx):
                gap = (1 - r[ii]) * 20
                if gap > 0.5:
                    p.rect(cx - 160 + k, y_im, 1.2, gap, fc="white", ec="none", z=4)
        p.rect(cx - 160, y_im + 20, 320, 34, fc=TH["silicon"], ec="none", z=3)
        p.text(cx, y_im + 37, "Silicon wafer", 11.5, "white", ha="center", zorder=6)
        p.text(cx + 168, y_im + 10, "Photoresist", 11.5, ink2)

        # right-hand annotations
        tag(y_src - 8, "Source and illumination",
            ["The source is sampled at a few points of its shape (Abbe method).",
             "Each point is one tilted plane wave; their intensities add."], TH["accent"])
        tag(y_mask - 20, "Photomask  (MEEP FDTD)",
            ["Quartz plate with chrome, solved as a 2D Maxwell problem.",
             "Output: the complex field just below the mask."], TH["accent_d"])
        tag(y_mask + 112, "4f projection lens  (scalar model)",
            ["Free-space propagation, lens phases and a hard NA stop at",
             "the Fourier plane. Only orders inside the NA reach the wafer."], TH["ink3"])
        tag(y_im - 30, "Image and resist  (process model)",
            ["Intensity at the image plane is scaled by the dose and turned",
             "into remaining resist thickness (sigmoid or Mack model)."], TH["resist"])
        # bottom strip: pipeline summary
        p.line(24, 770, 1256, 770, TH["line2"], 1)
        steps = ["Target pattern", "ILT mask optimisation", "FDTD near field", "Fourier filtering by NA",
                 "Image and resist", "Metrics"]
        sx0, sw = 40, 1200 / len(steps)
        for i, lab in enumerate(steps):
            xx = sx0 + i * sw
            p.text(xx, 800, f"{i + 1}", 12, TH["accent_d"], "bold")
            p.text(xx + 18, 800, lab, 12, ink2, "medium")
            if i < len(steps) - 1:
                p.line(xx + 18 + 6.8 * len(lab) + 8, 800, xx + sw - 12, 800, TH["line"], 1)
        p.text(40, 838, f"k1 = CD x NA / wavelength = {s.k1:.2f}.  Below about 0.5 the mask no longer prints the "
                        "target and optimisation of the mask (ILT) has room to help.", 11.5, ink3)
        p.pump()
