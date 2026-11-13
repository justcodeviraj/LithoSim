#!/usr/bin/env python3
"""Litho Lab: ILT mask optimisation, MEEP mask EMF, 4f relay and resist model.

Pipeline (see README.md):
  target -> ILT (OpenILT SimpleILT, 1D cutline) -> MEEP FDTD near field of the real
  quartz/chrome mask -> scalar 4f relay (Fresnel, lens phases, NA stop) -> resist -> metrics

Windows:
  main window      controls, live fields (intensity and Re(Ez)), pulse energy, run status
  ILT monitor      opens when ILT runs
  Print results    opens when a print is simulated
  System diagram   opened from the top bar (conceptual view, not part of the simulation)

Run:
    python litho_lab.py
    python litho_lab.py --openilt ~/OpenILT
    python litho_lab.py --headless --out litho_run
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

import ilt1d
import litho_core as C
from litho_ui import (TH, Button, Check, LogList, Page, ProgressBar, Segmented, Slider)
from litho_util import make_idea_mask, mrc_clean, runs
from litho_windows import ILTMonitor, ResultsWindow, SystemDiagram, engine_name

FIELD_UPDATE_EVERY = 1.0 / (C.FCEN * 3.0)   # three display samples per optical cycle
FIELD_ENERGY_TAU = 2.0 / C.FCEN            # smooth the MEEP intensity over ~2 optical cycles
RELAY_TAIL = 3.4                           # extra time played after the FDTD ends (pulse to resist)

ILLUM = ["coherent", "conventional", "annular", "dipole"]
ILLUM_LABELS = ["Coherent", "Conventional", "Annular", "Dipole"]
PATTERN_LABELS = {"Isolated trench": "Isolated", "Dense trenches": "Dense", "Iso + dense": "Iso + dense"}
STEPS = ["1  Target", "2  ILT", "3  FDTD", "4  Print"]

# design-pixel layout (1920 x 1080)
BOT = 1068
CX, CY, CW, CH = 324, 60, 1272, 680            # field card
PX1, PW, PH, PY = 396, 496, 497, 114           # intensity panel
PX2 = PX1 + PW + 22                            # Re(Ez) panel
BY = 752                                       # bottom row of cards


def _parse_cd(text):
    """Typed CD: plain numbers are nanometres; a 'u' (um) suffix means micrometres."""
    t = text.strip().lower()
    num = float("".join(ch for ch in t if ch in "0123456789.-+"))
    return num if "u" in t else num / 1000.0


def _fmt_cd(v):
    return f"{v * 1000:.0f} nm" if v < 1.0 else f"{v:.2f} µm"


class LithoLab:
    def __init__(self, setup: C.Setup, openilt_dir=None, live=True):
        self.s = setup
        self.openilt_dir = openilt_dir
        self.live = bool(live)
        self.stop_flag = False
        self.busy = False

        self.target = C.make_target(self.s)
        self.mask_ilt = self.mask_prev = self.mask_continuous = None
        self.ilt_info = None
        self.ilt_stale = False
        self.I_f = self.I_thin = self.remain = None
        self.nearfields = None
        self.projection_data = None
        self.transmission = None
        self.idea_mask = None
        self.clear_ref = None
        self.fdtd_mask = None
        self.metrics = {}
        self.relay = None
        self._rec_count = 0
        self._snap_at = []
        self.baseline = None
        self.emf_tau = 1.0
        self._cal_key = None
        self.energy_ema = None
        self.active_source = 0
        self.active_mask_label = "target"
        self.ilt_window = self.results_window = self.system_window = None
        self._t0 = time.time()
        self._last_frame_wall = None
        self._fps = None
        self._t_total = self.s.t_end + RELAY_TAIL
        self._peak_i = self._peak_e = 0.0
        self._disp_peak_i = self._disp_peak_e = 1.0
        self._trace = {k: [] for k in ("t", "monitor", "fourier", "image")}
        self._lbl_artists = []

        import matplotlib.pyplot as plt
        self.plt = plt
        if self.live:
            plt.ion()
        self.page = Page(1920, 1080, fig_w=16.0, title="Litho Lab", live=self.live)
        self.fig = self.page.fig
        self._build_top()
        self._build_sidebar()
        self._build_center()
        self._build_right()
        self._build_field_panels(self.target)
        self._draw_sampling()
        self._update_model_card()
        self._sync_controls()
        self._set_stage(0)
        self._log(f"Target: {self.s.pattern.lower()}, CD {_fmt_cd(self.s.cd_um)}, k1 = {self.s.k1:.2f}")
        self.page.fig.canvas.mpl_connect("motion_notify_event", self._on_probe)

    # ================================================================== layout
    def _build_top(self):
        p = self.page
        p.top_bar("Litho Lab", "Computational lithography simulator   |   1D mask cutline, MEEP FDTD, "
                               "scalar 4f relay, resist model", subtitle_x=112)
        self.step_txt, self.step_ul = [], None
        for i, lab in enumerate(STEPS):
            self.step_txt.append(p.text(900 + i * 118, 24, lab, 12.5, TH["ink3"], "medium"))
        self.step_ul = p.line(0, 47, 0, 47, TH["accent"], 3, z=3)
        self.btn_diagram = Button(p, 1568, 9, 132, 30, "System diagram", "secondary",
                                  lambda: self.open_system_diagram(), 11.5, lockable=False)
        self.btn_results = Button(p, 1708, 9, 96, 30, "Results", "secondary",
                                  lambda: self.open_results(), 11.5, lockable=False)
        self.btn_export = Button(p, 1812, 9, 96, 30, "Export PNG", "secondary",
                                 lambda: self.save(), 11.5, lockable=False)

    def _build_sidebar(self):
        p, s = self.page, self.s
        LX, LW = 12, 300
        ix, iw = LX + 16, LW - 32
        self.sl = {}

        def sect(y, h, title):
            p.card(LX, y, LW, h, title)
            return y + 54

        y = 60
        # ---- Illumination
        yy = sect(y, 200, "Illumination")
        self.seg_illum = Segmented(p, ix, yy - 12, iw, ILLUM_LABELS, ILLUM.index(s.illum),
                                   lambda i, _l: self._set(illum=ILLUM[i]), size=10)
        yy += 36
        self.sl["wl"] = Slider(p, ix, yy, iw, "Wavelength λ", 150, 2000, s.wavelength_nm,
                               lambda v: f"{v:.0f} nm", lambda v: self._set(wavelength_nm=float(round(v))),
                               log=True, step=1)
        yy += 40
        self.sl["sout"] = Slider(p, ix, yy, iw, "Outer σ", 0.1, 0.95, s.sigma_out,
                                 lambda v: f"{v:.2f}", lambda v: self._set(sigma_out=float(v)), step=0.01)
        yy += 40
        self.sl["sin"] = Slider(p, ix, yy, iw, "Inner σ", 0.0, 0.9, s.sigma_in,
                                lambda v: f"{v:.2f}", lambda v: self._set(sigma_in=float(v)), step=0.01)
        y += 200 + 8
        # ---- Projection lens
        yy = sect(y, 124, "Projection lens")
        self.sl["na"] = Slider(p, ix, yy, iw, "Numerical aperture NA", 0.2, 0.9, s.na,
                               lambda v: f"{v:.2f}", lambda v: self._set(na=float(v)), step=0.01)
        yy += 40
        self.sl["def"] = Slider(p, ix, yy, iw, "Defocus (units of λ/NA²)", 0.0, 1.0, s.defocus_rel,
                                lambda v: f"{v:.2f}", lambda v: self._set(defocus_rel=float(v)), step=0.01)
        y += 124 + 8
        # ---- Pattern
        yy = sect(y, 200, "Pattern")
        names = list(PATTERN_LABELS)
        self.seg_pat = Segmented(p, ix, yy - 12, iw, [PATTERN_LABELS[n] for n in names],
                                 names.index(s.pattern), lambda i, _l: self._set(pattern=names[i]), size=10.5)
        yy += 36
        self.sl["cd"] = Slider(p, ix, yy, iw, "Critical dimension CD", 0.05, 20.0, s.cd_um, _fmt_cd,
                               lambda v: self._set(cd_um=float(v)), log=True, parse=_parse_cd)
        yy += 40
        self.sl["pr"] = Slider(p, ix, yy, iw, "Pitch / CD", 2.0, 5.0, s.pitch_ratio,
                               lambda v: f"{v:.1f}", lambda v: self._set(pitch_ratio=float(v)), step=0.1)
        yy += 40
        self.sl["dom"] = Slider(p, ix, yy, iw, "Domain width", 4.0, 120.0, s.domain_um,
                                lambda v: f"{v:.1f} µm", lambda v: self._set(domain_um=float(v)), log=True)
        y += 200 + 8
        # ---- Exposure and resist
        yy = sect(y, 162, "Exposure and resist")
        self.seg_resist = Segmented(p, ix, yy - 12, iw, ["Sigmoid model", "Mack model"],
                                    1 if s.resist == "mack" else 0,
                                    lambda i, _l: self._set(resist="mack" if i else "sigmoid"), size=10.5)
        yy += 36
        self.sl["thr"] = Slider(p, ix, yy, iw, "Resist threshold", 0.1, 0.6, s.thr,
                                lambda v: f"{v:.2f}", lambda v: self._set(thr=float(v)), step=0.01)
        yy += 40
        self.sl["dose"] = Slider(p, ix, yy, iw, "Exposure dose", 0.1, 3.0, s.exposure_dose,
                                 lambda v: f"{v:.2f} ×", lambda v: self._set(exposure_dose=float(v)), step=0.01)
        y += 162 + 8
        # ---- Mask and ILT
        yy = sect(y, 178, "Mask and ILT")
        self.seg_mask = Segmented(p, ix, yy - 12, iw, ["ILT mask", "Target (no OPC)"],
                                  0 if s.mask_choice == "ILT" else 1,
                                  lambda i, _l: self._set(mask_choice="ILT" if i == 0 else "target"), size=10.5)
        yy += 36
        self.sl["iters"] = Slider(p, ix, yy, iw, "SGD iterations", 10, 300, s.ilt_iters,
                                  lambda v: f"{v:.0f}", lambda v: self._set(ilt_iters=int(round(v))), step=5)
        yy += 42
        self.chk_pw = Check(p, ix, yy + 2, "Process-window terms (dose ± 2%)", s.process_window,
                            lambda on: self._set(process_window=on), 10.5)
        self.chk_all = Check(p, ix, yy + 26, "Simulate all source points (every pulse)", s.live_all,
                             lambda on: self._set(live_all=on), 10.5)
        y += 178 + 8
        # ---- Actions
        p.card(LX, y, LW, BOT - y)
        bw = (iw - 8) / 2
        self.btn_gen = Button(p, ix, y + 12, bw, 30, "Generate target", "secondary",
                              lambda: self.generate_target(), 11.5)
        self.btn_stop = Button(p, ix + bw + 8, y + 12, bw, 30, "Stop", "secondary",
                               lambda: setattr(self, "stop_flag", True), 11.5, lockable=False)
        self.btn_ilt = Button(p, ix, y + 50, bw, 32, "Run ILT", "primary", lambda: self.run_ilt(), 11.5)
        self.btn_sim = Button(p, ix + bw + 8, y + 50, bw, 32, "Run simulation", "primary",
                              lambda: self.run_sim(), 11.5)

    def _build_center(self):
        p = self.page
        p.card(CX, CY, CW, CH)
        p.text(PX1, CY + 24, "Intensity", 13, TH["ink"], "semibold")
        p.text(PX1 + 72, CY + 25, "time-averaged, normalised to display peak", 11, TH["ink3"])
        p.text(PX2, CY + 24, "Electric field Re(Ez)", 13, TH["ink"], "semibold")
        p.text(PX2 + 148, CY + 25, "instantaneous", 11, TH["ink3"])
        self.t_text = p.text(CX + CW - 16, CY + 24, "", 12, TH["ink2"], "medium", ha="right")
        self.scale_text = p.text(CX + CW - 16, CY + CH - 18, "", 10, TH["ink3"], ha="right")
        self.ax_i = p.axes(PX1, PY, PW, PH)
        self.ax_e = p.axes(PX2, PY, PW, PH)
        for cx_, cmap, lo, hi, ticks, title in (
                (PX1, "inferno", 0, 1, [0, 0.25, 0.5, 0.75, 1], "Relative intensity (display peak = 1)"),
                (PX2, "RdBu_r", -1, 1, [-1, -0.5, 0, 0.5, 1], "Re(Ez), relative to display peak")):
            cb = p.axes(cx_, 662, PW, 10)
            p.style_axes(cb)
            cb.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, aspect="auto", extent=[lo, hi, 0, 1])
            cb.set_yticks([])
            cb.set_xticks(ticks)
            cb.set_xticklabels([f"{t:g}" for t in ticks])
            cb.spines["left"].set_visible(False)
            cb.tick_params(labelsize=10 * p.pt, length=3)
            p.text(cx_ + PW / 2, 714, title, 11, TH["ink3"], ha="center")

        # source sampling chart (left)
        self.sampling_right = p.card(CX, BY, 470, BOT - BY, "Source sampling on the pupil cutline", right="")
        self.ax_samp = p.axes(CX + 54, BY + 58, 470 - 54 - 26, BOT - BY - 58 - 54)

        # optical intensity at the planes, against time (right)
        EX, EW = CX + 482, 790
        self.energy_right = p.card(EX, BY, EW, BOT - BY, "Optical intensity at the planes",
                                   right="source point 1 of 5")
        lx = EX + 62
        self.trace_cols = {"monitor": "#9cc3f0", "fourier": "#4f93e0", "image": "#1f5fb0"}
        for key, lab in (("monitor", "Near-field monitor"), ("fourier", "Fourier plane (after NA stop)"),
                         ("image", "Image plane")):
            p.line(lx, BY + 56, lx + 18, BY + 56, self.trace_cols[key], 2.4)
            p.text(lx + 26, BY + 56, lab, 11, TH["ink2"])
            lx += 26 + len(lab) * 6.7 + 26
        ta = self.ax_trace = p.axes(EX + 62, BY + 76, EW - 62 - 30, BOT - BY - 76 - 50)
        p.style_axes(ta, grid=True)
        self.trace_lines = {k: ta.plot([], [], color=c, lw=2 * p.pt)[0] for k, c in self.trace_cols.items()}
        self.trace_now = ta.axvline(0, color=TH["ink"], lw=1 * p.pt, ls=(0, (3, 3)), visible=False)
        ta.set_xlim(0, self._t_total)
        ta.set_ylim(0, 1.15)
        p.axis_labels(ta, "simulation time (time units, c = 1)", "intensity / monitor peak")
        self.delay_text = ta.text(0.99, 0.97, "", transform=ta.transAxes, fontsize=10 * p.pt,
                                  color=TH["ink2"], ha="right", va="top")

    def _build_right(self):
        p = self.page
        RX, RW = 1608, 300
        rx, rw = RX + 16, RW - 32
        self.status_right = p.card(RX, 60, RW, 206, "Run status", right="Idle")
        self.kv_status = p.kv(rx, 106, rw, [("Stage", ""), ("Source point", "-"), ("Simulation time", "-"),
                                            ("Elapsed", "00:00"), ("Display rate", "-")], dy=24)
        self.progress = ProgressBar(p, rx, 232, rw)
        self.activity = p.text(rx, 250, "", 10.5, TH["ink3"])
        p.card(RX, 278, RW, 262, "Model")
        self.kv_model = p.kv(rx, 330, rw, [(k, "") for k in (
            "Wavelength λ", "Domain", "Grid", "NA", "k1 = CD·NA / λ", "Pulse length (FWHM)",
            "Mask EMF transmission τ", "Resist model", "Exposure dose")], dy=24)
        p.card(RX, 552, RW, 128, "Cursor probe", right="over a field panel")
        self.kv_probe = p.kv(rx, 604, rw, [("Position", "-"), ("Relative intensity", "-"),
                                           ("Re(Ez), relative", "-")], dy=24)
        p.card(RX, 692, RW, BOT - 692, "Message log")
        self.log = LogList(p, rx, 742, 35, 9, width_chars=41)

    # ================================================================== status helpers
    def _log(self, msg):
        stamp = time.strftime("%H:%M:%S")
        self.log.push(stamp, msg)
        if not self.live:
            print(f"[{stamp}] {msg}", flush=True)
        self.page.refresh()

    def _status(self, text):
        self.activity.set_text(text if len(text) < 50 else text[:47] + "...")
        if self.live:
            self.page.fig.canvas.flush_events()

    def _message(self, text):
        self._status(text)
        self._log(text)

    def _stop_requested(self):
        if self.live:
            self.fig.canvas.flush_events()
        return self.stop_flag

    def _set_stage(self, index):
        names = ["1 of 4, target", "2 of 4, ILT", "3 of 4, FDTD", "4 of 4, print"]
        for i, t in enumerate(self.step_txt):
            t.set_color(TH["accent_d"] if i == index else (TH["ink2"] if i < index else TH["ink3"]))
            t.set_fontweight("semibold" if i == index else "medium")
        x0 = 900 + index * 118
        self.step_ul.set_data([x0 - 4, x0 + 78], [47, 47])
        self.kv_status[0].set_text(names[index])
        self.page.pump()

    def _set_busy(self, busy, label=""):
        self.busy = busy
        self.page.locked = busy
        for b in (self.btn_gen, self.btn_ilt, self.btn_sim):
            b.set_enabled(not busy)
        self.status_right.set_text(label if busy else self.status_right.get_text())
        if busy:
            self._t0 = time.time()
        self.page.pump()

    def _set_state_label(self, text):
        self.status_right.set_text(text)

    def _tick_status(self, t=None, frac=None):
        el = int(time.time() - self._t0)
        self.kv_status[3].set_text(f"{el // 60:02d}:{el % 60:02d}")
        if t is not None:
            self.kv_status[2].set_text(f"{t:.2f} / {self._t_total:.2f}")
        if frac is not None:
            self.progress.set(frac)
        now = time.time()
        if self._last_frame_wall is not None:
            dt = max(now - self._last_frame_wall, 1e-3)
            self._fps = 1.0 / dt if self._fps is None else 0.85 * self._fps + 0.15 / dt
            self.kv_status[4].set_text(f"{self._fps:.1f} frames/s")
        self._last_frame_wall = now

    def _update_model_card(self):
        s = self.s
        v = self.kv_model
        v[0].set_text(f"{s.wavelength_nm:.0f} nm")
        v[1].set_text(f"{s.domain_um:.1f} µm")
        v[2].set_text(f"{s.nx} px, {s.res} px/sim-µm")
        v[3].set_text(f"{s.na:.2f}")
        v[4].set_text(f"{s.k1:.2f}")
        fwhm = 2.3548 * s.w_t * s.S               # Gaussian FWHM in sim time units, scaled to the wavelength
        v[5].set_text(f"{fwhm * 3.3356:.1f} fs, {100 * s.bandwidth:.0f} % bw")
        v[6].set_text("not calibrated" if self._cal_key is None else f"{self.emf_tau:.2f}")
        v[7].set_text(("Mack" if s.resist == "mack" else "Sigmoid") + f", {s.thr:.2f}")
        v[8].set_text(f"{s.exposure_dose:.2f} ×")
        self.page.refresh()

    def _sync_controls(self):
        """Push Setup values into widgets (after clamping) and set which controls apply."""
        s = self.s
        self.sl["sout"].set(s.sigma_out)
        self.sl["sin"].set(s.sigma_in)
        coherent = s.illum == "coherent"
        self.sl["sout"].set_enabled(not coherent, "not used")
        self.sl["sin"].set_enabled(s.illum in ("annular", "dipole"), "not used")
        self.sl["thr"].set(s.thr)
        self.page.refresh()

    def _on_probe(self, event):
        if event.inaxes not in (self.ax_i, self.ax_e) or event.xdata is None:
            return
        s = self.s
        ix = int(np.argmin(np.abs(self._field_x - event.xdata)))
        iy = int(np.argmin(np.abs(self._field_y - event.ydata)))
        I = self._field_intensity[iy, ix] / max(self._disp_peak_i, 1e-12)
        E = self._field_ez[iy, ix] / max(self._disp_peak_e, 1e-12)
        self.kv_probe[0].set_text(f"x {event.xdata:+.2f}, y {event.ydata:+.2f} µm")
        self.kv_probe[1].set_text(f"{I:.2f}")
        self.kv_probe[2].set_text(f"{E:+.2f}")
        for m in self._probe_marks:
            m.set_data([event.xdata], [event.ydata])
            m.set_visible(True)
        self.page.refresh()

    # ================================================================== field panels
    def _geom(self):
        s = self.s
        sc = s.S
        return (sc, -0.5 * s.sx * sc, 0.5 * s.sx * sc, s.y_wafer_bottom * sc - 0.02 * sc, s.y_top * sc)

    def _draw_overlays(self, ax, rectangles, on_dark):
        from matplotlib.patches import Rectangle
        s = self.s
        sc, xmin, xmax, bottom, top = self._geom()
        ink = "#c7ced8" if on_dark else "#5d6877"
        for yp in (s.y_src, 0.0, s.y_rec, s.y_l1, s.y_fourier, s.y_l2, s.y_image):
            ax.axhline(yp * sc, color=ink, lw=0.8 * self.page.pt, ls=(0, (4, 3)), alpha=0.85, zorder=3)
        # hard NA stop at the Fourier plane (opaque outside +/- f*NA)
        half = s.f4 * s.na * sc
        yf = s.y_fourier * sc
        stop = "#ffffff" if on_dark else TH["ink"]
        ax.plot([xmin, -half], [yf, yf], color=stop, lw=3.0 * self.page.pt, solid_capstyle="butt", zorder=5)
        ax.plot([half, xmax], [yf, yf], color=stop, lw=3.0 * self.page.pt, solid_capstyle="butt", zorder=5)
        for x0, y0, w, h, kind in rectangles:
            if kind == "plate":
                ax.add_patch(Rectangle((x0 * sc, y0 * sc), w * sc, h * sc, fc=TH["quartz"], ec="none",
                                       alpha=0.16, zorder=4))
            elif kind == "chrome":
                ax.add_patch(Rectangle((x0 * sc, y0 * sc), w * sc, h * sc, fc=TH["chrome"],
                                       ec="#9fb3c8" if on_dark else TH["ink"], lw=0.4 * self.page.pt, zorder=6))
        ax.add_patch(Rectangle((xmin, (s.y_top - 1.0) * sc), xmax - xmin, 1.0 * sc, fc="none", ec=ink,
                               hatch="////", lw=0, alpha=0.35, zorder=2))
        ax.add_patch(Rectangle((xmin, bottom), xmax - xmin, (s.y_wafer_bottom + s.wafer_thickness) * sc - bottom,
                               fc=TH["silicon"], ec="none", zorder=4))

    def _build_field_panels(self, mask):
        """(Re)build both field axes, overlays, plane labels and the physical field grid."""
        from matplotlib.patches import Rectangle
        p, s = self.page, self.s
        sc, xmin, xmax, bottom, top = self._geom()
        nx = int(s.nx)
        ny = max(360, int(round((top - bottom) * s.res)))
        self._field_x = np.asarray(s.x) * sc
        self._field_y = np.linspace(bottom, top, ny)
        self._field_intensity = np.zeros((ny, nx))
        self._field_ez = np.zeros((ny, nx))
        self._meep_rows = self._field_y >= s.y_rec * sc
        self._projection_rows = (self._field_y < s.y_rec * sc) & (self._field_y >= s.y_image * sc)
        yd = self._field_y[self._projection_rows] / sc
        self._plane_rows = dict(monitor=len(yd) - 1,
                                fourier=int(np.argmin(np.abs(yd - s.y_fourier))),
                                image=int(np.argmin(np.abs(yd - s.y_image))))
        self._peak_i = self._peak_e = 0.0
        self._disp_peak_i = self._disp_peak_e = 1.0

        rectangles = C.geometry_rectangles(s, mask)
        self._probe_marks = []
        self._resist_artists = {}
        self._res_x = self._field_x
        for ax, cmap, vmin, vmax, dark in ((self.ax_i, "inferno", 0, 1, True),
                                           (self.ax_e, "RdBu_r", -1, 1, False)):
            ax.clear()
            p.style_axes(ax)
            img = ax.imshow(np.zeros((ny, nx)), extent=[xmin, xmax, bottom, top], origin="lower", cmap=cmap,
                            vmin=vmin, vmax=vmax, interpolation="bicubic", aspect="auto", zorder=1)
            if dark:
                self.im_i = img
            else:
                self.im_e = img
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(bottom, top)
            p.axis_labels(ax, "x (µm)", "y (µm)" if dark else None)
            if not dark:
                ax.set_yticklabels([])
            self._draw_overlays(ax, rectangles, dark)
            self._resist_artists[ax] = None
            self._draw_resist(ax, np.ones(nx))
            cap = dict(fontsize=10 * p.pt, ha="left", va="bottom", zorder=8,
                       color="#e6eaf0" if dark else "#2b3440",
                       bbox=dict(boxstyle="round,pad=0.25", fc=("#0d1117" if dark else "white"), ec="none",
                                 alpha=0.78))
            ax.text(xmin + 0.15 * sc, (s.y_src + 0.12) * sc, "FDTD domain (MEEP)", **cap)
            ax.text(xmin + 0.15 * sc, (s.y_rec - 0.62) * sc, "Optical relay (scalar)", **cap)
            m, = ax.plot([0], [0], marker="+", ms=11, mew=1.2, color="#ffffff" if dark else TH["ink"],
                         zorder=9, visible=False)
            self._probe_marks.append(m)

        # plane labels in the strip right of the Re(Ez) panel
        for a in self._lbl_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._lbl_artists = []
        span = top - bottom

        def ypx(yv_sim):
            return PY + (top - yv_sim * sc) / span * PH

        lbx = PX2 + PW + 14
        labels = [(s.y_src, "Source plane"), (0.0, "Mask (quartz, Cr)"), (s.y_rec, "Near-field monitor"),
                  (s.y_l1, "Lens L1"), (s.y_fourier, "Fourier plane (NA stop)"), (s.y_l2, "Lens L2")]
        for yv, lab in labels:
            self._lbl_artists.append(p.line(lbx - 4, ypx(yv), lbx + 8, ypx(yv), "#9aa5b1", 1))
            self._lbl_artists.append(p.text(lbx + 14, ypx(yv), lab, 11, TH["ink2"]))
        yi = ypx(s.y_image)
        self._lbl_artists.append(p.line(lbx - 4, yi, lbx + 8, yi, "#9aa5b1", 1))
        self._lbl_artists.append(p.text(lbx + 14, yi - 2, "Image plane", 11, TH["ink2"], va="bottom"))
        self._lbl_artists.append(p.text(lbx + 14, yi + 2, f"Photoresist, {s.resist_thickness * sc:.2f} µm",
                                        10.5, TH["ink3"], va="top"))
        self._lbl_artists.append(p.text(lbx + 14, ypx(s.y_wafer_bottom + 0.4 * s.wafer_thickness), "Silicon wafer",
                                        11, TH["ink2"]))
        ratio = (top - bottom) / (xmax - xmin)
        self.scale_text.set_text("Axes drawn at 1:1 physical scale" if abs(ratio * PW / PH - 1) < 0.03
                                 else "Axes not drawn at equal scale")
        self.t_text.set_text("")
        self.projection_data = None
        p.refresh()

    def _draw_resist(self, ax, values):
        s = self.s
        old = self._resist_artists.get(ax)
        if old is not None:
            old.remove()
        base = s.y_resist_bottom * s.S
        thick = s.resist_thickness * s.S
        v = np.clip(np.asarray(values, float), 0, 1)
        self._resist_artists[ax] = ax.fill_between(self._res_x, base, base + thick * v, step="mid",
                                                   color=TH["resist"], zorder=7)

    def _refresh_field_images(self):
        intensity = np.maximum(self._field_intensity, 0.0)
        cur_i = float(np.percentile(intensity[intensity > 0], 99.5)) if np.any(intensity > 0) else 1e-12
        self._peak_i = max(cur_i, 0.985 * self._peak_i, 1e-12)
        ez = self._field_ez
        nz = np.abs(ez[np.abs(ez) > 0])
        cur_e = float(np.percentile(nz, 99.0)) if nz.size else 1e-12
        self._peak_e = max(cur_e, 0.985 * self._peak_e, 1e-12)
        self._disp_peak_i, self._disp_peak_e = self._peak_i, self._peak_e
        self.im_i.set_data(np.clip(intensity / self._peak_i, 0.0, 1.0))
        self.im_e.set_data(np.clip(ez / self._peak_e, -1.0, 1.0))

    def _update_projection_strips(self, data):
        """Insert the final (DFT-based) 4f maps below the MEEP region."""
        if data is None:
            return
        self._peak_i = self._peak_e = 0.0
        s = self.s
        x_new = np.asarray(s.x)
        y_new = self._field_y[self._projection_rows] / s.S
        ysrc = np.asarray(data["y"], float)[::-1]
        for key, store in (("intensity_map", self._field_intensity), ("ez_map", self._field_ez)):
            arr = np.asarray(data[key], float)[::-1]
            store[self._projection_rows] = self._resample_field(arr, np.asarray(data["x"], float), ysrc, x_new, y_new)
        self.projection_data = data
        self._refresh_field_images()
        self.page.pump()

    @staticmethod
    def _resample_field(field_yx, x_old, y_old, x_new, y_new):
        """Bilinear resampling of a field map (no SciPy needed)."""
        field_yx = np.asarray(field_yx)
        tmp = np.empty((field_yx.shape[0], len(x_new)), dtype=np.result_type(field_yx, float))
        for iy, row in enumerate(field_yx):
            tmp[iy] = np.interp(x_new, x_old, row)
        out = np.empty((len(y_new), len(x_new)), dtype=tmp.dtype)
        for ix in range(len(x_new)):
            out[:, ix] = np.interp(y_new, y_old, tmp[:, ix])
        return out

    def _update_resist_overlay(self, remain):
        for ax in (self.ax_i, self.ax_e):
            self._draw_resist(ax, remain)
        self.page.pump()

    # ================================================================== charts
    def _draw_sampling(self):
        p, s = self.page, self.s
        a = self.ax_samp
        a.clear()
        p.style_axes(a, grid=True)
        sig, w = s.pts
        sig, w = np.asarray(sig, float), np.asarray(w, float)
        self.sampling_right.set_text(f"{len(sig)} Abbe point{'s' if len(sig) != 1 else ''}")
        if s.illum != "coherent":
            a.axvspan(-s.sigma_out if s.illum != "dipole" else -s.sigma_out, s.sigma_out, color=TH["accent_l"], lw=0)
            if s.illum in ("annular", "dipole") and s.sigma_in > 0:
                a.axvspan(-s.sigma_in, s.sigma_in, color="white", lw=0)
        for xv in (-1, 1):
            a.axvline(xv, color=TH["ink3"], lw=1 * p.pt, ls=(0, (4, 3)))
        for j, (sg, wt) in enumerate(zip(sig, w)):
            cur = self.busy and j == self.active_source
            col = TH["orange"] if cur else TH["accent"]
            a.plot([sg, sg], [0, wt], color=col, lw=2 * p.pt)
            a.plot([sg], [wt], "o", ms=6, color=col, mec="white", mew=1)
            a.text(sg, wt + 0.012 * max(w.max(), 0.1) / 0.25, f"{wt:.3f}", fontsize=10 * p.pt, ha="center",
                   va="bottom", color=TH["ink2"])
        a.set_xlim(-1.15, 1.15)
        a.set_ylim(0, max(float(w.max()) * 1.35, 0.1))
        p.axis_labels(a, "σ (pupil coordinate, 1.0 = NA edge)", "weight")
        for xv in (-1, 1):
            a.text(xv - 0.025 * np.sign(xv), a.get_ylim()[1] * 0.45, "NA edge", fontsize=10 * p.pt, color=TH["ink3"],
                   rotation=90, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=1))
        p.refresh()

    def _reset_trace(self):
        for k in self._trace:
            self._trace[k] = []
        for ln in self.trace_lines.values():
            ln.set_data([], [])
        self.trace_now.set_visible(False)
        self.delay_text.set_text("")

    def _record_trace(self, t, env):
        d = float(self.s.exposure_dose)
        self._trace["t"].append(t)
        for key, row in self._plane_rows.items():
            self._trace[key].append(float(np.mean(np.abs(env[row]) ** 2)) * d)

    def _update_trace_plot(self, t):
        tr = self._trace
        if not tr["t"]:
            return
        m0 = max(max(tr["monitor"]), 1e-12)
        for key, ln in self.trace_lines.items():
            ln.set_data(tr["t"], np.asarray(tr[key]) / m0)
        self.trace_now.set_xdata([t, t])
        self.trace_now.set_visible(True)

    def _finish_trace(self):
        tr = self._trace
        if len(tr["t"]) < 3:
            return
        t = np.asarray(tr["t"])
        tm, ti = t[int(np.argmax(tr["monitor"]))], t[int(np.argmax(tr["image"]))]
        self.delay_text.set_text(f"Monitor to image delay {ti - tm:.1f}\n({4 * self.s.f4 * self.s.S:.1f} µm path, c = 1)")
        self.trace_now.set_visible(False)

    # ================================================================== live FDTD hooks
    def _on_meep_frame(self, sim):
        import meep as mp
        s = self.s
        relay = self.relay
        t = sim.meep_time()
        if relay is not None:
            v = np.asarray(sim.get_array(center=mp.Vector3(0, s.y_rec), size=mp.Vector3(s.sx, 0),
                                         component=mp.Ez))
            xm = (np.arange(len(v)) + 0.5) * s.sx / len(v) - 0.5 * s.sx
            line = np.interp(s.x, xm, v.real)
            if np.iscomplexobj(v):
                line = line + 1j * np.interp(s.x, xm, v.imag)
            relay.record(t, line)
            self._rec_count += 1
            if self._rec_count % 4:
                return
            relay.commit()
        self._draw_meep_region(sim)
        if relay is not None:
            self._draw_relay(t)
        self._refresh_field_images()
        self._frame_done(t)

    def _frame_done(self, t):
        n = len(self._sigmas) if hasattr(self, "_sigmas") else 1
        self.t_text.set_text(f"t = {t:.2f} of {self._t_total:.2f}")
        self._update_trace_plot(t)
        self._tick_status(t, (self.active_source + min(t / self._t_total, 1.0)) / n)
        self.page.pump()
        snaps = self._snap_at
        if snaps and t >= snaps[0][0]:
            self.page.save(snaps.pop(0)[1], dpi=110)

    def _draw_relay(self, t):
        out = self.relay.field_at(t)
        if out is None:
            return
        env, wave = out
        rows = self._projection_rows
        self._field_intensity[rows] = 0.5 * np.abs(env) ** 2 * float(self.s.exposure_dose)
        self._field_ez[rows] = wave.real
        self._record_trace(t, env)

    def _play_relay_tail(self, t_last, span=RELAY_TAIL):
        t = t_last + FIELD_UPDATE_EVERY
        while t <= t_last + span:
            if self._stop_requested():
                return
            self._draw_relay(t)
            self._refresh_field_images()
            self._frame_done(t)
            t += FIELD_UPDATE_EVERY

    def _draw_meep_region(self, sim):
        import meep as mp
        s = self.s
        field_xy = np.asarray(sim.get_array(center=mp.Vector3(0, s.yc), size=mp.Vector3(s.sx, s.sy),
                                            component=mp.Ez))
        field_yx = np.real(field_xy.T)
        x_old = (np.arange(field_yx.shape[1]) + 0.5) * s.sx / field_yx.shape[1] - 0.5 * s.sx
        y_old = s.y_bot + (np.arange(field_yx.shape[0]) + 0.5) * s.sy / field_yx.shape[0]
        y_target = self._field_y[self._meep_rows] / s.S
        meep_field = self._resample_field(field_yx, x_old, y_old, np.asarray(s.x), y_target)
        energy = np.square(meep_field)
        if self.energy_ema is None or self.energy_ema.shape != energy.shape:
            self.energy_ema = energy.copy()
        else:
            alpha = 1.0 - math.exp(-FIELD_UPDATE_EVERY / FIELD_ENERGY_TAU)
            self.energy_ema = (1.0 - alpha) * self.energy_ema + alpha * energy
        self._field_intensity[self._meep_rows] = self.energy_ema * float(s.exposure_dose)
        self._field_ez[self._meep_rows] = meep_field

    # ================================================================== parameter changes
    def _set(self, **kwargs):
        s = self.s
        if "sigma_in" in kwargs:
            kwargs["sigma_in"] = min(float(kwargs["sigma_in"]), s.sigma_out - 0.01)
        if "sigma_out" in kwargs:
            kwargs["sigma_out"] = min(max(float(kwargs["sigma_out"]), s.sigma_in + 0.01), 0.95)
        if self.busy:
            return
        for key, value in kwargs.items():
            setattr(s, key, value)
        self._sync_controls()
        self._update_model_card()

        optics = {"illum", "sigma_in", "sigma_out", "na", "wavelength_nm", "domain_um", "pattern", "cd_um",
                  "pitch_ratio"}
        if optics.intersection(kwargs):
            self._t_total = s.t_end + RELAY_TAIL
            self.ax_trace.set_xlim(0, self._t_total)
            self.generate_target(quiet=True)
            return
        if "mask_choice" in kwargs:
            self._message("Mask selection changed. Run the simulation to print it.")
            return
        if "process_window" in kwargs or "ilt_iters" in kwargs:
            self.ilt_stale = self.mask_ilt is not None
            return
        if "live_all" in kwargs:
            return
        if "defocus_rel" in kwargs:
            self.ilt_stale = self.mask_ilt is not None
            if self.nearfields is not None:
                self._reproject_cached_fields()
            else:
                self._message("Defocus updated. Run ILT and the simulation to apply it.")
            return
        if "thr" in kwargs or "exposure_dose" in kwargs or "resist" in kwargs:
            if "resist" not in kwargs:
                self.ilt_stale = self.mask_ilt is not None
            if self.I_f is not None:
                self._redevelop_and_refresh_results()
                self._log(f"Resist re-developed: {self.metrics['printed']} of {self.metrics['total']} printed")
            else:
                self._status("Updated. Run the simulation to print.")
            self.page.pump()

    def _current_mask(self):
        if self.s.mask_choice == "ILT" and self.mask_ilt is not None:
            return self.mask_ilt
        return self.target

    def generate_target(self, quiet=False):
        if self.busy:
            return
        self.stop_flag = False
        s = self.s
        self.target = C.make_target(s)
        self.mask_ilt = self.mask_prev = self.mask_continuous = self.idea_mask = None
        self.projection_data = self.transmission = self.ilt_info = None
        self.ilt_stale = False
        self.I_f = self.I_thin = self.remain = None
        self.nearfields = self.clear_ref = self.fdtd_mask = None
        self.metrics = {}
        self.baseline = None
        self.emf_tau = 1.0
        self._cal_key = None
        self.energy_ema = None
        self.active_source = 0
        self._build_field_panels(self.target)
        self._reset_trace()
        self._draw_sampling()
        self._update_model_card()
        self._set_stage(0)
        self.progress.set(0)
        self.kv_status[1].set_text("-")
        self.kv_status[2].set_text("-")
        self.status_right.set_text("Idle")
        if self.system_window is not None and self.plt.fignum_exists(self.system_window.fig.number):
            self.system_window.draw(s, self.target)
        if quiet:
            self._message("Settings changed, previous mask and results cleared")
        else:
            self._message(f"Target: {s.pattern.lower()}, CD {_fmt_cd(s.cd_um)}, k1 = {s.k1:.2f}")

    # ================================================================== ILT
    def run_ilt(self):
        if self.busy:
            return
        self._set_busy(True)
        self.stop_flag = False
        s = self.s
        self.status_right.set_text("Running ILT")
        self._set_stage(1)
        try:
            target_runs = runs(self.target)
            if not target_runs:
                self._message("Target has no features; cannot run ILT")
                return
            margin = int(round(1.2 / s.dx))
            region = (max(0, target_runs[0][0] - margin), min(s.nx, target_runs[-1][1] + margin))
            min_px = max(3, int(round(0.06 / s.dx)))

            # Closed-loop EMF calibration: one FDTD exposure of the no-OPC mask gives tau and the baseline.
            key = (s.cd_um, s.pitch_um, s.pattern, s.na, s.illum, s.sigma_in, s.sigma_out, s.n_src,
                   round(s.defocus, 8), s.wavelength_nm, s.domain_um, s.res, s.bandwidth)
            if self._cal_key != key or self.baseline is None:
                self._message("FDTD calibration of the no-ILT mask")
                cal = C.calibrate_emf(s, self.target, progress=self._status, stop=self._stop_requested)
                if cal is None:
                    self._message("Stopped during EMF calibration")
                    self.status_right.set_text("Stopped")
                    return
                self.emf_tau = cal["tau"]
                base_remain = C.develop(s, cal["I_f"], dose=s.exposure_dose)
                self.baseline = dict(remain=base_remain, I_f=cal["I_f"],
                                     metrics=self._cd_metrics(base_remain, self.target))
                self._cal_key = key
                self._log(f"EMF calibration done, tau = {self.emf_tau:.2f}")
                self._log(f"No-ILT baseline: {self.baseline['metrics']['printed']} of "
                          f"{self.baseline['metrics']['total']} printed")
            self._update_model_card()

            idea_seed = make_idea_mask(self.target, bias_px=4)
            self.idea_mask = idea_seed.copy()
            x_um = np.asarray(s.x) * s.S
            self.ilt_window = ILTMonitor(self.target, x_um, s.ilt_iters, live=self.live, initial_mask=idea_seed,
                                         thr=s.thr)
            self._log(f"ILT: SGD, up to {s.ilt_iters} iterations")

            def progress(iteration, mask_now, aerial_now, loss_now):
                self.mask_prev = np.asarray(mask_now).copy()
                self.ilt_window.update(iteration, mask_now, aerial_now, loss_now)
                self.progress.set(min(iteration / max(s.ilt_iters, 1), 1.0))
                self._tick_status()
                self._status(f"SGD update {iteration}, loss {loss_now:.4g}")

            result = ilt1d.solve_ilt_1d(
                self.target, C.LAM_SIM, s.na, *s.pts, s.sx, thr=s.thr, iterations=s.ilt_iters, chunk=1,
                region=region, amp=self.emf_tau, exposure_dose=s.exposure_dose, defocus=s.defocus,
                openilt_dir=self.openilt_dir, callback=progress if self.live else None,
                project=lambda mask: mrc_clean(mask, min_px), initial_mask=idea_seed,
                process_window=bool(s.process_window))
            if result is None:
                self._message("ILT did not return a mask")
                return

            self.mask_continuous = np.asarray(result["mask"], float).copy()
            self.mask_ilt = mrc_clean(result["mask_bin"], min_px)
            self.ilt_stale = False
            self.mask_prev = None
            from_seed = 100.0 * float(np.mean(self.mask_ilt != idea_seed))
            from_target = 100.0 * float(np.mean(self.mask_ilt != self.target))
            base_loss = float(result["target_binary_loss"])
            fin = float(result["binary_loss"])
            self.ilt_info = dict(
                engine=result["engine"], iters=int(result.get("accepted_iterations", 0)),
                initial_loss=float(result["initial_loss"]), final_loss=float(result["final_loss"]),
                binary_loss=fin, seed_binary_loss=float(result["seed_binary_loss"]),
                target_binary_loss=base_loss,
                no_opc_improvement_pct=(100.0 * (base_loss - fin) / base_loss if base_loss > 1e-12 else 0.0),
                binarization_threshold=float(result["binarization_threshold"]),
                mask_changed_pct=from_seed, mask_target_delta_pct=from_target,
                dose=s.exposure_dose, thr=s.thr, defocus=s.defocus,
                process_window=bool(s.process_window), tau=self.emf_tau)
            self.ilt_window.finish(
                f"ILT complete: {from_target:.2f} % of pixels differ from the target",
                final_mask=self.mask_ilt,
                loss_comparison=[("idea seed binary", result["seed_binary_loss"], None),
                                 ("no-OPC target", result["target_binary_loss"], None),
                                 ("final ILT binary", result["binary_loss"], None)])
            self._build_field_panels(self._current_mask())
            self._reset_trace()
            self._log(f"ILT: {self.ilt_info['iters']} SGD updates ({engine_name(self.ilt_info)})")
            self._log(f"Loss {base_loss:.4f} to {fin:.4f} (binary mask)")
            self.status_right.set_text("ILT done")
            self.progress.set(1.0)
            if self.system_window is not None and self.plt.fignum_exists(self.system_window.fig.number):
                self.system_window.draw(s, self._current_mask())
        finally:
            self._set_busy(False)
            self.page.pump()

    # ================================================================== FDTD -> relay -> resist
    def run_sim(self):
        if self.busy:
            return
        s = self.s
        if s.mask_choice == "ILT" and (self.mask_ilt is None or self.ilt_stale):
            self.run_ilt()
            if self.mask_ilt is None or self.ilt_stale or self.stop_flag:
                return
        self._set_busy(True)
        self.stop_flag = False
        mask = np.asarray(self._current_mask(), float).copy()
        self._set_stage(2)
        self.status_right.set_text("Running FDTD")
        self.I_f = self.I_thin = self.remain = None
        self.metrics = {}
        self.nearfields = self.projection_data = self.transmission = None
        self.fdtd_mask = mask.copy()
        self.active_mask_label = "ILT" if (s.mask_choice == "ILT" and self.mask_ilt is not None) else "target"
        self._build_field_panels(mask)
        self._reset_trace()
        self.progress.set(0)
        try:
            self._message("Clear-field reference run")
            reference = C.clear_field_reference(s, progress=self._status, stop=self._stop_requested)
            if reference is None:
                self._message("Stopped during the clear-field run")
                self.status_right.set_text("Stopped")
                return
            sigma, _w = s.pts
            self._sigmas = list(sigma)
            fields = []
            for index, sg in enumerate(sigma):
                if self._stop_requested():
                    self._message("Stopped by user")
                    self.status_right.set_text("Stopped")
                    return
                self.active_source = index
                self._draw_sampling()
                self.kv_status[1].set_text(f"{index + 1} of {len(sigma)}")
                self.energy_right.set_text(f"source point {index + 1} of {len(sigma)}, σ = {sg:+.2f}")
                self._log(f"FDTD source {index + 1}/{len(sigma)}, σ = {sg:+.2f}")
                self._status(f"FDTD source point {index + 1} of {len(sigma)}")
                animate = index == 0 or s.live_all
                self.energy_ema = None
                self.relay = None
                self._rec_count = 0
                if animate:
                    self._reset_trace()
                    self.relay = C.RelayLive(s)
                    self.relay.set_display_rows(self._field_y[self._projection_rows] / s.S)
                field = C.run_nearfield(s, mask, float(sg), frame_cb=self._on_meep_frame if animate else None,
                                        stop=self._stop_requested, frame_period=FIELD_UPDATE_EVERY / 4.0)
                if field is None:
                    self.relay = None
                    self._message("Stopped during the FDTD run")
                    self.status_right.set_text("Stopped")
                    return
                if animate and self.relay is not None and self.relay.n > 0:
                    self._status("Pulse travelling through the relay to the resist")
                    self._play_relay_tail(float(self.relay.times[self.relay.n - 1]))
                    self._finish_trace()
                    if index == 0:
                        self._log("Pulse reached the resist")
                self.relay = None
                fields.append(field)
                self.progress.set((index + 1) / len(sigma))

            self.nearfields = fields
            self.clear_ref = float(reference)
            self.projection_data = C.fourf_plane_data(s, fields, samples_per_f=64)
            self._update_projection_strips(self.projection_data)
            self.I_f = self.projection_data["image_intensity"] / max(self.clear_ref, 1e-12)
            self.transmission = float(np.mean(self.I_f))
            self.I_thin = C.thin_mask_image(s, mask)
            self._set_stage(3)
            self.t_text.set_text("")
            self._redevelop_and_refresh_results(show=True)
            m = self.metrics
            self._log(f"Printed {m['printed']} of {m['total']}, mean |dCD| {m['cd_err_nm']:.1f} nm")
            self._log(f"Pattern mismatch {100 * m['mismatch']:.2f} %")
            self.status_right.set_text("Done")
            self._status("Print complete")
            self.progress.set(1.0)
        finally:
            self._set_busy(False)
            self.page.pump()

    def _reproject_cached_fields(self):
        if self.nearfields is None or self.fdtd_mask is None:
            return
        self._status("Recomputing the pupil filtering at the new defocus")
        reference = C.clear_field_reference(self.s, progress=self._status, stop=self._stop_requested)
        if reference is None:
            self._message("Could not recompute the clear-field reference")
            return
        self.clear_ref = float(reference)
        self.projection_data = C.fourf_plane_data(self.s, self.nearfields)
        self._update_projection_strips(self.projection_data)
        self.I_f = self.projection_data["image_intensity"] / max(self.clear_ref, 1e-12)
        self.transmission = float(np.mean(self.I_f))
        self.I_thin = C.thin_mask_image(self.s, self.fdtd_mask)
        self._redevelop_and_refresh_results(show=False)
        self._log("Defocus applied to the cached near fields. Re-run ILT to optimise at this defocus.")

    def _animate_resist_etch(self, final_remain):
        final = np.clip(np.asarray(final_remain, float), 0.0, 1.0)
        if not self.live:
            self._update_resist_overlay(final)
            return
        frames = 18
        for k in range(frames + 1):
            current = 1.0 - (k / frames) * (1.0 - final)
            self._update_resist_overlay(current)
            self._status(f"Developing resist, {100 * k / frames:.0f} %")
            time.sleep(0.02)

    def _redevelop_and_refresh_results(self, show=False):
        if self.I_f is None:
            return
        s = self.s
        self.remain = C.develop(s, self.I_f, dose=s.exposure_dose)
        self.metrics = self._cd_metrics(self.remain, self.target)
        if self.baseline is not None:
            br = C.develop(s, self.baseline["I_f"], dose=s.exposure_dose)
            self.baseline.update(remain=br, metrics=self._cd_metrics(br, self.target))
        if show:
            self._animate_resist_etch(self.remain)
        else:
            self._update_resist_overlay(self.remain)
        if self.results_window is None or not self.plt.fignum_exists(self.results_window.fig.number):
            self.results_window = ResultsWindow(live=self.live)
        cur = self.fdtd_mask if self.fdtd_mask is not None else self._current_mask()
        use_ilt = (s.mask_choice == "ILT" and self.mask_ilt is not None and not self.ilt_stale
                   and np.array_equal(cur, self.mask_ilt))
        self.results_window.update(s, self.target, cur, self.I_f, self.I_thin, self.remain, self.metrics,
                                   self.ilt_info if use_ilt else None,
                                   baseline=self.baseline if use_ilt else None)
        if self.system_window is not None and self.plt.fignum_exists(self.system_window.fig.number):
            self.system_window.draw(s, cur, self.remain)
        self._update_model_card()
        if show:
            self.results_window.show()

    def _cd_metrics(self, remain, target):
        s = self.s
        printed_runs = runs(np.asarray(remain) < 0.5)
        target_runs = runs(target)
        printed, errors, matched = 0, [], set()
        for a, b in target_runs:
            best = None
            for idx, (c, d) in enumerate(printed_runs):
                ov = min(b, d) - max(a, c)
                if ov > 0 and (best is None or ov > best[0]):
                    best = (ov, idx, c, d)
            if best is not None:
                printed += 1
                matched.add(best[1])
                errors.append(abs((best[3] - best[2]) - (b - a)) * s.dx * s.S * 1000.0)
        spurious = sum(1 for idx in range(len(printed_runs)) if idx not in matched)
        return dict(printed=printed, total=len(target_runs), spurious=spurious,
                    cd_err_nm=float(np.mean(errors)) if errors else float("nan"),
                    mismatch=C.mismatch(remain, target))

    # ================================================================== windows and files
    def open_results(self):
        if self.results_window is not None and self.plt.fignum_exists(self.results_window.fig.number):
            self.results_window.show()
        else:
            self._message("No printed result yet. Run the simulation first.")

    def open_system_diagram(self):
        s = self.s
        if self.system_window is not None and self.plt.fignum_exists(self.system_window.fig.number):
            self.system_window.draw(s, self._current_mask(), self.remain)
        else:
            self.system_window = SystemDiagram(s, self._current_mask(), live=self.live)
            if self.remain is not None:
                self.system_window.draw(s, self._current_mask(), self.remain)
        self.system_window.show()

    def save(self, path=None):
        path = path or time.strftime("litho_dashboard_%Y%m%d_%H%M%S.png")
        self.page.save(path)
        self._log(f"Saved {path}")
        return path

    def save_results(self, path=None):
        if self.results_window is None:
            raise RuntimeError("No printed result is available yet")
        path = path or time.strftime("litho_results_%Y%m%d_%H%M%S.png")
        self.results_window.save(path)
        return path


# ---------------------------------------------------------------------- CLI
def main():
    parser = argparse.ArgumentParser(description="Litho Lab: ILT -> MEEP FDTD -> 4f relay -> resist")
    parser.add_argument("--openilt", help="path to an OpenILT clone (or set OPENILT_DIR)")
    parser.add_argument("--headless", action="store_true", help="run and save PNGs without opening windows")
    parser.add_argument("--out", default="litho_lab", help="output filename prefix in headless mode")
    parser.add_argument("--cd", type=float, default=C.Setup().cd_um, help="trench CD in micrometres")
    parser.add_argument("--pitch-ratio", type=float, default=C.Setup().pitch_ratio)
    parser.add_argument("--dose", type=float, default=C.Setup().exposure_dose, help="relative exposure dose")
    parser.add_argument("--na", type=float, default=C.Setup().na)
    parser.add_argument("--resist", choices=["sigmoid", "mack"], default="sigmoid")
    parser.add_argument("--mask", choices=["ILT", "target"], default="ILT")
    args = parser.parse_args()

    if args.headless:
        import matplotlib
        matplotlib.use("Agg")

    setup = C.Setup(cd_um=args.cd, pitch_ratio=args.pitch_ratio, exposure_dose=args.dose, na=args.na,
                    resist=args.resist, mask_choice=args.mask)
    lab = LithoLab(setup, openilt_dir=args.openilt, live=not args.headless)
    if args.headless:
        if args.mask == "ILT":
            lab.run_ilt()
        lab.run_sim()
        lab.save(args.out + ".png")
        if lab.results_window is not None:
            lab.save_results(args.out + "_results.png")
    else:
        lab.plt.show(block=True)


if __name__ == "__main__":
    main()
