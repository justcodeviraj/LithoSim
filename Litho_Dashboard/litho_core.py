#!/usr/bin/env python3
"""
litho_core.py -- physics engine for the Litho Lab dashboard (no GUI code here)
===============================================================================

Hybrid optical model:

  1. MASK EMF             MEEP FDTD, 2-D cross-section, tilted pulsed plane waves
                          with Bloch-periodic k. Quartz + finite lossy chrome.
                          A complex DFT monitor records the field below the mask.
  2. 4f PROJECTION        The recorded complex field propagates through free space,
                          L1 quadratic phase, a hard Fourier-plane NA aperture,
                          L2 quadratic phase, and the image plane using sampled
                          scalar Fresnel propagation.
  3. PARTIAL COHERENCE    Source-point intensities are averaged incoherently.
  4. RESIST               Normalised image-plane dose drives the positive-tone
                          sigmoid or Mack development approximation.

L1/L2 are paraxial thin-lens phase screens, not physical Meep material objects.
The Fourier stop is an actual numerical aperture mask in the propagation model.
This separation keeps the rigorous FDTD region focused on the sub-wavelength
mask EMF while still computing and displaying the field throughout the 4f relay.


Units. FDTD runs in "sim units" where lambda = 0.193 (MEEP units, 1 unit = 1 um at 193 nm).
Maxwell's equations are scale-invariant, so any physical wavelength / domain size maps to
sim units by S = lambda_phys / 193 nm:  length_sim = length_phys / S.  Pixel count (and
cost) depends only on domain_width / lambda, not on the absolute scale.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field

import numpy as np

import ilt1d

LAM_SIM = 0.193                     # lambda in sim units
FCEN = 1.0 / LAM_SIM
N_QUARTZ = 1.56


# --------------------------------------------------------------------------
# Setup: physical (user-facing) parameters -> sim units
# --------------------------------------------------------------------------
@dataclass
class Setup:
    wavelength_nm: float = 193.0       # source wavelength  (frequency = c / lambda)
    domain_um: float = 10.0            # physical domain width (periodic cell)
    cd_um: float = 0.20                # physical trench width (k1 = 0.47 at NA 0.45: where ILT matters)
    pitch_ratio: float = 2.5           # pitch / CD for dense groups
    process_window: bool = True        # include ±2% dose robustness in ILT loss
    live_all: bool = False             # animate every pulse (else only the first)
    pattern: str = "Iso + dense"
    na: float = 0.45
    illum: str = "annular"
    sigma_in: float = 0.6
    sigma_out: float = 0.9
    n_src: int = 5
    thr: float = 0.30                  # resist threshold (relative to clear field)
    exposure_dose: float = 1.0          # applied exposure dose, relative to clear-field dose
    resist: str = "mack"            # sigmoid | mack
    defocus_rel: float = 0.0           # defocus in units of lambda/NA^2
    res: int = 40                      # FDTD pixels per sim-um (>= 40 for quartz)
    bandwidth: float = 0.20            # pulse fwidth / f0
    mask_choice: str = "ILT"           # ILT | target (no OPC)
    ilt_iters: int = 100
    t_mask: float = 0.10               # chrome thickness (sim um)
    air_below: float = 2.6             # air below the recording plane (sim um)

    def __post_init__(self):
        if self.wavelength_nm <= 0 or self.domain_um <= 0 or self.cd_um <= 0:
            raise ValueError("wavelength, domain width, and CD must be positive")
        if self.na <= 0 or self.res <= 0 or self.bandwidth <= 0:
            raise ValueError("NA, FDTD resolution, and source bandwidth must be positive")
        if self.exposure_dose < 0:
            raise ValueError("exposure_dose must be non-negative")
        if self.sigma_in < 0 or self.sigma_out <= 0 or self.sigma_in >= self.sigma_out:
            raise ValueError("illumination sigma must satisfy 0 <= sigma_in < sigma_out")

    @property
    def pitch_um(self): return self.pitch_ratio * self.cd_um
    # ---- unit scale ----
    @property
    def S(self): return self.wavelength_nm / 193.0
    @property
    def freq_thz(self): return 299792.458 / self.wavelength_nm          # c/lambda in THz
    @property
    def sx(self): return self.domain_um / self.S
    @property
    def nx(self): return int(round(self.sx * self.res))
    @property
    def dx(self): return self.sx / self.nx
    @property
    def x(self): return (np.arange(self.nx) + 0.5) * self.dx - self.sx / 2
    @property
    def k1(self): return (self.cd_um / self.S) * self.na / LAM_SIM
    @property
    def pts(self): return ilt1d.illum_points(self.illum, self.sigma_in, self.sigma_out, self.n_src)
    @property
    def defocus(self): return self.defocus_rel * LAM_SIM / self.na ** 2
    # ---- FDTD cell (y up; chrome bottom at y = 0) ----
    @property
    def y_src(self): return 1.2
    @property
    def y_top(self): return round((self.y_src + 0.5 + 1.0) * self.res) / self.res
    @property
    def y_rec(self): return -0.45
    @property
    def y_bot(self):
        raw = self.y_top - (self.y_rec - self.air_below - 1.0)       # round height to whole pixels
        return self.y_top - math.ceil(raw * self.res) / self.res
    @property
    def sy(self): return self.y_top - self.y_bot
    @property
    def yc(self): return 0.5 * (self.y_top + self.y_bot)
    @property
    def w_t(self): return 1.0 / (self.bandwidth * FCEN)
    @property
    def cutoff(self): return 4.0
    @property
    def t_src_end(self): return 2 * self.cutoff * self.w_t
    @property
    def t_end(self): return self.t_src_end + 6.0

    # Ideal 4f relay reference planes. These are Fourier-optics planes, not
    # additional MEEP geometry. Distances are expressed in simulation units.
    @property
    def f4(self): return 1.5
    @property
    def y_l1(self): return self.y_rec - self.f4
    @property
    def y_fourier(self): return self.y_rec - 2.0 * self.f4
    @property
    def y_l2(self): return self.y_rec - 3.0 * self.f4
    @property
    def y_image(self): return self.y_rec - 4.0 * self.f4
    @property
    def resist_thickness(self): return 0.25
    @property
    def wafer_thickness(self): return 0.60
    @property
    def y_resist_bottom(self): return self.y_image - self.resist_thickness
    @property
    def y_wafer_bottom(self): return self.y_resist_bottom - self.wafer_thickness


# --------------------------------------------------------------------------
# Target patterns (1 = exposed = etched trench; mask opening = 1)
# --------------------------------------------------------------------------
PATTERNS = ["Isolated trench", "Dense trenches", "Iso + dense"]


def make_target(s: Setup):
    x = s.x
    cd, p = s.cd_um / s.S, s.pitch_um / s.S

    def trench(xc):
        return (np.abs(x - xc) <= cd / 2).astype(float)

    t = np.zeros_like(x)
    if s.pattern == "Isolated trench":
        t += trench(0.0)
    elif s.pattern == "Dense trenches":
        for k in range(-2, 3):
            t += trench(k * p)
    else:                                                  # iso + dense
        t += trench(-0.22 * s.sx)
        for k in range(3):
            t += trench(0.15 * s.sx + k * p)
    return np.clip(t, 0, 1)


# --------------------------------------------------------------------------
# Thin-mask Abbe image (what ILT optimises) and ideal projection of FDTD near fields
# --------------------------------------------------------------------------
def thin_mask_image(s: Setup, mask):
    sig, w = s.pts
    return ilt1d.aerial_1d(mask, LAM_SIM, s.na, sig, w, s.sx, s.defocus)


def fresnel_propagate(field, distance, x, wavelength=LAM_SIM):
    """Paraxial scalar Fresnel propagation over ``distance`` (simulation µm).

    The transfer function is H(f)=exp(-i*pi*lambda*z*f^2).  The global
    carrier phase is omitted because it does not change intensity; the complex
    transverse field and its phase curvature are retained.
    """
    field = np.asarray(field, dtype=np.complex128)
    if abs(float(distance)) < 1e-15:
        return field.copy()
    dx = float(x[1] - x[0])
    fx = np.fft.fftfreq(field.size, d=dx)
    transfer = np.exp(-1j * np.pi * float(wavelength) * float(distance) * fx**2)
    return np.fft.ifft(np.fft.fft(field) * transfer)


def thin_lens_phase(field, x, focal_length, wavelength=LAM_SIM):
    """Apply a paraxial thin-lens phase screen (not a Meep material object)."""
    phase = np.exp(-1j * np.pi * np.asarray(x, dtype=float)**2 /
                   (float(wavelength) * float(focal_length)))
    return np.asarray(field, dtype=np.complex128) * phase


def fourf_plane_data(s: Setup, u_list, samples_per_f=64):
    """Propagate complex MEEP near fields through an ideal scalar 4f relay.

    The sampled plane sequence is:
        near-field monitor -> free space -> L1 phase screen -> focal/Fourier plane
        -> hard NA aperture -> free space -> L2 phase screen -> image plane.

    This is a real scalar-wave propagation calculation (Fresnel/AS paraxial
    propagation), not a painted schematic. It is not a second Meep FDTD domain:
    MEEP solves the mask EMF; the downstream lenses are thin phase elements.

    Meep's Bloch-periodic source phase is restored before propagation. The
    lower relay is displayed in wafer-side x coordinates; the 4f relay has
    unity magnification and one transverse inversion, so the x-coordinate
    convention is reflected at L2. This keeps the wafer image aligned with the
    desired target coordinate system.
    """
    sig, weights = s.pts
    if len(u_list) != len(sig):
        raise ValueError(f"Expected {len(sig)} source fields, received {len(u_list)}")
    if not u_list:
        raise ValueError("fourf_plane_data requires at least one near-field array")

    x = np.asarray(s.x, dtype=float)
    dx = float(s.dx)
    wavelength = LAM_SIM
    focal = float(s.f4)
    nx = int(s.nx)
    nseg = max(12, int(samples_per_f))
    dz = focal / nseg
    fx = np.fft.fftfreq(nx, d=dx)
    fresnel_transfer = np.exp(-1j * np.pi * wavelength * dz * fx**2)
    lens_phase = np.exp(-1j * np.pi * x**2 / (wavelength * focal))
    na_aperture = (np.abs(x) <= focal * float(s.na)).astype(float)

    # Descending physical y: monitor, L1, Fourier stop, L2, final image.
    y_map = np.linspace(float(s.y_rec), float(s.y_image), 4 * nseg + 1)
    intensity_map = np.zeros((len(y_map), nx), dtype=float)
    ez_map = np.zeros((len(y_map), nx), dtype=float)
    first_raw_fourier = None
    first_filtered_fourier = None
    first_image = None
    image_fields = []

    for source_index, (nearfield, source_sigma, source_weight) in enumerate(
            zip(u_list, sig, weights)):
        envelope = np.asarray(nearfield, dtype=np.complex128)
        if envelope.shape != (nx,):
            raise ValueError(f"Near field must have shape ({nx},), got {envelope.shape}")

        # MEEP's run_nearfield removes the Bloch phase so the returned envelope
        # can be represented on a periodic x grid. Restore the physical tilt.
        f_source = float(source_sigma) * float(s.na) / wavelength
        field = envelope * np.exp(2j * np.pi * f_source * x)
        source_rows = np.empty((len(y_map), nx), dtype=np.complex128)
        source_rows[0] = field

        row = 0
        # Object/monitor -> L1, with L1 lens phase applied at the final sample.
        for step in range(1, nseg + 1):
            field = np.fft.ifft(np.fft.fft(field) * fresnel_transfer)
            if step == nseg:
                field = field * lens_phase
            row += 1
            source_rows[row] = field

        # L1 -> Fourier plane. Record the field just before and after the NA stop.
        for step in range(1, nseg + 1):
            field = np.fft.ifft(np.fft.fft(field) * fresnel_transfer)
            if step == nseg:
                raw_fourier = field.copy()
                field = field * na_aperture
                filtered_fourier = field.copy()
            row += 1
            source_rows[row] = field

        # Fourier plane -> L2. The Fourier-plane aperture diffracts in this gap.
        for step in range(1, nseg + 1):
            field = np.fft.ifft(np.fft.fft(field) * fresnel_transfer)
            if step == nseg:
                field = field * lens_phase
                # Unity-magnification 4f imaging is inverted. Switch to the
                # wafer-side x convention at the second lens so the final image
                # is aligned with the requested wafer target coordinates.
                field = field[::-1].copy()
            row += 1
            source_rows[row] = field

        # L2 -> nominal image/resist plane.
        for step in range(1, nseg + 1):
            field = np.fft.ifft(np.fft.fft(field) * fresnel_transfer)
            row += 1
            source_rows[row] = field

        # Defocus is a further paraxial propagation at the image plane. It uses
        # the same defocus value as the OpenILT Abbe pupil model.
        if abs(float(s.defocus)) > 1e-15:
            field = fresnel_propagate(field, float(s.defocus), x, wavelength)
            source_rows[-1] = field

        weight = float(source_weight)
        intensity_map += weight * np.abs(source_rows)**2
        if source_index == 0:
            ez_map = np.real(source_rows)
            first_raw_fourier = raw_fourier
            first_filtered_fourier = filtered_fourier
            first_image = field.copy()
        image_fields.append(field.copy())

    image_intensity = intensity_map[-1].copy()
    return dict(
        x=x.copy(),
        y=y_map,
        intensity_map=intensity_map,
        ez_map=ez_map,
        x_fourier=x.copy(),
        x_image=x.copy(),
        mask_intensity=intensity_map[0].copy(),
        mask_ez=ez_map[0].copy(),
        fourier_raw_intensity=np.abs(first_raw_fourier)**2,
        fourier_filtered_intensity=np.abs(first_filtered_fourier)**2,
        fourier_raw_ez=np.real(first_raw_fourier),
        fourier_filtered_ez=np.real(first_filtered_fourier),
        image_intensity=image_intensity,
        image_ez=np.real(first_image),
        image_fields=image_fields,
        pupil_half_width=focal * float(s.na),
        aperture=na_aperture,
        lens1_y=float(s.y_l1),
        fourier_y=float(s.y_fourier),
        lens2_y=float(s.y_l2),
        image_y=float(s.y_image),
        samples_per_focal_length=nseg,
        model="Fresnel propagation + thin-lens phase screens + hard Fourier-plane NA stop",
    )


def project(s: Setup, u_list):
    """Return the final image-plane intensity from the full Fresnel 4f relay."""
    return fourf_plane_data(s, u_list)["image_intensity"]


# FDTD near field (one source point)
# --------------------------------------------------------------------------
def opaque_runs(row):
    opq = (row < 0.5).astype(int)
    d = np.diff(np.concatenate(([0], opq, [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def geometry_rectangles(s: Setup, mask):
    """Return the exact quartz/chrome rectangles used by MEEP, without importing MEEP."""
    rects = [(-s.sx / 2, 0.0, s.sx, s.y_top, "plate")]
    for i0, i1 in opaque_runs(np.asarray(mask, dtype=float)):
        x0 = i0 * s.dx - s.sx / 2
        x1 = i1 * s.dx - s.sx / 2
        rects.append((x0, 0.0, x1 - x0, s.t_mask, "chrome"))
    return rects


def build_geometry(s: Setup, mask):
    """mask: 1 = clear quartz, 0 = chrome. Returns (MEEP geometry, rectangles)."""
    import meep as mp
    chrome = mp.Medium(epsilon=1.0, D_conductivity=2 * math.pi * FCEN * 10.0)   # eps = 1 + 10i
    quartz = mp.Medium(index=N_QUARTZ)
    rects = geometry_rectangles(s, mask)
    geo = [mp.Block(size=mp.Vector3(s.sx, s.y_top), center=mp.Vector3(0, s.y_top / 2),
                    material=quartz)]
    for x0, y0, width, height, kind in rects:
        if kind == "chrome":
            geo.append(mp.Block(size=mp.Vector3(width, height),
                                center=mp.Vector3(x0 + width / 2, y0 + height / 2),
                                material=chrome))
    return geo, rects


class StopRun(Exception):
    pass


def run_nearfield(s: Setup, mask, sigma, frame_cb=None, stop=None, frame_period=None):
    """One pulsed tilted plane wave through the mask. Returns the Bloch-phase-free near field
    u(x) on s.x (complex) at the recording plane, or None if stopped."""
    import meep as mp
    f_s = sigma * s.na / LAM_SIM                      # transverse frequency (cycles/um)
    geo, _ = build_geometry(s, mask)
    pulse = mp.GaussianSource(frequency=FCEN, fwidth=s.bandwidth * FCEN, cutoff=s.cutoff)
    kw = {}
    if abs(f_s) > 1e-12:
        kw["amp_func"] = lambda p, f_s=f_s: cmath.exp(2j * math.pi * f_s * p.x)
    src = [mp.Source(pulse, component=mp.Ez, center=mp.Vector3(0, s.y_src),
                     size=mp.Vector3(s.sx, 0), **kw)]
    sim = mp.Simulation(cell_size=mp.Vector3(s.sx, s.sy), geometry_center=mp.Vector3(0, s.yc),
                        resolution=s.res, boundary_layers=[mp.PML(1.0, direction=mp.Y)],
                        geometry=geo, sources=src, k_point=mp.Vector3(f_s, 0, 0),
                        eps_averaging=False)
    dft = sim.add_dft_fields([mp.Ez], FCEN, 0, 1, center=mp.Vector3(0, s.y_rec),
                             size=mp.Vector3(s.sx, 0))
    cb = [mp.at_every(frame_period or 1.0 / FCEN / 3, frame_cb)] if frame_cb else []
    stopped = [False]

    def done(sm):
        if stop is not None and stop():
            stopped[0] = True
            return True
        return sm.meep_time() >= s.t_end

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):      # silence MEEP's progress spam
        sim.run(*cb, until=done)
    if stopped[0]:
        return None
    ez = np.asarray(sim.get_dft_array(dft, mp.Ez, 0))
    try:
        xm = np.asarray(sim.get_dft_array_metadata(dft=dft)[0], dtype=float)
    except Exception:
        xm = np.linspace(-s.sx / 2, s.sx / 2, len(ez))
    u = ez * np.exp(-2j * math.pi * f_s * xm)             # remove Bloch phase -> periodic
    x = s.x
    return np.interp(x, xm, u.real) + 1j * np.interp(x, xm, u.imag)


_REF_CACHE = {}


def clear_field_reference(s: Setup, progress=None, stop=None):
    """Clear-field intensity (uniform) from FDTD runs with an open mask, one per |sigma|."""
    sig, w = s.pts
    # The reference is projected with the current pupil, so defocus must be part
    # of the cache key.  Include the full source sampling and geometry-defining
    # settings to avoid reusing a stale clear-field reference after UI changes.
    sig_key, weights_key = s.pts
    key = (round(s.sx, 8), s.nx, s.res, s.bandwidth, s.na, s.illum,
           round(s.sigma_in, 8), round(s.sigma_out, 8), s.n_src,
           tuple(np.round(sig_key, 8)), tuple(np.round(weights_key, 8)),
           round(s.defocus, 8), round(s.f4, 8), 64, round(s.air_below, 8), round(s.y_rec, 8),
           round(s.y_src, 8), round(s.t_mask, 8))
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    ones = np.ones(s.nx)
    cache, u_list = {}, []
    for k, sj in enumerate(sig):
        kk = round(abs(float(sj)), 6)
        if kk not in cache:
            if progress:
                progress(f"clear-field reference {k + 1}/{len(sig)}")
            u = run_nearfield(s, ones, kk * np.sign(sj) if sj != 0 else 0.0, stop=stop)
            if u is None:
                return None
            cache[kk] = u
        u_list.append(cache[kk])
    I = project(s, u_list)
    ref = float(np.mean(I))
    _REF_CACHE[key] = ref
    return ref


def calibrate_emf(s: Setup, mask, progress=None, stop=None):
    """Closed-loop EMF calibration: FDTD of ``mask`` vs the thin-mask model.

    Returns dict(tau, fields, image, I_f, ref) or None if stopped.  tau is the
    effective field-amplitude transmission (least-squares fit of the FDTD image
    to tau^2 * thin-mask image).  The run is a complete no-OPC FDTD exposure, so
    its fields/image double as the "without ILT" baseline.
    """
    ref = clear_field_reference(s, progress=progress, stop=stop)
    if ref is None:
        return None
    sig, _w = s.pts
    fields = []
    for k, sj in enumerate(sig):
        if progress:
            progress(f"EMF calibration (no-ILT baseline) {k + 1}/{len(sig)}")
        u = run_nearfield(s, mask, float(sj), stop=stop)
        if u is None:
            return None
        fields.append(u)
    image = project(s, fields) / max(ref, 1e-12)
    thin = thin_mask_image(s, mask)
    tau2 = float(np.sum(image * thin) / max(np.sum(thin * thin), 1e-12))
    return dict(tau=float(np.sqrt(max(tau2, 1e-6))), fields=fields, image=image,
                I_f=image, thin=thin, ref=float(ref))


def relay_rows(s: Setup, field, nseg=32):
    """Vectorised 4f relay for ONE complex field (Bloch phase included).

    Same physics and plane order as fourf_plane_data (Fresnel -> L1 phase ->
    Fourier-plane hard NA stop -> L2 phase + inversion -> image plane [+ defocus]),
    but returns every sampled plane at once.  Returns (y, rows): y descends from
    the monitor (y_rec) to the image plane; rows has shape (4*nseg+1, nx).
    """
    x = np.asarray(s.x, dtype=float)
    nx, f = int(s.nx), float(s.f4)
    fx = np.fft.fftfreq(nx, d=float(s.dx))
    dist = (np.arange(1, nseg + 1) * f / nseg)[:, None]
    H = np.exp(-1j * np.pi * LAM_SIM * dist * fx[None, :] ** 2)
    lens = np.exp(-1j * np.pi * x ** 2 / (LAM_SIM * f))
    stop = (np.abs(x) <= f * float(s.na)).astype(float)
    rows = np.empty((4 * nseg + 1, nx), dtype=np.complex128)
    cur = np.asarray(field, dtype=np.complex128)
    rows[0] = cur
    for seg in range(4):
        seg_rows = np.fft.ifft(np.fft.fft(cur)[None, :] * H, axis=1)
        last = seg_rows[-1]
        if seg == 0:
            last = last * lens
        elif seg == 1:
            last = last * stop
        elif seg == 2:
            last = (last * lens)[::-1].copy()
        seg_rows[-1] = last
        rows[1 + seg * nseg: 1 + (seg + 1) * nseg] = seg_rows
        cur = last
    if abs(float(s.defocus)) > 1e-15:
        rows[-1] = fresnel_propagate(rows[-1], float(s.defocus), x, LAM_SIM)
    return np.linspace(float(s.y_rec), float(s.y_image), 4 * nseg + 1), rows


class RelayLive:
    """Live, time-resolved wave propagation from the near-field monitor to the resist.

    MEEP only simulates the mask region.  Below the monitor the light travels
    through the (scalar, paraxial) 4f relay.  This class records the monitor line
    during the FDTD run, builds the analytic signal Z = A e^{-i w t}, pushes the
    slowly varying envelope A through the relay, and displays plane y at time t
    as  relay[A(t - d/c)] * e^{-i w (t - d/c)}  with d = y_rec - y  (c = 1):
    a genuine delayed wavefront, not a snapshot.
    """

    def __init__(self, s: Setup, nseg=32, n_max=320):
        self.s, self.nseg = s, nseg
        self.omega = 2.0 * math.pi * FCEN
        self.y, _ = relay_rows(s, np.ones(s.nx, dtype=complex), nseg)
        self.delay = float(s.y_rec) - self.y                  # d / c per relay row
        self.R = np.zeros((n_max, len(self.y), s.nx), dtype=np.complex64)
        self.times = np.zeros(n_max)
        self.n = 0
        self._hist = []                                       # recent (t, line) for quadrature
        self.disp = None

    # -- recording ---------------------------------------------------------
    def record(self, t, line):
        """Cheap: remember the monitor line (called every T/12)."""
        self._hist.append((float(t), np.asarray(line)))
        if len(self._hist) > 8:
            self._hist.pop(0)

    def commit(self):
        """Turn the newest monitor line into an envelope sample and push it through the relay."""
        if self.n >= len(self.times) or not self._hist:
            return
        t, line = self._hist[-1]
        if np.iscomplexobj(line):                             # Bloch (sigma != 0): already analytic
            Z = line.astype(complex)
        else:                                                 # real field: quadrature from T/4 earlier
            target = t - 0.25 / FCEN
            tp, lp = min(self._hist[:-1] or [self._hist[-1]], key=lambda h: abs(h[0] - target))
            d = max(t - tp, 1e-9)
            wd = self.omega * d
            # Z = E + i*Im with Im = (cos(wd) E(t) - E(t-d)) / sin(wd)
            Z = line.astype(float) + 1j * ((math.cos(wd) * line - lp) / max(math.sin(wd), 1e-6))
        A = Z * np.exp(1j * self.omega * t)
        _, rows = relay_rows(self.s, A, self.nseg)
        self.R[self.n] = rows.astype(np.complex64)
        self.times[self.n] = t
        self.n += 1

    # -- display -------------------------------------------------------------
    def set_display_rows(self, y_disp):
        """Prepare interpolation onto the dashboard's lower-panel rows (sim units, any order)."""
        y_disp = np.asarray(y_disp, dtype=float)
        order = np.argsort(self.y)                            # ascending
        frac = np.interp(y_disp, self.y[order], np.arange(len(self.y)))
        i0 = np.clip(np.floor(frac).astype(int), 0, len(self.y) - 2)
        w = frac - i0
        self.disp = dict(y=y_disp, order=order, i0=order[i0], i1=order[i0 + 1], w=w,
                         tau=np.clip(float(self.s.y_rec) - y_disp, 0.0, None))

    def field_at(self, t):
        """Complex field on the display rows at time t, shape (len(y_disp), nx)."""
        if self.disp is None or self.n == 0:
            return None
        times = self.times[:self.n]
        dt = float(np.median(np.diff(times))) if self.n > 1 else 0.05
        want = t - self.delay                                  # envelope time needed per relay row
        idx = np.clip(np.searchsorted(times, want), 0, self.n - 1)
        prev = np.clip(idx - 1, 0, self.n - 1)
        idx = np.where(np.abs(times[prev] - want) < np.abs(times[idx] - want), prev, idx)
        live = (want >= times[0] - dt) & (want <= times[-1] + dt)
        env = self.R[idx, np.arange(len(self.y))].astype(np.complex128)
        env[~live] = 0.0
        d = self.disp
        wgt = d["w"][:, None]
        env_disp = (1.0 - wgt) * env[d["i0"]] + wgt * env[d["i1"]]
        carrier = np.exp(-1j * self.omega * (t - d["tau"]))[:, None]
        return env_disp, env_disp * carrier


# --------------------------------------------------------------------------
# Resist (positive tone: exposed dissolves). Return remaining thickness in [0, 1].
# --------------------------------------------------------------------------
def resist_sigmoid(I, thr, steepness=25.0):
    return 1.0 / (1.0 + np.exp(np.clip(steepness * (I - thr), -60, 60)))


def resist_mack(I, dose, c_abs=1.0, rmax=100.0, rmin=0.1, n=5.0, mth=0.5, t_dev=60.0, thick=100.0):
    m = np.exp(-c_abs * dose * np.asarray(I))
    a = (n + 1.0) / (n - 1.0) * (1.0 - mth) ** n
    rate = rmax * (a + 1.0) * (1.0 - m) ** n / (a + (1.0 - m) ** n) + rmin
    return np.clip(1.0 - rate * t_dev / thick, 0.0, 1.0)


def mack_dose_for(thr):
    """Mack dose that puts the 50 % remaining-thickness contour at I = thr (bisection)."""
    lo, hi = 1e-3, 1e3
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if resist_mack(thr, mid) > 0.5:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def develop(s: Setup, I_norm, dose=None):
    """Return remaining resist thickness for a normalised aerial image.

    ``dose`` is an intensity-dose multiplier (not a field-amplitude multiplier).
    If omitted, ``Setup.exposure_dose`` is used.
    """
    dose_scale = s.exposure_dose if dose is None else float(dose)
    if dose_scale < 0:
        raise ValueError("exposure dose must be non-negative")
    effective_intensity = np.asarray(I_norm, dtype=float) * dose_scale
    if s.resist == "mack":
        return resist_mack(effective_intensity, mack_dose_for(s.thr))
    return resist_sigmoid(effective_intensity, s.thr)


def mismatch(remain, target_exposed):
    """Fraction of pixels whose etched/kept state differs from the target."""
    return float(np.mean((remain < 0.5) != (target_exposed > 0.5)))
