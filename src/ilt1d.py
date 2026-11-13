#!/usr/bin/env python3
"""1D inverse-lithography adapter around OpenILT's 2D solver.

A 1D mask is embedded as four identical rows so that OpenILT's 2D FFT/kernel
implementation can be reused.  The custom Abbe kernels are generated for the
current wavelength, NA, source points, and simulation tile.

The optimisation model is deliberately kept consistent with the FDTD projection
settings: the selected defocus is used for all three dose conditions, and the
high/low conditions are intensity-dose multipliers (not field-amplitude
multipliers).  The FDTD model itself is not differentiated through; MEEP is
used for verification and for the scalar EMF transmission calibration.
"""

from __future__ import annotations

import math
import os
import sys
import types

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:  # pragma: no cover - checked by the public API
    torch = None
    nn = None
    F = None
    HAVE_TORCH = False

NY = 4
_OL_CACHE: dict[tuple, dict | None] = {}


def illum_points(mode, s_in, s_out, n):
    """Return 1D source sigma positions and normalised source weights.

    Supported modes: coherent, conventional, annular and dipole. Conventional
    illumination is the special case of the annular-source sampler with sigma_in=0.
    """
    mode = str(mode).strip().lower()
    s_in, s_out = float(s_in), float(s_out)
    if mode == "coherent":
        return np.array([0.0]), np.array([1.0])
    if mode == "dipole":
        c = 0.5 * (s_in + s_out)
        return np.array([-c, c]), np.array([0.5, 0.5])
    if mode == "conventional":
        s_in = 0.0
    elif mode != "annular":
        raise ValueError(
            f"Unknown illumination mode: {mode!r}. Choose coherent, conventional, annular or dipole."
        )
    if not (0.0 <= s_in < s_out <= 1.0):
        raise ValueError(
            f"Illumination sigma must satisfy 0 <= sigma_in < sigma_out <= 1; "
            f"got sigma_in={s_in}, sigma_out={s_out}"
        )
    n = max(int(n), 1)
    sig = ((np.arange(n) + 0.5) / n * 2.0 - 1.0) * float(s_out)
    outer = np.sqrt(np.clip(float(s_out) ** 2 - sig ** 2, 0, None))
    inner = np.sqrt(np.clip(float(s_in) ** 2 - sig ** 2, 0, None))
    weights = 2.0 * (outer - inner)
    keep = weights > 1e-9
    sig, weights = sig[keep], weights[keep]
    if sig.size == 0 or float(weights.sum()) <= 0:
        raise ValueError("Illumination settings produced no source points")
    return sig, weights / weights.sum()


def pupil(f, f_shift, lam, na, defocus=0.0):
    """Complex projection pupil at transverse frequency f + f_shift (cycles/um)."""
    lam = float(lam)
    na = float(na)
    ft = np.asarray(f) + float(f_shift)
    inside = np.abs(ft) <= na / lam + 1e-12
    kz = np.sqrt(np.clip(1.0 / lam**2 - ft**2, 0.0, None))
    phase = 2.0 * np.pi * float(defocus) * (kz - 1.0 / lam)
    return inside * np.exp(1j * phase)


def abbe_kernels(lam, na, sig, w, L, nx, defocus=0.0):
    """Build OpenILT-layout SOCS kernels: [source, 3, 2*Mk+1]."""
    lam, na, L = float(lam), float(na), float(L)
    sig = np.asarray(sig, dtype=float)
    w = np.asarray(w, dtype=np.float32)
    if sig.ndim != 1 or w.ndim != 1 or sig.size != w.size or sig.size == 0:
        raise ValueError("sig and w must be non-empty 1D arrays of equal length")
    if nx <= 0 or L <= 0 or lam <= 0 or na <= 0:
        raise ValueError("lam, na, L and nx must be positive")
    Mk = int(math.ceil(na * (1.0 + float(np.max(np.abs(sig)))) * L / lam)) + 1
    if 2 * Mk + 1 >= nx:
        raise ValueError(
            "mask grid too coarse for this NA/source extent; increase res or domain width"
        )
    m = np.arange(-Mk, Mk + 1, dtype=float)
    ker = np.zeros((len(sig), 3, 2 * Mk + 1), dtype=np.complex64)
    for j, source_sigma in enumerate(sig):
        ker[j, 1, :] = pupil(
            m / L, source_sigma * na / lam, lam, na, defocus
        ).astype(np.complex64)
    return ker, w


def load_openilt(path=None):
    """Try to import OpenILT from a clone; cache per candidate path set."""
    candidates = [path, os.environ.get("OPENILT_DIR"), "./OpenILT", "../OpenILT",
                  os.path.expanduser("~/OpenILT")]
    candidates = [os.path.abspath(os.path.expanduser(c)) for c in candidates if c]
    cache_key = tuple(candidates)
    if cache_key in _OL_CACHE:
        return _OL_CACHE[cache_key]

    for root in candidates:
        if not os.path.isdir(os.path.join(root, "pylitho")):
            continue
        old_cwd = os.getcwd()
        try:
            os.chdir(root)  # OpenILT loads its ./kernel data during import.
            if root not in sys.path:
                sys.path.insert(0, root)
            adaptive_boxes = os.path.join(root, "thirdparty", "adaptive-boxes")
            if os.path.isdir(adaptive_boxes) and adaptive_boxes not in sys.path:
                sys.path.insert(0, adaptive_boxes)
            try:
                import adabox  # used by OpenILT's optional shot-count utility
            except Exception:
                stub = types.ModuleType("adabox")
                stub.proc = stub.tools = types.SimpleNamespace()
                sys.modules["adabox"] = stub
            import pylitho.simple as ol_simple
            import pyilt.simpleilt as ol_ilt
            import pycommon.settings as ol_settings
            mods = dict(path=root, simple=ol_simple, ilt=ol_ilt, settings=ol_settings)
            _OL_CACHE[cache_key] = mods
            return mods
        except Exception as exc:  # missing optional OpenILT dependencies, e.g. cv2
            print(f"[ilt1d] OpenILT import failed ({exc!r}); using built-in solver")
        finally:
            os.chdir(old_cwd)

    _OL_CACHE[cache_key] = None
    return None


def _compute_image_fallback(cmask, kernel, scale):
    """PyTorch equivalent of pylitho.simple._computeImage for this kernel layout."""
    knx, kny = kernel.shape[-2:]
    knxh, knyh = knx // 2, kny // 2
    if cmask.dim() == 2:
        cmask = cmask.unsqueeze(0)
    spectrum = torch.fft.fft2(cmask, norm="forward")
    out = torch.zeros(
        (kernel.shape[0],) + spectrum.shape[-2:],
        dtype=spectrum.dtype, device=spectrum.device
    )
    out[:, :knxh + 1, :knyh + 1] = (
        spectrum[:, :knxh + 1, :knyh + 1] * kernel[:, -(knxh + 1):, -(knyh + 1):]
    )
    out[:, :knxh + 1, -knyh:] = (
        spectrum[:, :knxh + 1, -knyh:] * kernel[:, -(knxh + 1):, :knyh]
    )
    out[:, -knxh:, :knyh + 1] = (
        spectrum[:, -knxh:, :knyh + 1] * kernel[:, :knxh, -(knyh + 1):]
    )
    out[:, -knxh:, -knyh:] = (
        spectrum[:, -knxh:, -knyh:] * kernel[:, :knxh, :knyh]
    )
    fields = torch.fft.ifft2(out, norm="forward")
    return torch.sum(scale[:, None, None] * fields.abs().square(), dim=0)


if HAVE_TORCH:
    class LithoSim1D(nn.Module):
        """Differentiable mask-to-print model compatible with OpenILT's LithoSim API.

        ``amp`` is an effective *field-amplitude* transmission calibrated from FDTD.
        ``exposure_dose`` and the process-window multipliers are *intensity-dose*
        scales, so their field-amplitude factor is the square root of the dose.
        """

        def __init__(self, kernel, scales, thr, steepness=50.0,
                     dose_factors=(1.0, 1.02, 0.98), amp=1.0,
                     exposure_dose=1.0, compute_image=None, device="cpu"):
            super().__init__()
            self.register_buffer("kernel", torch.as_tensor(kernel, device=device))
            self.register_buffer("scales", torch.as_tensor(scales, device=device))
            self.thr = float(thr)
            self.steep = float(steepness)
            self.dose_factors = tuple(float(x) for x in dose_factors)
            self.amp = float(amp)
            self.exposure_dose = float(exposure_dose)
            self.compute_image = compute_image

        def aerial(self, mask, dose_factor=1.0):
            dose_intensity = max(self.exposure_dose * float(dose_factor), 0.0)
            field_multiplier = self.amp * math.sqrt(dose_intensity)
            complex_mask = (field_multiplier * mask).to(torch.complex64)
            if self.compute_image is None:
                return _compute_image_fallback(complex_mask, self.kernel, self.scales)
            return self.compute_image(
                complex_mask, self.kernel, self.scales, self.kernel.shape[0]
            )

        def forward(self, mask):
            nominal = self.aerial(mask, self.dose_factors[0])
            high_dose = self.aerial(mask, self.dose_factors[1])
            low_dose = self.aerial(mask, self.dose_factors[2])
            to_print = lambda intensity: torch.sigmoid(self.steep * (intensity - self.thr))
            return to_print(nominal), to_print(high_dose), to_print(low_dose)


def _builtin_solve(litho, target, params, filt, backup, iters, lr,
                   steep_mask, w_pvbl2=1.0):
    """Backtracking SGD; return parameters AFTER accepted updates.

    The previous code stored the best parameters before opt.step(). With
    chunk=1, every call therefore returned the unchanged mask. The revised
    solver uses a mean-reduced loss and a small backtracking line search.
    """
    fixed = torch.sigmoid(float(steep_mask) * backup.detach())
    p = params.detach().clone()

    def make_mask(p_value):
        return torch.sigmoid(float(steep_mask) * p_value) * filt + fixed * (1.0 - filt)

    def objective(p_value):
        mask_value = make_mask(p_value)
        nominal, high, low = litho(mask_value)
        l_nom = F.mse_loss(nominal, target, reduction="mean")
        l_pv = (F.mse_loss(high, target, reduction="mean")
                + F.mse_loss(low, target, reduction="mean"))
        return l_nom + float(w_pvbl2) * l_pv, mask_value

    current_loss, current_mask = objective(p)
    current_value = float(current_loss.detach().item())
    step_size = float(lr)

    for _ in range(int(iters)):
        p_var = p.detach().clone().requires_grad_(True)
        loss, _ = objective(p_var)
        grad = torch.autograd.grad(loss, p_var)[0] * filt
        if not torch.isfinite(grad).all() or float(torch.linalg.vector_norm(grad).item()) < 1e-12:
            break

        trial_lr = step_size
        accepted = False
        for _line_search in range(14):
            candidate = (p - trial_lr * grad).detach()
            with torch.no_grad():
                candidate_loss, candidate_mask = objective(candidate)
                candidate_value = float(candidate_loss.item())
            if candidate_value < current_value - max(1e-12, abs(current_value) * 1e-9):
                p, current_loss, current_mask = candidate, candidate_loss, candidate_mask
                current_value = candidate_value
                step_size = min(trial_lr * 1.15, 500.0)
                accepted = True
                break
            trial_lr *= 0.5
        if not accepted:
            break

    return current_value, 0.0, p.detach(), current_mask.detach()


def aerial_1d(mask_vec, lam, na, sig, w, L, defocus=0.0):
    """Thin-mask Abbe aerial image for a 1D binary or grayscale mask."""
    mask = np.asarray(mask_vec, dtype=float)
    if mask.ndim != 1 or mask.size == 0:
        raise ValueError("mask_vec must be a non-empty 1D array")
    f = np.fft.fftfreq(mask.size, d=float(L) / mask.size)
    spectrum = np.fft.fft(mask) / mask.size
    intensity = np.zeros(mask.size, dtype=float)
    for source_sigma, source_weight in zip(sig, w):
        H = pupil(f, source_sigma * na / lam, lam, na, defocus)
        field = np.fft.ifft(spectrum * H) * mask.size
        intensity += float(source_weight) * np.abs(field) ** 2
    return intensity


def solve_ilt_1d(target, lam, na, sig, w, L, thr=0.3, iterations=60, chunk=1,
                 lr=None, steep_mask=6.0, defocus=0.0, region=None, amp=1.0,
                 exposure_dose=1.0, project=None, openilt_dir=None, callback=None,
                 use_openilt=True, process_window=True, initial_mask=None,
                 steep_final=60.0):
    """Optimise a continuous idea mask, then binarise it once for MEEP.

    The differentiable Abbe/SOCS model is the ILT forward model. MEEP is used
    after optimisation to verify the final binary mask. MRC cleanup is applied
    only once, after SGD; applying it at every iteration destroys sub-pixel
    updates and can make the mask appear frozen.
    """
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch is required for ILT (install a compatible torch build)")
    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 1 or target.size == 0:
        raise ValueError("target must be a non-empty 1D array")
    if iterations < 1 or chunk < 1:
        raise ValueError("iterations and chunk must be positive")
    if exposure_dose < 0:
        raise ValueError("exposure_dose must be non-negative")
    nx, L = target.size, float(L)
    defocus = 0.0 if defocus is None else float(defocus)

    kernel, scales = abbe_kernels(lam, na, sig, w, L, nx, defocus)
    mods = load_openilt(openilt_dir) if use_openilt else None
    device = mods["settings"].DEVICE if mods else torch.device("cpu")
    compute_image = mods["simple"]._computeImage if mods else None
    # OpenILT's SimpleILT applies its configured sigmoid steepness to the mask
    # logits. Match the seed logits to that steepness to avoid saturation.
    # Mask-sigmoid steepness is ANNEALED from steep_mask (soft, gradients flow
    # everywhere) up to steep_final (nearly binary), so the continuous mask the
    # optimiser ends with is already close to the binary mask sent to MEEP.
    mask_param_steep = max(float(steep_mask), 1e-6)
    steep_final = max(float(steep_final), mask_param_steep)
    litho = LithoSim1D(
        kernel, scales, thr, steepness=50.0, amp=amp,
        exposure_dose=exposure_dose, compute_image=compute_image, device=device
    )

    i0, i1 = region if region is not None else (0, nx)
    i0, i1 = max(0, int(i0)), min(nx, int(i1))
    if i1 <= i0:
        raise ValueError("ILT region must contain at least one column")

    target_2d = torch.as_tensor(np.tile(target, (NY, 1)), device=device)
    # Start from a softly blurred idea mask. Its edge pixels sit near 0.5,
    # allowing SGD to move trench boundaries instead of only changing values
    # deep inside already-open/closed regions.
    seed = target.copy() if initial_mask is None else np.asarray(initial_mask, dtype=np.float32)
    if seed.shape != target.shape:
        raise ValueError("initial_mask must have the same shape as target")
    freq0 = np.fft.fftfreq(nx, d=1.0)
    blur_sigma_px = 1.25
    blur_kernel = np.exp(-2.0 * (np.pi * blur_sigma_px * freq0) ** 2)
    soft_1d = np.fft.ifft(np.fft.fft(seed) * blur_kernel).real
    soft_1d = np.clip(0.06 + 0.88 * soft_1d, 0.03, 0.97)
    initial_probability = torch.as_tensor(
        np.tile(soft_1d, (NY, 1)), dtype=torch.float32, device=device
    )
    params = (torch.log(initial_probability / (1.0 - initial_probability))
              / mask_param_steep).detach()
    backup = params.detach().clone()
    filt = torch.zeros_like(target_2d)
    filt[:, i0:i1] = 1.0

    def objective(mask_value):
        nominal, high, low = litho(mask_value)
        l_nom = F.mse_loss(nominal, target_2d, reduction="mean")
        if process_window:
            l_pv = (F.mse_loss(high, target_2d, reduction="mean")
                    + F.mse_loss(low, target_2d, reduction="mean"))
        else:
            l_pv = torch.zeros((), device=device)
        return l_nom + l_pv

    def mask_of(p_value, steep):
        return (torch.sigmoid(steep * p_value) * filt
                + torch.sigmoid(steep * backup) * (1.0 - filt))

    current_mask = mask_of(params, mask_param_steep)
    with torch.no_grad():
        initial_loss = float(objective(current_mask).item())
        seed_tensor = torch.as_tensor(np.tile(seed, (NY, 1)), device=device)
        seed_binary_loss = float(objective(seed_tensor).item())
        target_tensor = torch.as_tensor(np.tile(target, (NY, 1)), device=device)
        target_binary_loss = float(objective(target_tensor).item())
    history = [initial_loss]
    engine = "built-in SGD (Abbe model)"
    solver = None
    n_elem = float(target_2d.numel())

    # OpenILT's SimpleILT is the optimiser when a clone is installed.
    # NOTE: SimpleILT.solve() stores bestParams/bestMask BEFORE opt.step(), so a
    # call with Iterations=1 returns its input unchanged (the "frozen mask" bug).
    # Iterations=2 returns the parameters after exactly ONE real SGD step (or
    # the input again if that step increased the loss).
    if mods:
        cfg = mods["ilt"].SimpleCfg(dict(
            Iterations=2, TargetDensity=float(thr), SigmoidSteepness=mask_param_steep,
            WeightEPE=0.0, WeightPVBL2=1.0 if process_window else 0.0,
            WeightPVBand=0.0, StepSize=1.0,
            TileSizeX=NY, TileSizeY=nx, OffsetX=0, OffsetY=i0,
            ILTSizeX=NY, ILTSizeY=i1 - i0
        ))
        solver = mods["ilt"].SimpleILT(cfg, litho, device=device)
        engine = "OpenILT SimpleILT.solve (Iterations=2 -> one real step)"

    lr_cur = float(lr) if lr is not None else 2000.0
    lr_max, lr_min = 1e5, 1e-4
    previous_loss = initial_loss
    done = 0
    steep_now = mask_param_steep
    n_iter = int(iterations)

    for it in range(n_iter):
        # ---- annealed mask steepness (geometric schedule) ----
        frac = it / max(n_iter - 1, 1)
        steep_it = mask_param_steep * (steep_final / mask_param_steep) ** frac
        if abs(steep_it - steep_now) > 1e-9:
            steep_now = steep_it
            with torch.no_grad():       # same params, sharper mask -> new reference loss
                previous_loss = float(objective(mask_of(params, steep_now)).item())

        accepted = False
        trial_lr = lr_cur
        for _try in range(16):
            if solver is not None:
                solver._config._config["SigmoidSteepness"] = steep_now
                # OpenILT sums the MSE; our objective is a mean -> divide the step.
                solver._config._config["StepSize"] = trial_lr / n_elem
                _l2, _pv, p_new, m_new = solver.solve(target_2d, params)
                cand_params, cand_mask = p_new.detach(), m_new.detach()
            else:
                _l, _pv, cand_params, cand_mask = _builtin_solve(
                    litho, target_2d, params, filt, backup, 1, trial_lr,
                    steep_now, w_pvbl2=1.0 if process_window else 0.0)
            with torch.no_grad():
                cand_loss = float(objective(cand_mask).item())
            if cand_loss < previous_loss - max(1e-12, abs(previous_loss) * 1e-9):
                accepted = True
                break
            trial_lr *= 0.5
            if trial_lr < lr_min:
                break
        if not accepted:
            break                                   # converged: no descent step left
        params, current_mask, previous_loss = cand_params, cand_mask, cand_loss
        # grow the step after a first-try success, keep the shrunk one otherwise
        lr_cur = min(trial_lr * (1.5 if trial_lr >= lr_cur else 1.0), lr_max)

        done += 1
        history.append(previous_loss)
        with torch.no_grad():
            aerial = litho.aerial(current_mask).detach().cpu().numpy()[0]
        if callback is not None:
            callback(done, current_mask[0].detach().cpu().numpy(), aerial, previous_loss)

    continuous_mask = current_mask[0].detach().cpu().numpy()
    # Convert the continuous idea mask to a manufacturable binary mask once.
    # Search a narrow threshold range: the best level-set threshold can differ
    # slightly from 0.5 because the resist threshold and mask sigmoid are not
    # symmetric. Every candidate is scored with the same process-window loss.
    best_binary, best_binary_loss, best_threshold = None, float("inf"), 0.5
    for threshold in np.linspace(0.35, 0.65, 13):
        candidate = (continuous_mask >= float(threshold)).astype(float)
        if project is not None:
            candidate = np.asarray(project(candidate), dtype=np.float32)
            if candidate.shape != (nx,):
                raise ValueError("project(mask) must return a 1D array with the same length")
        candidate_tensor = torch.as_tensor(np.tile(candidate, (NY, 1)), device=device)
        with torch.no_grad():
            candidate_loss = float(objective(candidate_tensor).item())
        # Prefer the lower objective; near-ties prefer a threshold closer to 0.5.
        if (candidate_loss < best_binary_loss - 1e-10 or
            (abs(candidate_loss-best_binary_loss) <= 1e-10 and
             abs(float(threshold)-0.5) < abs(best_threshold-0.5))):
            best_binary = candidate
            best_binary_loss = candidate_loss
            best_threshold = float(threshold)

    mask_bin = best_binary
    binary_tensor = torch.as_tensor(np.tile(mask_bin, (NY, 1)), device=device)
    with torch.no_grad():
        final_cont_loss = float(objective(current_mask).item())
        aerial = litho.aerial(binary_tensor).detach().cpu().numpy()[0]

    return dict(
        mask=continuous_mask,
        mask_bin=mask_bin,
        aerial=aerial,
        loss=history,
        initial_loss=initial_loss,
        seed_binary_loss=seed_binary_loss,
        target_binary_loss=target_binary_loss,
        final_loss=final_cont_loss,
        binary_loss=best_binary_loss,
        binarization_threshold=best_threshold,
        accepted_iterations=done,
        engine=engine,
        lr=lr_cur,
        defocus=defocus,
    )

