# Litho Lab

Litho Lab is an interactive computational-lithography simulator. It takes a one-dimensional line pattern, optimises a photomask for it with inverse lithography (ILT), solves Maxwell's equations for the real quartz and chrome mask with MEEP, carries the resulting light through a projection lens, develops a photoresist and measures how closely the printed pattern matches the design. Its purpose is to show, on one screen and with real physics, what ILT buys you over printing the design directly.

The simulator is a proof of concept for teaching and discussion. It is two-dimensional (a 1D cutline through the mask), scalar in the lens and does not model a production resist stack. Section 14 lists what it leaves out.

## 1. Quick start

Litho Lab needs Python 3.10 or newer, NumPy, Matplotlib, PyTorch (CPU build is enough) and MEEP. MEEP is easiest to install from conda-forge:

```
conda create -n lith -c conda-forge python=3.11 pymeep matplotlib pytorch-cpu
conda activate lith
git clone https://github.com/OpenOPC/OpenILT.git ~/OpenILT      # optional, see below
python litho_lab.py
```

OpenILT is optional. When it can be found (the `--openilt` flag, the `OPENILT_DIR` environment variable, or a folder named `OpenILT` next to the code or in your home directory) its `SimpleILT` solver performs the SGD steps. Without it, or if it fails to descend, Litho Lab falls back to a built-in PyTorch implementation of the same objective and says so in the message log.

```
python litho_lab.py                              # interactive
python litho_lab.py --openilt ~/OpenILT          # point at an OpenILT clone
python litho_lab.py --headless --out run1        # no windows; writes run1.png and run1_results.png
python litho_lab.py --headless --resist mack --cd 0.18 --na 0.5 --dose 1.1
```

A normal session is: choose settings, press **Run ILT** (this also runs the calibration described in section 5), press **Run simulation**, then read the Results window. Pressing **Run simulation** with the ILT mask selected runs ILT first if it has not been run, or if settings changed since.

## 2. What happens when you press the buttons

The pipeline has four stages and the header of the main window shows which one is active.

**Target.** The wafer pattern you want is generated from the pattern, CD, pitch and domain settings. It is a binary line along x in which 1 means "this part of the resist must be removed" (a trench). It is also the starting point for the mask.

**ILT.** A mask is optimised so that its predicted image, after the resist threshold, reproduces the target. The prediction uses a fast Abbe imaging model that is differentiable, so gradient descent can be used. A first FDTD run on the unmodified mask is made beforehand to calibrate that fast model (section 5).

**FDTD.** MEEP solves Maxwell's equations for the actual mask, once per source point, and records the complex electric field just below the mask (the near field). The near field then travels through a numerical projection lens (section 6). With the live display on, you watch the light pulse leave the mask, pass the lens and arrive at the resist.

**Print.** The intensity at the image plane is normalised by the clear-field intensity, scaled by the exposure dose and turned into remaining resist thickness by the resist model. Metrics are computed against the target, and the Results window opens.

Changing the resist model, threshold or dose after a print does not repeat the FDTD. The cached image is simply developed again, so you can compare the Sigmoid and Mack models instantly. Changing anything optical (source, wavelength, NA, pattern, CD, pitch, domain) discards the mask and the results, because they no longer apply.

## 3. Units, scale and the quantity that matters: k1

All physics runs in "simulation units" in which the wavelength is 0.193. A physical length is converted with the scale `S = wavelength / 193 nm`: `length_sim = length_physical / S`. Maxwell's equations are scale invariant, so a run at 193 nm with a 160 nm trench is the same calculation as a run at 248 nm with a 205 nm trench. The grid size, and therefore the run time, depends only on the ratio of the domain width to the wavelength.

The number that decides whether optical proximity effects are strong is

```
k1 = CD * NA / wavelength
```

Above about 0.5 a binary mask prints the target and ILT has little to improve. Below it the image of a plain mask rounds off, shrinks or fails to print, and a corrected mask can recover it. The default settings (193 nm, NA 0.45, CD 160 nm) give k1 = 0.37, which is where the difference between "no ILT" and "ILT" is easiest to see. The mask is stored on a pixel grid of 25 nm at the default resolution, so critical dimensions change in steps of that size. This is why the reported ΔCD moves in 25 nm steps.

## 4. Illumination and partial coherence

A real scanner does not illuminate the mask with a single plane wave. Light arrives from a range of angles described by the source shape, whose radius is measured in σ: σ = 1 is the largest angle the lens can accept, so a ray at σ = 0.8 enters at 0.8 times the lens NA. Litho Lab supports four shapes.

| Source | Meaning |
|---|---|
| Coherent | One on-axis plane wave. Sharpest edges, strongest ringing. |
| Conventional | A filled disc from σ = 0 to the outer σ. |
| Annular | A ring between the inner and outer σ. Improves dense lines at the cost of isolated ones. |
| Dipole | Two poles centred at the mean of inner and outer σ. Strongly favours one line orientation. |

Partial coherence is computed with the Abbe method: the source is sampled at a handful of points along a one-dimensional cutline (five by default), each point is simulated as its own tilted plane wave, and the resulting intensities are added with weights equal to the area of the source each point represents. Intensities are added, not fields, because different source points are mutually incoherent. The "Source sampling" chart shows the points and weights; the point being simulated is highlighted while the FDTD runs. For the conventional source with outer σ = 0.8 the five points sit at σ = −0.64, −0.32, 0, 0.32, 0.64 with weights 0.149, 0.227, 0.248, 0.227, 0.149.

## 5. The mask near field, and what "EMF" correction means

### 5.1 Why the mask needs Maxwell's equations

The usual shortcut treats a mask as a perfectly thin screen: each point is either fully transparent or fully opaque, and the light just behind the mask is the incident light multiplied by 1 or 0. This is the thin-mask or Kirchhoff approximation, and it is what a fast imaging model such as the one used in ILT assumes.

At 193 nm the openings are as small as the wavelength and the chrome is about 100 nm thick, so the shortcut is inaccurate. Light diffracts around chrome edges, narrow openings behave a little like waveguides that pass less light, and the field has phase and amplitude variations close to the edges that a binary screen cannot represent. These are called electromagnetic field (EMF) effects, or mask topography effects. To capture them, Litho Lab solves Maxwell's equations on a cross-section of the real mask with the finite-difference time-domain method in MEEP.

### 5.2 The FDTD calculation

The cell is two-dimensional. A quartz plate (refractive index 1.56) has chrome blocks, modelled as a lossy medium with permittivity 1 + 10i and 100 nm thickness, wherever the mask is opaque. The field is polarised along the line direction (Ez, the "TE" case). The x direction is periodic over the domain width, and absorbing layers close the cell above and below. A Gaussian light pulse with 20 % fractional bandwidth is launched from a line source above the mask. For an off-axis source point the pulse is a tilted plane wave, produced with a phase ramp on the source and a matching Bloch wave vector so that the tilted wave is periodic in x. A discrete Fourier transform monitor, placed 0.45 µm below the mask, accumulates the complex field at the centre frequency. This is the near field.

Because the quantity of interest is the ratio to a bare wafer, one additional FDTD run per source point is made with no chrome at all (the clear-field reference). All printed intensities are divided by it, which is why an open area prints at relative intensity 1.

### 5.3 Calibration: closing the loop between the fast model and FDTD

ILT needs a model it can differentiate and evaluate hundreds of times, so it cannot call FDTD inside its loop. It uses the thin-mask Abbe model. If that model were used uncorrected, ILT would optimise a mask for an idealised screen and the real mask would print differently.

Litho Lab therefore performs an EMF calibration before ILT. It runs the full FDTD pipeline once on the uncorrected target mask, and compares the resulting image `I_fdtd` with the thin-mask prediction `I_thin` for the same mask. It fits one number τ by least squares so that `I_fdtd ≈ τ² · I_thin`:

```
τ² = Σ ( I_fdtd · I_thin ) / Σ ( I_thin² )
```

τ is the effective amplitude transmission of the real mask relative to the ideal one. For 160 to 200 nm trenches at 193 nm it comes out around 0.85 to 0.9: the real mask passes about 25 % less intensity than the thin-mask model assumes. ILT then multiplies the fields in its forward model by τ, so the mask it designs is optimised for the light MEEP will actually deliver. Without this step the optimiser would, for example, choose trenches that are too narrow because it believes more light gets through.

The calibration run is not wasted: it is a complete FDTD exposure of the mask without ILT, so it is also the baseline that the Results window shows next to the ILT result. It is cached and reused until a setting that affects it changes. The value is shown as "Mask EMF transmission τ" in the Model panel.

A single number cannot reproduce everything FDTD does. Edge-dependent phase and polarisation effects are not captured by τ. That is why the final judgement is always made on the FDTD image itself and never on the ILT prediction.

## 6. The projection lens

Below the near-field monitor, light is carried to the wafer by a scalar, paraxial 4f relay. It is a numerical model, not additional MEEP geometry. Starting from the complex near field, the code applies, in order: free-space propagation over one focal length `f` (Fresnel transfer function), the phase screen of lens L1, propagation to the Fourier plane, a hard aperture, propagation to lens L2, the phase screen of L2, and propagation to the image plane. Free-space steps use the transfer function `exp(−iπλz·fx²)` applied with FFTs; each lens multiplies by `exp(−iπx²/(λf))`.

The Fourier plane is where the diffraction orders of the mask separate, with order `m` landing at a position proportional to its angle. The aperture there is opaque outside ±f·NA, which is exactly what the numerical aperture of a lens does: only orders whose angle lies inside the NA reach the wafer. A smaller NA discards higher orders and blurs fine features. Unity magnification and the inversion of a 4f system are accounted for so that the image lines up with the target. Defocus is a further propagation of the image field by a distance `defocus_rel · λ / NA²`.

The tilt of each source point is restored before propagation (MEEP returns the field with its Bloch phase removed), so oblique illumination lands at the correct place in the pupil. The final image is the weighted sum of the source-point intensities.

### The light pulse

Every source point launches the same pulse; the source shape changes only the direction of the wave, not its duration. The pulse is Gaussian in time, centred on the chosen wavelength, with a fractional bandwidth of 20 % (`bandwidth` in `Setup`). Its temporal width parameter is `w = 1 / (bandwidth · f0)`, which is 0.965 time units at 193 nm. The full width at half maximum is 2.355 w, about 2.3 time units. Since one time unit is 1 µm of light travel (3.34 fs at 193 nm, scaled by the wavelength factor S), this is roughly 7.6 fs at 193 nm; the Model panel shows the value for the current wavelength. The source is switched on for 2 × 4 w = 7.7 time units and the run ends 6 time units later, at 13.7. The pupil position σ tilts the wave, so a tilted pulse reaches the near-field monitor a few hundredths of a time unit later than an on-axis one, which is invisible on the chart.

### Live display

The Re(Ez) and intensity panels show the FDTD region above the near-field monitor and the relay below it. The relay region is not a second FDTD simulation. While MEEP runs, the field on the monitor line is sampled several times per optical cycle, converted into a slowly varying complex envelope, propagated through the relay once per sample, and displayed with the delay that light needs to reach each plane. The result is a pulse that visibly crosses L1, is clipped at the NA stop and refocuses at the image plane, at the right moment in time. After the FDTD ends, the last few time units are played out so the pulse reaches the resist.

The Optical intensity chart records the mean intensity on three planes against time: the near-field monitor, the Fourier plane after the stop and the image plane. Its curves show the delay between planes and how much energy the stop removes. By default only the first source point is animated; the other four run at full speed. Tick "Simulate all source points" to animate every pulse, which makes the run noticeably longer.

## 7. Inverse lithography (ILT)

ILT searches for the mask whose predicted print equals the target. Litho Lab builds it on OpenILT's `SimpleILT`, a gradient-descent mask optimiser with a differentiable imaging model, by embedding the one-dimensional cutline as four identical rows of a 2D mask. The Abbe kernels for the current wavelength, NA, source points and defocus are generated exactly, so the imaging model is the same optical system the rest of the code uses.

The mask is represented by a continuous variable `p` per pixel and the transmission is `sigmoid(k·p)`. The loss is the mean squared difference between the predicted print and the target, evaluated at the nominal dose and, when the process-window option is on, also at +2 % and −2 % dose (intensity multipliers), so the mask has to print correctly across a small dose range rather than at one dose only. The predicted print is `sigmoid(50·(I − threshold))`, a smooth version of the resist threshold. Only the region around the features is free to change; the rest of the mask is held fixed. Because the loss is a mean over pixels, its absolute value is small.

The starting mask is the target shrunk inward by up to four pixels per edge, the "idea mask". It is deliberately not the target itself so that visible work is left for the optimiser. Each iteration takes one real `SimpleILT` step with a line search on the step size: a step is accepted only if the loss decreases, the step size grows after a successful first trial and shrinks otherwise, and at most sixteen trials are made. Mask steepness `k` is annealed geometrically from 6 (soft, gradients flow everywhere) to 60 (nearly binary) over the iterations, so the continuous mask ends close to a binary one. If no step descends the run stops early. Typically most of the improvement comes in the first ten to twenty updates and the later ones refine edges.

At the end the continuous mask is binarised once. A threshold between 0.35 and 0.65 is scanned and the one giving the lowest loss is kept, then a mask-rule check removes openings and chrome segments narrower than three pixels (about 75 nm at the default resolution). The binary array that results is exactly what MEEP receives. The ILT monitor shows the current mask (the target used directly) on top and, below it, the ILT mask redrawn in place at every accepted update, with orange ticks wherever it differs from the current mask and a curve of its transmission. At the bottom are the objective against iteration, with the "no ILT" and initial-mask levels marked at the end, and a short summary.

## 8. Resist models

A positive-tone resist dissolves where it received enough light. The remaining thickness is a number between 1 (fully kept) and 0 (fully removed). A trench is where less than half the thickness remains.

**Sigmoid model.** Remaining thickness is `1 / (1 + exp(25·(I·dose − threshold)))`. It is a smooth threshold: the resist is removed where the dose-scaled intensity exceeds the threshold. The threshold is relative to the clear-field intensity.

**Mack model.** A physical development-rate model. Exposure destroys the inhibitor, whose remaining fraction is `m = exp(−C·dose·I)`. The development rate follows Mack's expression `r = rmax·(a+1)(1−m)ⁿ / (a+(1−m)ⁿ) + rmin` with `n = 5`, `rmax = 100`, `rmin = 0.1` and `m_th = 0.5`, and the thickness lost is the rate times a fixed development time. The exposure constant `C` is calibrated automatically so that the 50 % thickness contour falls at the same intensity as the threshold you chose, which keeps the two models comparable. What differs is the shape of the transition: the Mack model has a finite contrast and a slower roll-off, so shallow image slopes print with sloped sidewalls instead of the abrupt edge of the sigmoid.

Neither model includes acid diffusion, post-exposure bake, resist absorption through the depth, or stochastic effects. ILT always uses its own sigmoid print model internally, independently of the selection here; the selection affects the simulated print and the metrics.

## 9. Metrics

The Results window compares the print with the target and, when the ILT mask was used, with the baseline print of the unmodified mask. Each print is drawn as a cross-section: the mask on top, and the yellow resist on silicon below it, with the resist that was etched away shown pale. Dashed guides link each target trench to the resist, numbered as in the per-trench table.

| Metric | Definition |
|---|---|
| Trenches printed | Number of target trenches that have a printed trench overlapping them. |
| ΔCD | For each printed trench, printed width minus target width, measured on the mask pixel grid. The mean absolute value over trenches is shown in the summary. |
| Pattern mismatch | Fraction of pixels on the line whose printed state (removed or kept) differs from the target. |
| Spurious features | Printed trenches that overlap no target trench. |

## 10. Reference: every parameter

Parameters are grouped as in the sidebar. The number next to any slider can be clicked and typed over: Enter accepts the value, Escape cancels, and values outside the slider range are clamped. For the critical dimension a plain number is read as nanometres and a `um` suffix as micrometres. Everything except the ones marked "fixed" is changed from the interface; fixed values are fields of `Setup` in `litho_core.py` and can be edited there.

### Illumination

| Parameter | Range | Effect |
|---|---|---|
| Source type | coherent, conventional, annular, dipole | Shape of the illumination in the pupil (section 4). |
| Wavelength λ | 150 to 2000 nm | Sets the scale S and k1. Smaller wavelengths print smaller features. |
| Outer σ | 0.1 to 0.95 | Outer radius of the source. Larger σ is more incoherent: less ringing, lower contrast of fine pitches. Unused for the coherent source. |
| Inner σ | 0 to 0.9 | Inner radius of an annular source or the inner edge of a dipole. Disabled for the other types. |

### Projection lens

| Parameter | Range | Effect |
|---|---|---|
| Numerical aperture NA | 0.2 to 0.9 | Half-width of the Fourier-plane stop is `f·NA`. Higher NA passes more diffraction orders, giving finer resolution (smaller `λ/(2NA)` pitch limit) and higher k1. |
| Defocus | 0 to 1 in units of λ/NA² | Distance by which the wafer is out of focus. Blurs the image and shrinks the process window. Changing it re-filters the cached near fields without a new FDTD run, but the ILT mask must be re-run to be optimised for it. |

### Pattern

| Parameter | Range | Effect |
|---|---|---|
| Pattern | isolated, dense, iso + dense | Isolated: one trench at the centre. Dense: five trenches at the chosen pitch. Iso + dense: one isolated trench on the left and a group of three dense trenches on the right, so one run shows both behaviours. |
| Critical dimension CD | 50 nm to 20 µm | Width of each trench. With λ and NA it sets k1. |
| Pitch / CD | 2.0 to 5.0 | Centre-to-centre distance of dense trenches in units of CD. Smaller means denser and harder. |
| Domain width | 4 to 120 µm | Width of the periodic cell. Larger domains cost more FDTD time but separate the pattern from its periodic neighbours. |

### Exposure and resist

| Parameter | Range | Effect |
|---|---|---|
| Resist model | Sigmoid, Mack | Development model (section 8). |
| Resist threshold | 0.1 to 0.6 | Relative intensity at which the resist's remaining thickness is 50 %. Lower means more sensitive resist and wider trenches. |
| Exposure dose | 0.1 to 3.0 × | Multiplies the intensity before development. At fixed threshold, more dose widens trenches. It is a quick way to see sensitivity of the print to dose. |

### Mask and ILT

| Parameter | Range | Effect |
|---|---|---|
| Mask | ILT mask, Target (no OPC) | Which mask the simulation prints. "Target" prints the design directly. |
| SGD iterations | 10 to 300 | Upper limit on accepted ILT updates. The optimiser can stop earlier when no step lowers the loss. |
| Process-window terms | on, off | Add the ±2 % dose terms to the ILT loss so that the mask is robust to dose variation, not only correct at nominal dose. |
| Simulate all source points | on, off | Animate every pulse in the live display rather than only the first. Does not change the physics. |

### Fixed simulation parameters

| Parameter | Default | Meaning |
|---|---|---|
| `n_src` | 5 | Number of Abbe source points. |
| `res` | 40 | FDTD pixels per simulation µm; also the mask pixel size, 25 nm at 193 nm. |
| `bandwidth` | 0.20 | Fractional bandwidth of the source pulse. |
| `t_mask` | 0.10 | Chrome thickness in simulation µm (100 nm at 193 nm). |
| `air_below` | 2.6 | Air between the monitor and the lower PML. |
| `f4` | 1.5 | Focal length of each lens in the 4f relay (simulation µm). The lens separation of 3 f and the total path of 6 µm follow from it. |
| Resist thickness | 0.25 | Used by the display of the photoresist. |

## 11. Reading the main window

The left column holds all controls. The field card shows two panels on shared axes: time-averaged intensity and the instantaneous electric field Re(Ez), both scaled to a running display peak. Horizontal dashed lines mark the source plane, mask, near-field monitor, L1, Fourier plane with its opaque stop, L2 and image plane; the hatched band at the top is the absorbing layer, and below the image plane the orange layer is the resist and the grey layer the silicon wafer. The orange layer takes on the developed profile when printing finishes. The cursor probe shows position and local values under the mouse. The right column reports progress, a summary of the optical model including τ, and a timestamped message log. The **System diagram** button opens a conceptual picture of the whole machine in a separate window, and **Results** reopens the print results.

## 12. Choosing settings where ILT matters

ILT can only help where a plain mask fails. Aim for k1 between 0.3 and 0.45: for 193 nm and NA 0.45 that means a CD of about 130 to 190 nm. A conventional source with outer σ near 0.8 and the "Iso + dense" pattern shows the difference clearly: without ILT, the dense group typically fails to print or prints with the wrong width, while the ILT mask prints all trenches. Towards k1 = 0.25, the physical limit for dense lines, no mask can print the pattern, and above 0.5 there is nothing to correct. Because ΔCD moves in steps of one mask pixel, differences smaller than 25 nm (at the default resolution) cannot be resolved; raise `res` in `Setup` for finer steps at the cost of run time.

## 13. Code layout

| File | Contents |
|---|---|
| `litho_core.py` | Physics, no GUI: setup and unit scaling, target patterns, MEEP geometry and runs, EMF calibration, 4f relay, live relay, resist models, metrics. |
| `ilt1d.py` | One-dimensional ILT on top of OpenILT: Abbe kernels, SGD with line search and annealing, binarisation, fallback solver. |
| `litho_lab.py` | Main window and the run logic. |
| `litho_windows.py` | ILT monitor, Results window and System diagram. |
| `litho_ui.py` | Small widget toolkit (sliders, segmented controls, buttons) built on Matplotlib. |
| `litho_util.py` | Mask helpers: run detection, mask-rule cleanup, idea-mask seeding. |

## 14. Limitations

The model is two-dimensional: a one-dimensional line pattern is simulated, so corners, line ends and 2D proximity effects are absent. The incident field is polarised along the lines and the lens is scalar and paraxial, so vector effects of high NA are not included. The lenses are ideal thin phase screens with no aberrations, and the Fourier-plane stop is a hard edge. The ILT forward model is the thin-mask model with a single calibrated transmission τ, and FDTD is not differentiated through. The resist models act on the intensity at the top surface, with no depth dependence, diffusion, bake or noise. Source shapes are sampled along a one-dimensional cutline with a small number of points. These choices keep the whole chain interactive and transparent; they should be relaxed before drawing quantitative conclusions about a real process.

## 15. Troubleshooting

If the ILT report shows zero SGD updates, make sure all files from the same release are in use, since an old `ilt1d.py` freezes the mask. If OpenILT cannot be imported the log says "built-in SGD"; the results are equivalent. If MEEP is slow, reduce the domain width: FDTD cost scales with the domain-to-wavelength ratio. If the window is larger than your screen, resize it: the layout scales with the window.
