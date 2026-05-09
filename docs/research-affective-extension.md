# Research note: affective extension for `neurable_connector`

Date: 2026-05-09.
Scope: justify the four new affective dimensions (joy, calm, excitement,
neutral) and the two latents (valence, arousal) added to the MW75 pipeline,
under the strict constraint that the headset has **no frontal or midline
electrodes**.

This document is the spec the implementation conforms to. The score formulas
in the final section are binding.

## 1. Framing — the valence/arousal circumplex

We model affect on Russell's two-dimensional circumplex (Russell 1980): a
**valence** axis (pleasant ↔ unpleasant) crossed with an **arousal** axis
(activated ↔ deactivated). Discrete labels (joy, calm, excitement) are then
defined as regions of (valence, arousal) space, anchored to self-report
coordinates from the DEAP music-emotion dataset (Koelstra et al. 2012):

- **joy** ≈ moderate-to-high valence, moderate arousal
- **calm** ≈ moderate-to-high valence, low arousal
- **excitement** ≈ neutral-to-high valence, high arousal

`neutral` has no canonical EEG marker in the literature; we define it
operationally as a Gaussian collapse of the other intensities (low when any
other channel is loud; high when everything is quiet). This is honest about
the fact that `neutral` is a **derived** label, not a measured one.

References:
- Russell, J. A. (1980). *A circumplex model of affect.* J. Pers. Soc.
  Psychol., 39(6), 1161–1178.
- Koelstra, S. et al. (2012). *DEAP: A database for emotion analysis using
  physiological signals.* IEEE Trans. Affective Computing, 3(1), 18–31.

## 2. Valence — posterior alpha asymmetry

The most-cited EEG marker of valence is **frontal alpha asymmetry (FAA)**
(Davidson, 1992): right-hemisphere α dominance ↔ approach motivation /
positive valence. **MW75 has no frontal electrodes**, so FAA is unavailable.

The literature does support a weaker but non-zero **posterior** analogue.
Tomarken et al. (1992) reported alpha asymmetries at parietal sites
(P3/P4-equivalent) tracking trait positive vs negative affect. Papousek &
Schulter (2002, 2006) replicated parietal alpha asymmetry effects with
emotion-induction protocols, finding consistent if smaller effect sizes than
frontal sites. Reznik & Allen (2018, *Psychophysiology* review) is the
canonical recent source for the "weaker but real" caveat: across studies
they estimate within-subject r ≈ 0.15–0.30 for posterior asymmetry vs
self-reported valence — about half the magnitude reported for frontal
asymmetry, but reliably positive in sign.

We extend Davidson's approach/withdrawal hypothesis posteriorly: **right-
hemisphere posterior α dominance is associated with positive valence**.

Convention used here:
```
posterior_asymmetry = log(α[TP8] + α[P8]) - log(α[TP7] + α[P7])
                      \_______right_______/   \_______left______/
```
Higher values → more right > left α → more positive valence. The log
contrast is symmetric in the ratio and stable for small/zero powers when
implemented with `log(x + ε)`.

The four sites used (TP7, P7, TP8, P8) are the most posterior available on
MW75 and the closest analogues to the parietal sites in the cited work.

References:
- Tomarken, A. J., Davidson, R. J., Wheeler, R. E., & Doss, R. C. (1992).
  *Individual differences in anterior brain asymmetry and fundamental
  dimensions of emotion.* J. Pers. Soc. Psychol., 62(4), 676–687.
- Papousek, I., & Schulter, G. (2002). *Covariations of EEG asymmetries and
  emotional states indicate that activity at frontopolar locations is
  potentially affected by emotions.* Neuropsychobiology, 45(2), 86–93.
- Papousek, I., & Schulter, G. (2006). *Individual differences in functional
  asymmetries of the cortical hemispheres. Revival of laterality research in
  emotion and psychopathology.* Cogn. Brain Behav., 10, 269–298.
- Reznik, S. J., & Allen, J. J. B. (2018). *Frontal asymmetry as a mediator
  and moderator of emotion: An updated review.* Psychophysiology, 55(1),
  e12965.
- Davidson, R. J. (1992). *Anterior cerebral asymmetry and the nature of
  emotion.* Brain Cogn., 20(1), 125–151.

## 3. Arousal — β/α ratio at central + parietal sites

EEG arousal correlates well with **broadband β-band activity rising while
α-band falls** at central and parietal sites. The ratio `β/α` integrates
both directions and is the most common scalp-EEG arousal proxy.

Aftanas & Pavlov (2005) reported β increases and α decreases at central
sites during emotionally arousing stimuli, with within-subject Hedges'
g ≈ 0.4. Olbrich & Arns (2013, NeuroImage) review depression-related α/β
changes at central+parietal sites using an explicit β/α ratio. Reuderink,
Mühl & Poel (2013) used a comparable β/α index for affective BCI work and
report within-subject AUC ≈ 0.65–0.75 for binary arousal classification.

We use the six available central+parietal channels on MW75 — three on each
hemisphere:
```
β/α ratio = mean over {CP5, P7, C5, CP6, P8, C6} of (β_i / α_i)
```
Indices 3, 4, 5, 9, 10, 11 in our channel order. Symmetric across
hemispheres, so it is not confounded with the asymmetry signal used for
valence.

References:
- Aftanas, L. I., & Pavlov, S. V. (2005). *Trait anxiety impact on the EEG
  theta band power changes during appraisal of threatening and pleasant
  visual stimuli.* Int. J. Psychophysiol., 57(3), 213–222.
- Olbrich, S., & Arns, M. (2013). *EEG biomarkers in major depressive
  disorder: Discriminative power and prediction of treatment response.*
  Int. Rev. Psychiatry, 25(5), 604–618.
- Reuderink, B., Mühl, C., & Poel, M. (2013). *Valence, arousal and
  dominance in the EEG during game play.* Int. J. Auton. Adapt. Commun.
  Syst., 6(1), 45–62.

## 4. Discrete labels — joy / calm / excitement

DEAP self-report coordinates (Koelstra et al. 2012) anchor the discrete
labels in (valence, arousal) space:
- **joy**: positive valence, mid-to-high arousal.
- **calm**: positive valence, low arousal.
- **excitement**: neutral-to-high valence, high arousal.

We deliberately use *soft* membership rather than hard thresholds — each
label is a non-negative scalar, not a boolean. Multiple labels can be
non-zero at once; the consumer (or the differential view in `pidview`)
decides how to display dominant labels.

Non-frontal validation of the qualitative directions (positive valence
correlates with right-hemisphere α at parietal sites; high arousal with
broadband β at temporal/parietal sites) comes from:
- Lin, Y. P. et al. (2010). *EEG-based emotion recognition in music
  listening.* IEEE Trans. Biomed. Eng., 57(7), 1798–1806.
- Sammler, D. et al. (2007). *Music and emotion: Electrophysiological
  correlates of the processing of pleasant and unpleasant music.*
  Psychophysiology, 44(2), 293–304.
- Lagopoulos, J. et al. (2009). *Increased theta and alpha EEG activity
  during nondirective meditation.* J. Altern. Complement. Med., 15(11),
  1187–1192. (relaxed/calm states show parietal α increase with low β —
  consistent with our calm = positive valence × negative arousal.)

## 5. Neutral — operational definition

There is no published EEG marker for "neutral affect" because neutral is
defined by the absence of other affective signals. We encode this directly
as a Gaussian collapse:
```
neutral = exp( -(focus^2 + stress^2 + joy^2 + calm^2 + excitement^2) / 5 )
```
- Bounded in `[0, 1]`.
- Approaches 1 when every other dimension's z-score is near zero.
- Decays smoothly as any one dimension grows.
- The `/5` denominator (count of input terms) keeps the typical scale
  comparable across the input dimensions.

This is **operational, not empirical**. We do not claim a neural basis for
this number — it is a convenience for consumers who want a single
"is anything happening?" signal.

## 6. Honest expected performance

Quoting the cited literature, applied to our montage and self-reported
ground truth:
- Valence vs self-report: within-subject r ≈ 0.15–0.30 (Reznik & Allen
  2018, posterior subset only — about half the FAA magnitude).
- Arousal binary classification: within-subject AUC ≈ 0.65–0.75 (Aftanas &
  Pavlov 2005; Reuderink et al. 2013).
- Cross-subject label classification (joy/calm/excitement/neutral, 4-way):
  ≈ 65–75% (Lin et al. 2010, music-listening, 32-channel — we should
  expect to be at the lower end of this range with 12 channels and no
  frontal sites).

These are **trends, not thresholds**. The pipeline is intended to surface
within-subject changes over minutes, not point classifications.

## 7. Score formulas (binding spec)

These are the exact formulas the implementation must produce. The Python
expressions match the source in `neurable_connector/scores.py`.

```
relu(x)             = max(0, x)
gauss(x, mu, sigma) = exp(-((x - mu) / sigma)^2)

z_asym  = (posterior_asymmetry - μ_asym) / σ_asym
z_ba    = (beta_alpha_ratio    - μ_ba)   / σ_ba

valence    = z_asym
arousal    = z_ba

joy        = relu(valence) * gauss(arousal, mu=0.5, sigma=1.0)
calm       = relu(valence) * relu(-arousal)
excitement = gauss(valence, mu=0.5, sigma=1.5) * relu(arousal)
neutral    = exp( -(focus^2 + stress^2 + joy^2 + calm^2 + excitement^2) / 5 )
```

Channel index conventions, matching `neurable_connector/types.py`:
```
0:FT7  1:T7  2:TP7  3:CP5  4:P7  5:C5
6:FT8  7:T8  8:TP8  9:CP6  10:P8  11:C6
```

`posterior_asymmetry` = `log(α[8] + α[10]) - log(α[2] + α[4])` (right − left).
`beta_alpha_ratio` = mean over `{3, 4, 5, 9, 10, 11}` of `β_i / α_i`.

`focus` and `stress` are unchanged from the original spec
(`portable-stack-design.md`, "Scores" section).

## 8. Implementation notes

- All baseline statistics (μ_asym, σ_asym, μ_ba, σ_ba) are fit against the
  same eyes-open calibration window as the existing focus/stress baselines.
  The user must recalibrate to get the new latents — old baseline JSON
  files load with `μ=0`, `σ=1` defaults and emit a one-shot stderr warning.
- No new runtime dependencies. The math uses `numpy` (already present) and
  stdlib `math`.
- The connector's async generator now yields `AffectSample`. The legacy
  `FocusStressSample` symbol is preserved as a re-export alias for source
  compatibility.
