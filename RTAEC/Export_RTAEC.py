import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from STFT_Process import STFT_Process

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "audio_onnx_metadata.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from audio_onnx_metadata import build_audio_metadata_from_globals, build_model_metadata
from aec_delay_align import DynamicDelayAlign, _crossfade_profile, _segment_bounds


# User settings.
onnx_model_A           = str(Path(__file__).resolve().parent / "RTAEC_ONNX" / "RTAEC.onnx") # The exported onnx model path.
IN_SAMPLE_RATE        = 16000                         # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
OUT_SAMPLE_RATE       = 16000                         # [8000, 16000, 22500, 24000, 44000, 48000]; It accepts various sample rates as input.
INPUT_AUDIO_LENGTH    = 32000                         # Maximum input audio length: recommended to be greater than 4096. Integer multiple of HOP_LENGTH (160).
IN_AUDIO_DTYPE        = 'F32'                         # ['F16', 'F32', 'INT16'] dtype of the ONNX model's input audio tensor. Default 'INT16'.
OUT_AUDIO_DTYPE       = 'F32'                         # ['F16', 'F32', 'INT16'] dtype of the ONNX model's output audio tensor. Default 'INT16'.
BATCH_WINDOW_SECONDS  = 1.5                           # Minimum input length (seconds) that triggers window folding.
USE_BATCH_FOLD        = False                         # If true, fold input-rate windows into fixed 16 kHz model-rate batches.

# Fixed RTAEC model and ONNX export parameters.
DYNAMIC_AXES          = False
OPSET                 = 20
MODEL_SAMPLE_RATE     = 16000
WINDOW_TYPE           = 'hann'
NFFT                  = 512
WINDOW_LENGTH         = 512
HOP_LENGTH            = 160
NUM_LAGS              = 96                            # 10 ms grid covering bulk delays from 0 through 950 ms.
NUM_TAPS              = 24                            # Convolutive taps after bulk-delay removal (240 ms at 16 kHz).
                                                      # 32 taps chased the last 1.3% of the long-reverb fixture's impulse
                                                      # energy at O(W^2*T) cost; 24 taps already capture 97.2%, and the
                                                      # short-reverb fixture saturates at 16 taps because its entire 74.9 ms
                                                      # span fits inside a 160 ms window. For scale, WebRTC AEC3 ships a
                                                      # 52 ms linear filter.
NUM_PRE_TAPS          = 2                             # Non-causal taps; GCC-PHAT is biased late by reverberation, so the
                                                      # direct path lands before the estimate. 0 taps loses 17 dB on the
                                                      # short-reverb fixture, but 8 wasted a quarter of the fixed tap budget
                                                      # before the direct path: 2 leaves 22 causal taps for the room tail,
                                                      # worth +3 to +7 dB linear ERLE on real speech at every tested delay.
NUM_SEGMENTS          = 8                             # Independent tracking solves per block (dynamic delay granularity).
                                                      # 4 -> 8 bought 7.4 dB on the mid-block echo-path gain step, the
                                                      # binding case: a whole-block solve cannot represent a gain change at
                                                      # constant delay, so only finer time resolution localises it.
SEGMENT_OVERLAP       = 12                            # Extra analysis frames per side. The segment solves are warm-started
                                                      # and DTD-masked rather than relying on over-determination, so this is
                                                      # cost rather than robustness; kept at 12 because cutting it to 8 or 4
                                                      # measurably costs 5-8 dB of linear-stage ERLE (34-37 vs 42 dB) for
                                                      # ~10% of runtime, which is not a good trade.
SEGMENT_CROSSFADE     = 10                            # Output crossfade width; decoupled from the analysis overlap so a
                                                      # delay switch stays temporally sharp.
GLOBAL_CG_SCHEDULE    = (8, 4)                        # Stage-1 CG steps per IRLS round (whole-block reference solve).
                                                      # The *second, weighted* round is worth 8.8 dB of linear-stage ERLE:
                                                      # a single unweighted round plateaus at 32.2 dB no matter how many
                                                      # iterations it gets (measured 12/16/20/24/32 all within 0.03 dB), so
                                                      # the IRLS re-weighting is the active ingredient, not raw iteration
                                                      # count. Two rounds at (6,4) or (5,5) retain most of it (~1 dB less)
                                                      # if this ever needs to shrink further.
SEGMENT_CG_SCHEDULE   = (5,)                          # Stage-2 warm-started CG steps per tracking segment. Each solve is
                                                      # short and seeded from stage 1; raising this costs time without
                                                      # recovering linear ERLE (measured 3/4/5/8 all within 0.1 dB).
DIAG_LOAD_DB          = -42.0                         # Ridge relative to the mean autocorrelation diagonal.
ROBUST_C              = 0.7                           # Geman-McClure scale; smaller downweights near-end speech harder.
IRLS_BAND_SMOOTH      = 4                             # Frequency smoothing radius for the robust weights.
IRLS_SCALE_MODE       = 'geometric'                   # ['geometric', 'mean'] Robust scale statistic. The arithmetic mean is
                                                      # itself dominated by near-end speech during double-talk, which drives
                                                      # every weight to 1.0 exactly when robustness is needed.
DTD_SOLVER_THRESHOLD  = 1.5                           # Residual/echo power ratio above which a frame is dropped from the
                                                      # tracking solves. Deliberately high: the solver needs evidence, so it
                                                      # should only discard frames that are genuinely near-end dominated.
DTD_RES_THRESHOLD     = 0.3                           # Same ratio for the post-filter, which instead wants to react early
                                                      # and lift its gain floor before near-end speech gets suppressed.
                                                      # 0.15 fired on 17-27% of echo-only frames with a nonlinear speaker
                                                      # (+4 dB ERLE at 0.3); 0.5 fails the double-talk guard.
DTD_HOLD              = 6                             # Causal hold (frames) after a double-talk trigger.
DTD_FLOOR             = 0.15                          # Weight retained for double-talk frames in the tracking solves.
DTD_FREEZE_LO         = 0.0                           # Clean-frame fraction below which segment adaptation fully freezes.
DTD_FREEZE_HI         = 0.35                          # Clean-frame fraction above which segments adapt freely. Between the
                                                      # two the segment filter is blended toward the carried one, so a
                                                      # segment starved of echo-only evidence cannot invent an echo path.
TDE_BAND_LO           = 2                             # Lowest STFT bin used for GCC-PHAT delay estimation.
TDE_BAND_HI           = 220                           # Highest STFT bin used for GCC-PHAT delay estimation.
TDE_SMOOTH            = 3                             # Lag smoothing width on the GCC-PHAT surface.
TDE_PRIOR_WEIGHT      = 0.35                          # Penalty on lags far from the previous segment estimate.
TDE_PRIOR_SIGMA       = 6.0                           # Width (lags) of the Gaussian continuity prior. A linear ramp
                                                      # under-penalises distant spurious peaks, which is the reverberant
                                                      # failure mode; a Gaussian decays fast enough to reject them.
TDE_COARSE_FLOOR_RATIO = 1.8                          # Minimum peak-to-surface-floor ratio for a candidate worth holding.
TDE_COARSE_SECOND_RATIO = 1.1                         # Minimum peak-to-next-separated-peak ratio for coarse confidence.
TDE_REFINED_FLOOR_RATIO = 2.5                         # Refined peaks may make an immediate local move.
TDE_REFINED_SECOND_RATIO = 1.5
TDE_PEAK_EXCLUSION    = 2                             # Radius excluded when finding the next distinct lag peak.
TDE_REFINED_RADIUS    = 2                             # Largest immediate move; wider jumps need independent support.
TDE_MIC_ACTIVITY_RATIO = 1e-4                         # Minimum mic/far-end energy (-40 dB) for a segment to vote on the lag.
                                                      # Without it, a delay beyond ~300 ms left segment 0 echo-free, its
                                                      # noise peak was accepted as lag 0, and stage 1 (anchored on the
                                                      # earliest lag) solved the whole block at the wrong delay: 6.5 dB.
PATH_DRIFT_LO         = 0.06                          # Worst per-segment stage-1 residual/echo ratio below which the echo
PATH_DRIFT_HI         = 0.25                          # path counts as stationary, and above which tracking is fully
                                                      # enabled. Catches a path change at constant delay, such as a moving
                                                      # talker or far-end AGC, which the delay tracker cannot see.
PATH_DRIFT_CLEAN_LO   = 0.85                          # Clean-frame share below which a large stage-1 residual is read as
PATH_DRIFT_CLEAN_HI   = 0.90                          # near-end speech rather than a path change. Double-talk and a real
                                                      # gain step leave near-identical residuals, so this share is the only
                                                      # statistic that tells them apart (measured 0.76 against 0.92).
PATH_SELECT_MARGIN    = 0.25                          # How much worse than stage 1 a segment solve must be before it is
                                                      # rejected, as a normalised residual-energy difference. A veto, not
                                                      # a vote: ties keep the solve, so tracking is never throttled.

AUTO_DELAY_ALIGN      = True                          # Auto-align bulk latency between near-end and far-end in the STFT domain.
TARGET_TDE_LAG        = 0                             # Target latency offset (frames); 0 directly aligns direct-path acoustic delay.
NONLINEAR_BASIS       = False                         # Hammerstein x|x| basis for loudspeaker distortion. Doubling the
                                                      # unknowns halves each tracking segment's degrees of freedom, which
                                                      # measured -16 dB on moving paths, so it stays off by default.
REVERB_DECAY          = 0.85                          # Late-reverb tail decay per frame in the post-filter.
ADAPTIVE_REVERB_DECAY = True                          # Fit the decay per block to the stage-1 filter tail instead.
REVERB_DECAY_MIN      = 0.60                          # Clamp range for the fitted per-frame power decay.
REVERB_DECAY_MAX      = 0.95
REVERB_FIT_OFFSET     = 6                             # Taps after the direct path skipped by the fit (early reflections).
REVERB_GAIN           = 0.35                          # Weight of the modelled late tail in the residual echo estimate.
RES_EST_WEIGHT        = 0.20                          # Direct echo estimate weight in residual echo power estimate.
RES_COH_WEIGHT        = 0.80                          # Complex coherence residual weight in residual echo power estimate.
RES_MIN_GAIN_DB       = -24.0                         # Deepest post-filter attenuation when echo is confidently present.
GAIN_MAX_INC_DB       = 3.0                           # Fastest the post-filter gain may rise, per frame. Only the rise is
                                                      # limited, as in AEC3: releasing the gain abruptly re-admits an
                                                      # audible echo burst, whereas closing it fast is the whole point of
                                                      # a suppressor. Worth +8.0 dB ERLE and +3.2 dB on reverberant tails.
GAIN_LIMIT_SPAN       = 8                             # Frames of gain history the rate limit is enforced over. Bounded so
                                                      # the recursion becomes a fixed-width gather, not a scan. 8 frames
                                                      # times the 3 dB step spans the full 24 dB gain range, so a longer
                                                      # window cannot bind any harder and only costs time (2.0 ms at 12).

# Regression guards, set just below the measured values of this configuration. The
# double-talk figures are bounded by the post-filter itself: an oracle with a perfect
# linear stage scores 11.5 dB keep / 13.2 dB gain, so those two can never approach the
# echo-only numbers.
#
# The linear-stage guards were re-baselined when `_linear_stage_erle_db` moved to the time
# domain. The old STFT-only ratio understated the same residual by ~19 dB (22.05 dB for a
# residual that measures 41.11 dB once reconstructed), so the previous 18.0 / 15.0 floors
# pinned a number that could not rank configurations at all. The values below sit just
# below the measured 42.2 / 21.5 dB of the shipped configuration.
MIN_SYNTHETIC_ERLE_DB = 44.0
MIN_DYNAMIC_ERLE_DB   = 27.0
MIN_REVERB_ERLE_DB    = 34.0
MIN_LONG_REVERB_ERLE_DB = 35.0
MIN_LINEAR_ERLE_DB    = 38.0
MIN_LINEAR_DYNAMIC_ERLE_DB = 19.0
MIN_NEAR_SISDR_DB     = 45.0
MIN_DOUBLE_TALK_GAIN_DB = 6.0
MIN_DOUBLE_TALK_KEEP_DB = 4.5
MIN_GAIN_STEP_ERLE_DB = 17.0
INV_INT16             = float(1.0 / 32768.0)

# Derived export dimensions.
FOLD_WINDOW_LENGTH    = ((int(BATCH_WINDOW_SECONDS * MODEL_SAMPLE_RATE) + HOP_LENGTH - 1) // HOP_LENGTH) * HOP_LENGTH
FOLD_INPUT_LENGTH     = max(1, int(round(FOLD_WINDOW_LENGTH * IN_SAMPLE_RATE / MODEL_SAMPLE_RATE)))
FOLD_OUTPUT_LENGTH    = max(1, int(round(FOLD_WINDOW_LENGTH * OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)))
EXPORT_AUDIO_LENGTH   = (((INPUT_AUDIO_LENGTH + FOLD_INPUT_LENGTH - 1) // FOLD_INPUT_LENGTH) * FOLD_INPUT_LENGTH) if USE_BATCH_FOLD else INPUT_AUDIO_LENGTH
STATIC_MODEL_BATCH    = None if DYNAMIC_AXES else (EXPORT_AUDIO_LENGTH // FOLD_INPUT_LENGTH if USE_BATCH_FOLD else 1)
MODEL_AUDIO_LENGTH    = None if DYNAMIC_AXES else (FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else int(round(EXPORT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)))
OUTPUT_AUDIO_LENGTH   = None if DYNAMIC_AXES else (STATIC_MODEL_BATCH * FOLD_OUTPUT_LENGTH if USE_BATCH_FOLD else int(round(EXPORT_AUDIO_LENGTH * OUT_SAMPLE_RATE / IN_SAMPLE_RATE)))
STATIC_SIGNAL_LENGTH  = None if DYNAMIC_AXES else (MODEL_AUDIO_LENGTH + 2 * (NFFT // 2) - NFFT) // HOP_LENGTH + 1
MAX_SIGNAL_LENGTH     = 2048 if DYNAMIC_AXES else STATIC_SIGNAL_LENGTH
# The delay tracker needs ~25 frames per segment; shorter blocks get fewer segments.
MIN_SEGMENT_FRAMES    = 25
BLOCK_SEGMENTS        = max(1, min(NUM_SEGMENTS, STATIC_SIGNAL_LENGTH // MIN_SEGMENT_FRAMES))
# A delay longer than half the block leaves too little overlapping echo to estimate.
BLOCK_LAGS            = max(1, min(NUM_LAGS, STATIC_SIGNAL_LENGTH // 2))
# Keep the block solve over-determined (>= 4 frames per tap) so ONNX parity holds.
BLOCK_TAPS            = max(NUM_PRE_TAPS + 4, min(NUM_TAPS, STATIC_SIGNAL_LENGTH // 4))
# The MIN_* quality floors were measured on 2 s blocks; shorter blocks only warn.
GUARD_CALIBRATION_FRAMES = 201
if DYNAMIC_AXES:
    raise ValueError("The optimized RTAEC exporter requires a static audio-length profile; export separate fixed profiles instead.")


class ResidualEchoSuppressor(nn.Module):
    """
    Post-filter combining coherent residual subtraction, an AEC3-style exponential late
    reverb model, and a hysteretic double-talk detector.

    The reverb term covers what the FIR cannot reach: a 320 ms tap span leaves the tail
    of a 600 ms room unmodelled, and that leftover energy is what a purely coherent
    estimator misses. The DTD hold is a causal trigger/hold state machine expressed as a
    MaxPool over a left-padded trigger, which is exactly equivalent to the sequential
    counter form and stays a single ONNX node.
    """
    def __init__(self, freq_bins=257, t_smooth=6, min_gain_db=RES_MIN_GAIN_DB,
                 reverb_decay=REVERB_DECAY, reverb_gain=REVERB_GAIN,
                 dtd_hold=DTD_HOLD, signal_length=STATIC_SIGNAL_LENGTH,
                 res_est_weight=RES_EST_WEIGHT, res_coh_weight=RES_COH_WEIGHT):
        super().__init__()
        self.freq_bins = freq_bins
        self.signal_length = signal_length
        self.min_gain = 10.0 ** (min_gain_db / 20.0)
        self.one_minus_min_gain = 1.0 - self.min_gain
        self.c_dtd = 0.85 * self.one_minus_min_gain
        self.inv_dtd_range = float(1.0 / 1.5)
        self.dtd_offset = float(-0.5 / 1.5)
        self.reverb_decay = reverb_decay
        self.reverb_gain = reverb_gain
        self.dtd_hold = dtd_hold
        self.res_est_weight = res_est_weight
        self.res_coh_weight = res_coh_weight
        self.register_buffer(
            'pad_zeros',
            torch.zeros(STATIC_MODEL_BATCH, 5, freq_bins, t_smooth - 1, dtype=torch.float32),
        )
        self.register_buffer(
            'time_kernel',
            torch.full((5, 1, 1, t_smooth), 1.0 / t_smooth, dtype=torch.float32),
        )
        # Truncated exponential: the causal IIR tail as one FIR kernel, so no Scan/Loop.
        # The kernel is not normalised to unit DC gain: `reverb_gain` scales the modelled
        # tail, and the residual is *subtracted*, so an over-large tail would suppress
        # genuine near-end speech rather than merely miss echo.
        tail_length = min(signal_length, 48)
        self.register_buffer('reverb_exponent', torch.arange(
            1, tail_length + 1, dtype=torch.float32).flip(0).reshape(1, 1, tail_length))
        self.tail_length = tail_length

        self.gain_limit_span = min(GAIN_LIMIT_SPAN, signal_length)
        max_inc = 10.0 ** (GAIN_MAX_INC_DB / 20.0)
        # `inc ** j` for each lag j in the window, so the windowed minimum of the scaled
        # history equals the recursive per-frame bound.
        lag_axis = torch.arange(self.gain_limit_span, dtype=torch.float32).flip(0)
        self.register_buffer('ramp_up', (max_inc ** lag_axis).reshape(1, -1, 1))
        self.register_buffer(
            'window_index',
            (torch.arange(self.gain_limit_span, dtype=torch.int64).reshape(1, -1, 1)
             + torch.arange(signal_length, dtype=torch.int64).reshape(1, 1, -1)
             ).expand(STATIC_MODEL_BATCH * freq_bins, -1, -1).contiguous(),
        )


    def _late_reverb(self, echo_power, decay):
        """Exponentially decaying tail of past echo power, strictly causal, per bin."""
        kernel = torch.exp(self.reverb_exponent * torch.log(decay)) * (1.0 - decay)
        lanes = echo_power.reshape(-1, 1, self.signal_length)
        padded = F.pad(lanes, (self.tail_length, 0))[..., :-1]
        return F.conv1d(padded, kernel).reshape_as(echo_power)

    def _rate_limit(self, gain):
        """
        Asymmetric per-bin slew limit on the post-filter gain, as in AEC3's suppressor.

        The exact recursion `g[t] <= g[t-1] * inc` is sequential, but unrolling it gives
        `g[t] <= min over j of (g[t-j] * inc**j)`, a windowed minimum of a geometrically
        scaled history: a gather plus a reduction, so the limiter stays a few static ONNX
        nodes with no Scan or Loop. Staying multiplicative rather than working in the log
        domain matters for export, because log/exp around a near-zero gain amplified the
        ONNX-vs-PyTorch error past the parity tolerance. The window is finite rather than
        infinite, which is sufficient: a gain excursion only needs bounding for as long as
        it is audible.

        Only the rise is limited. The closed form above is exact for a one-sided bound, but
        the two-sided recursion is not the composition of two one-sided ones, and applying
        both cost exactness (measured 8.7x per-frame jumps against a 1.41x limit). The
        rise is the direction worth constraining: releasing the gain too quickly re-admits
        an audible echo burst, whereas closing it quickly is what suppression is for.
        """
        frames = self.signal_length
        lanes = gain.reshape(-1, 1, frames)
        padded = F.pad(lanes, (self.gain_limit_span - 1, 0), value=1.0)
        # A single strided gather beats unrolled slices here, and `unfold` has no ONNX
        # lowering; the index is pre-expanded to its static shape so no Expand survives.
        windows = torch.gather(padded.expand(-1, self.gain_limit_span, -1), 2,
                               self.window_index)
        rise = (windows * self.ramp_up).amin(dim=1, keepdim=True)
        return torch.minimum(lanes, rise).reshape_as(gain)

    def forward(self, linear_res, echo_est, aligned_far, reverb_decay):
        e_r, e_i = linear_res.split(1, dim=1)
        d_r, d_i = echo_est.split(1, dim=1)
        f_r, f_i = aligned_far.split(1, dim=1)

        p_res = e_r * e_r + e_i * e_i
        p_est = d_r * d_r + d_i * d_i
        p_far = f_r * f_r + f_i * f_i
        cross_re = e_r * d_r + e_i * d_i
        cross_im = e_i * d_r - e_r * d_i

        stats = torch.cat([p_res, p_est, p_far, cross_re, cross_im], dim=1)
        padded_stats = torch.cat([self.pad_zeros, stats], dim=-1)
        smoothed = F.conv2d(padded_stats, self.time_kernel, groups=5)
        p_res_s, p_est_s, p_far_s, cross_re_s, cross_im_s = smoothed.split(1, dim=1)

        res_frame_energy = p_res_s.sum(dim=2, keepdim=True)
        est_frame_energy = p_est_s.sum(dim=2, keepdim=True)

        far_active = torch.clamp(p_far_s * 1e5 - 0.1, min=0.0, max=1.0)

        # Hysteretic DTD: trigger on near-end dominance, then hold (AEC3 hold_counters).
        trigger = (res_frame_energy > DTD_RES_THRESHOLD * est_frame_energy).to(linear_res.dtype)
        if self.dtd_hold > 1:
            trigger = F.max_pool1d(
                F.pad(trigger.reshape(1, 1, -1), (self.dtd_hold - 1, 0)),
                self.dtd_hold, stride=1,
            ).reshape_as(trigger)
        dtd_ratio = res_frame_energy * torch.reciprocal(est_frame_energy + 1e-12)
        dtd = torch.clamp(dtd_ratio * self.inv_dtd_range + self.dtd_offset, min=0.0, max=1.0)
        dtd = torch.maximum(dtd, trigger * 0.5)

        cross_mag_sq = cross_re_s.square() + cross_im_s.square()
        coh = torch.clamp(cross_mag_sq * torch.reciprocal(p_res_s * p_est_s + 1e-12),
                          min=0.0, max=1.0)
        late = self._late_reverb(p_est_s, reverb_decay) * self.reverb_gain
        res_echo_pow = ((self.res_est_weight * p_est_s + self.res_coh_weight * coh * p_res_s + late)
                        * far_active * (1.0 - 0.7 * dtd))

        floor = torch.clamp(1.0 - self.one_minus_min_gain * far_active + self.c_dtd * dtd, max=1.0)
        raw_gain = (p_res_s - res_echo_pow) * torch.reciprocal(p_res_s + 1e-12)
        gain = torch.clamp(torch.maximum(raw_gain, floor), min=self.min_gain, max=1.0)
        # Re-applying the floor keeps the limiter from holding the gain below what the
        # double-talk floor guarantees near-end speech.
        gain = torch.maximum(self._rate_limit(gain), floor)
        return linear_res * gain


class RTAECModel(nn.Module):
    """
    Two-stage frequency-domain convolutive AEC.

    Stage 1 solves one block-Wiener filter over the whole block. With T frames against
    2K real unknowns per bin it is strongly over-determined, so it cannot absorb near-end
    speech, which makes its residual a trustworthy double-talk detector.

    Stage 2 runs short per-segment solves that track a moving delay. Each is warm-started
    from stage 1 and masked by the stage-1 double-talk decision, so the segments gain
    time resolution without gaining the freedom to fit near-end speech. Solving the
    segments directly at their own integer lag is what makes dynamic delay tracking
    correct: the reference is never warped, so every individual solve stays
    time-invariant, which is precisely what the block-Wiener normal equations assume.
    """
    def __init__(self, custom_istft, num_lags=NUM_LAGS, num_taps=NUM_TAPS,
                 num_pre_taps=NUM_PRE_TAPS, num_segments=BLOCK_SEGMENTS,
                 segment_overlap=SEGMENT_OVERLAP, segment_crossfade=SEGMENT_CROSSFADE,
                 global_cg=GLOBAL_CG_SCHEDULE, segment_cg=SEGMENT_CG_SCHEDULE,
                 output_length=32000, auto_delay_align=AUTO_DELAY_ALIGN,
                 target_lag=TARGET_TDE_LAG, nonlinear_basis=NONLINEAR_BASIS,
                 reverb_gain=REVERB_GAIN, reverb_decay=REVERB_DECAY,
                 res_min_gain_db=RES_MIN_GAIN_DB):
        super().__init__()
        self.custom_istft = custom_istft
        self.num_lags = num_lags
        self.num_taps = num_taps
        self.num_pre_taps = num_pre_taps
        self.num_segments = num_segments
        self.global_cg = tuple(global_cg)
        self.segment_cg = tuple(segment_cg)
        self.output_length = output_length
        self.model_batch = STATIC_MODEL_BATCH
        self.signal_length = STATIC_SIGNAL_LENGTH
        self.freq_bins = NFFT // 2 + 1
        self.ridge_scale = 10.0 ** (DIAG_LOAD_DB / 10.0)
        self.robust_c_sq = ROBUST_C * ROBUST_C
        self.band_smooth = IRLS_BAND_SMOOTH
        self.path_select_margin = PATH_SELECT_MARGIN
        self.auto_delay_align = auto_delay_align
        self.nonlinear_basis = nonlinear_basis
        self.num_channels = 2 if nonlinear_basis else 1
        self.width = self.num_taps * self.num_channels
        self.bounds = _segment_bounds(self.signal_length, num_segments, segment_overlap)
        # Left context so that even the largest lag plus the tap window reads real samples.
        self.pad_left = (num_lags - 1) + (num_taps - 1)
        self.pad_right = num_pre_taps
        self.pad_total = self.pad_left + self.pad_right
        # The tap skew reads one sample past the last tap it keeps.
        self.pad_right_skew = self.pad_right + 1

        if self.auto_delay_align:
            self.bulk_delay_align = DynamicDelayAlign(
                NFFT, HOP_LENGTH, MODEL_SAMPLE_RATE, self.signal_length,
                batch=STATIC_MODEL_BATCH, num_lags=num_lags, estimate_only=True,
                target_lag=target_lag, lo=TDE_BAND_LO, hi=TDE_BAND_HI,
                num_segments=num_segments, overlap=segment_overlap,
                prior_weight=TDE_PRIOR_WEIGHT, smooth=TDE_SMOOTH,
                prior_sigma=TDE_PRIOR_SIGMA,
                coarse_floor_ratio=TDE_COARSE_FLOOR_RATIO,
                coarse_second_ratio=TDE_COARSE_SECOND_RATIO,
                refined_floor_ratio=TDE_REFINED_FLOOR_RATIO,
                refined_second_ratio=TDE_REFINED_SECOND_RATIO,
                peak_exclusion=TDE_PEAK_EXCLUSION, refined_radius=TDE_REFINED_RADIUS,
                mic_activity_ratio=TDE_MIC_ACTIVITY_RATIO,
            )
        self.res_suppressor = ResidualEchoSuppressor(
            freq_bins=self.freq_bins, signal_length=self.signal_length,
            min_gain_db=res_min_gain_db,
            reverb_gain=reverb_gain, reverb_decay=reverb_decay)

        self.register_buffer('eye', torch.eye(self.width, dtype=torch.float32).reshape(1, self.width, self.width))
        self.register_buffer('zero_filter', torch.zeros(self.freq_bins, self.width, 2, dtype=torch.float32))
        # (ur, ui) + (lr, li) @ twist = (ur - li, ui + lr): the complex product's twist.
        self.register_buffer('complex_twist',
                             torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32))
        self.register_buffer('default_lag', torch.zeros(1, dtype=torch.int32))
        self.register_buffer('unit_weight', torch.ones(self.freq_bins, 1, self.signal_length,
                                                       dtype=torch.float32))
        self.register_buffer(
            'diag_index',
            torch.arange(0, self.width * self.width, self.width + 1, dtype=torch.int32))
        # Rows of the stacked [taps_r; taps_i] operand that hold tap 0 of each part.
        self.register_buffer('tap0_index', torch.tensor([0, self.width], dtype=torch.int32))
        self.register_buffer('fixed_decay', torch.full((1, 1, 1), reverb_decay, dtype=torch.float32))
        # Centred least-squares slope weights over the tail taps.
        self.fit_start = min(num_pre_taps + REVERB_FIT_OFFSET, num_taps - 2)
        fit_axis = torch.arange(num_taps - self.fit_start, dtype=torch.float32)
        fit_axis = fit_axis - fit_axis.mean()
        self.register_buffer('decay_fit', (fit_axis / fit_axis.square().sum()).reshape(-1, 1))
        self.inv_freeze_range = float(1.0 / max(DTD_FREEZE_HI - DTD_FREEZE_LO, 1e-6))
        drift_range = max(PATH_DRIFT_HI - PATH_DRIFT_LO, 1e-6)
        self.drift_scale = float(1.0 / drift_range)
        self.drift_offset = float(PATH_DRIFT_LO / drift_range)
        self.drift_clean_scale = float(
            1.0 / max(PATH_DRIFT_CLEAN_HI - PATH_DRIFT_CLEAN_LO, 1e-6))
        crossfade = _crossfade_profile(self.signal_length, num_segments, segment_crossfade)
        # Contiguous reference window per analysis span, biased by the span start so the
        # forward pass adds nothing but the lag; `_build_taps` skews it into taps.
        for index, (start, end, _, _) in enumerate(self.bounds):
            span = end - start
            self.register_buffer(f'tap_window_{index}', torch.arange(
                start - (num_taps - 1), end + 1, dtype=torch.int32))
            self.register_buffer(f'blend_{index}', crossfade[index, start:end].reshape(1, 1, span))
        self.register_buffer('tap_window_global', torch.arange(
            -(num_taps - 1), self.signal_length + 1, dtype=torch.int32))

        # Core span selector matrix for vectorizing path drift ratio reductions into one matmul.
        selector = torch.zeros(self.signal_length, num_segments, dtype=torch.float32)
        for index, (_, _, core_start, core_end) in enumerate(self.bounds):
            selector[core_start:core_end, index] = 1.0
        self.register_buffer('core_selector', selector)

    def _reference_stack(self, far_comp):
        """
        Padded reference, optionally with the Hammerstein x|x| distortion basis.

        Channels stay grouped as [all real parts, all imaginary parts] so that the
        real/imaginary split in `_build_taps` is a single contiguous cut.
        """
        if self.nonlinear_basis:
            real, imag = far_comp.split(1, dim=1)
            magnitude = (real * real + imag * imag).sqrt()
            far_comp = torch.cat([real, real * magnitude, imag, imag * magnitude], dim=1)
        return F.pad(far_comp, (self.pad_left, self.pad_right_skew))

    def _build_taps(self, padded, window, lag, span):
        """
        Lagged reference copies at one integer lag; no reference warping is involved.

        Tap k at frame t is window[t - k + K - 1]. Rather than gathering all K * span
        elements, the window is repeated K times with a row stride of L - 1 instead of L,
        which shifts each row one sample right: a contiguous Expand/Reshape/Slice that is
        bit-exact and ~2.5x faster in ORT than the per-element Gather it replaces.
        """
        base = self.pad_total - lag.reshape(()).to(torch.int32)
        rows = 2 * self.num_channels * self.freq_bins
        taps = self.num_taps
        length = span + taps
        flat = padded.reshape(rows, -1)
        line = torch.index_select(flat, -1, base + window)
        skew = line.unsqueeze(1).expand(rows, taps, length).reshape(rows, taps * length)
        skew = skew[:, :taps * (length - 1)].reshape(rows, taps, length - 1)
        selected = skew[..., taps - 1:taps - 1 + span]
        if self.num_channels == 1:
            # The channel permutation is the identity here, so gather straight into shape.
            real, imag = selected.reshape(2 * self.freq_bins, self.width, span).split(
                self.freq_bins, dim=0)
            return real, imag
        selected = selected.reshape(2, self.num_channels, self.freq_bins, self.num_taps, span)
        real, imag = selected.split(1, dim=0)
        real = real.reshape(self.num_channels, self.freq_bins, self.num_taps, span)
        imag = imag.reshape(self.num_channels, self.freq_bins, self.num_taps, span)
        real = real.permute(1, 0, 2, 3).reshape(self.freq_bins, self.width, span)
        imag = imag.permute(1, 0, 2, 3).reshape(self.freq_bins, self.width, span)
        return real, imag

    def _gram_operands(self, taps_r, taps_i, mic_r, mic_i):
        """
        The two matmul operands, which depend only on the taps and the microphone.

        IRLS re-solves the same system under new weights, so these concatenations are
        loop-invariant; building them once per tap set removes a full copy of the largest
        tensors from every round after the first.
        """
        return (torch.cat([taps_r, mic_r, taps_i, mic_i], dim=1).transpose(-1, -2),
                torch.cat([taps_r, taps_i], dim=1))

    def _normal_equations(self, operands, weight):
        """Weighted Hermitian Gram matrix and cross-correlation right-hand side."""
        width = self.width
        bundle, stacked_taps = operands
        weighted = stacked_taps if weight is None else stacked_taps * weight
        product = torch.matmul(weighted, bundle)
        top, bottom = product.split(width, dim=1)
        wr_tr, wr_ti = top.split(width + 1, dim=-1)
        wi_tr, wi_ti = bottom.split(width + 1, dim=-1)
        real_part = wr_tr + wi_ti
        imag_part = wi_tr - wr_ti
        matrix_r, rhs_r = real_part.split([width, 1], dim=-1)
        matrix_i, rhs_i = imag_part.split([width, 1], dim=-1)
        diag = torch.index_select(
            matrix_r.reshape(self.freq_bins, width * width), -1, self.diag_index
        ).reshape(self.freq_bins, width, 1)
        ridge = diag.mean(dim=1, keepdim=True) * self.ridge_scale + 1e-12
        # The CG state is packed as (Fb, W, 2), so hand back a packed right-hand side.
        stacked = torch.cat([matrix_r + self.eye * ridge, matrix_i], dim=1)
        return stacked, torch.cat([rhs_r, rhs_i], dim=-1), torch.reciprocal(diag + ridge)

    def _conjugate_gradient(self, stacked, rhs, precond, sol, iterations, is_zero):
        """
        Jacobi-preconditioned CG on the Hermitian system.

        Real and imaginary parts ride together in a trailing length-2 axis, so every
        vector update is one node rather than two and each inner product is a single
        reduction over both axes. The complex twist `(ur - li, ui + lr)` becomes a 2x2
        constant matmul on that axis, which is cheaper than a flip plus a signed multiply.
        """
        width = self.width

        def apply(direction):
            product = torch.matmul(stacked, direction)
            upper, lower = product.split(width, dim=1)
            return upper + torch.matmul(lower, self.complex_twist)

        if is_zero:
            res = rhs
        else:
            res = rhs - apply(sol)
        z = res * precond
        direction = z
        rz = (res * z).sum(dim=(1, 2), keepdim=True)
        for it in range(iterations):
            curve = apply(direction)
            step = rz / ((direction * curve).sum(dim=(1, 2), keepdim=True) + 1e-20)
            sol = sol + step * direction
            if it == iterations - 1:
                break
            res = res - step * curve
            z = res * precond
            rz_next = (res * z).sum(dim=(1, 2), keepdim=True)
            direction = z + (rz_next / (rz + 1e-20)) * direction
            rz = rz_next
        return sol

    def _synthesize_echo(self, filt, taps_r, taps_i, span):
        """
        Echo synthesis as two (Fb,2,W)x(Fb,W,span) matmuls.

        Stacking along the filter axis instead of the span axis keeps both operands
        span-sized rather than 2*span-sized, which is ~4.8x faster here because this
        kernel is memory-bound on the tap tensors, not compute-bound. The packed CG
        state already carries real/imag side by side, so this is a bare transpose.
        """
        filt = filt.transpose(-1, -2)
        prod_r = torch.matmul(filt, taps_r)
        prod_i = torch.matmul(filt, taps_i)
        fr_tr, fi_tr = prod_r.split(1, dim=1)
        fr_ti, fi_ti = prod_i.split(1, dim=1)
        return fr_tr + fi_ti, fr_ti - fi_tr

    def _robust_weight(self, res_r, res_i, span):
        """
        Geman-McClure weights against a geometric-mean scale.

        The arithmetic mean of the residual power is itself inflated by near-end speech,
        which pushes every weight to 1.0 exactly during double-talk. The geometric mean
        tracks the echo-only floor instead, so loud near-end frames stay downweighted.
        """
        power = (res_r * res_r + res_i * res_i).reshape(1, 1, self.freq_bins, span)
        power = F.avg_pool2d(
            power, (2 * self.band_smooth + 1, 1), stride=1,
            padding=(self.band_smooth, 0), count_include_pad=False,
        ).reshape(self.freq_bins, 1, span)
        if IRLS_SCALE_MODE == 'geometric':
            scale = torch.exp(torch.log(power + 1e-12).mean(dim=-1, keepdim=True))
        else:
            scale = power.mean(dim=-1, keepdim=True)
        scale = scale * self.robust_c_sq + 1e-12
        return scale / (power + scale)

    def _double_talk_mask(self, res_r, res_i, echo_r, echo_i):
        """Frame-level near-end dominance with a causal hold, from the stage-1 residual."""
        res_power = (res_r * res_r + res_i * res_i).sum(dim=0, keepdim=True)
        echo_power = (echo_r * echo_r + echo_i * echo_i).sum(dim=0, keepdim=True)
        trigger = (res_power > DTD_SOLVER_THRESHOLD * echo_power).to(res_r.dtype)
        if DTD_HOLD > 1:
            trigger = F.max_pool1d(F.pad(trigger, (DTD_HOLD - 1, 0)), DTD_HOLD, stride=1)
        return 1.0 - (1.0 - DTD_FLOOR) * trigger, res_power, echo_power

    def _adapt_trust(self, mask):
        """How far a segment may move from the carried filter, from its clean-frame share."""
        clean = mask.mean()
        return torch.clamp((clean - DTD_FREEZE_LO) * self.inv_freeze_range, 0.0, 1.0)

    @staticmethod
    def _path_error(res_r, res_i):
        """Total residual energy of one candidate filter over a segment."""
        return (res_r * res_r + res_i * res_i).sum()

    def _path_drift(self, res_power, echo_power, dtd_mask):
        """
        How badly stage 1's single time-invariant filter failed to explain the block.

        A delay move is not the only reason it can fail; an echo-path gain or spectral
        change at constant delay is invisible to the delay tracker but leaves the same
        symptom, a segment stage 1 cannot fit. The worst per-segment residual-to-echo
        ratio is that symptom, and it separates the cases by more than an order of
        magnitude (measured 0.02 stationary against 0.40 for a mid-block gain step).
        Relative spread does not work here: when stage 1 fits well every ratio is near
        zero, so their spread is dominated by noise and reads highest on clean blocks.

        Near-end speech produces a numerically indistinguishable symptom without any path
        change (measured 0.42 against 0.35 for a real gain step), so the ratio alone
        cannot be trusted. What does separate them is how much echo-only evidence the
        block holds: those same two cases score 0.76 and 0.92 clean-frame share. The
        ratio is therefore admitted only in proportion to that share, so double-talk
        cannot unfreeze the very adaptation the mask exists to freeze.
        """
        res_frame = (res_power * dtd_mask).reshape(1, self.signal_length)
        echo_frame = (echo_power * dtd_mask).reshape(1, self.signal_length)
        res_core = torch.matmul(res_frame, self.core_selector)
        echo_core = torch.matmul(echo_frame, self.core_selector)
        worst = (res_core / (echo_core + 1e-20)).amax()
        drift = torch.clamp(worst * self.drift_scale - self.drift_offset, 0.0, 1.0)
        evidence = torch.clamp(
            (dtd_mask.mean() - PATH_DRIFT_CLEAN_LO) * self.drift_clean_scale, 0.0, 1.0)
        return (drift * evidence).reshape(1, 1, 1)

    def _path_select(self, carried_error, solved_error):
        """
        Per-bin two-path preference, the batch analogue of AEC3's subtractor output
        analyzer and SpeexDSP's foreground/background swap.

        Without this, a segment solve that is *worse* than the filter it replaces is still
        adopted, because `_adapt_trust` only looks at double-talk share and never at
        whether the new filter actually explains the microphone better. The decision is a
        hard choice, not a soft blend: the two candidates are built at different lags, and
        echoes at different lags do not superpose, so an intermediate weight yields an
        estimate matching neither path. Softening this cost 5 dB on the reverberant cases.

        The verdict is one decision for the whole segment rather than per bin. On a moving
        path stage 1 is fitted across both delays, so it still wins in individual bins by
        accident; letting it win those bins cost 3 dB on the moving cases. Aggregating
        over bins leaves the veto sensitive only to gross failure of a whole solve, which
        is what it exists to catch.
        """
        advantage = (carried_error - solved_error) / (carried_error + solved_error + 1e-20)
        return (advantage + self.path_select_margin > 0.0).to(carried_error.dtype).reshape(1, 1, 1)

    def linear_stage(self, mix_comp, far_comp):
        """
        Two-stage linear echo estimate. Split out of `forward` so the validation harness
        measures the shipped solver rather than a copy of it.
        """
        batch = self.model_batch
        frames = self.signal_length
        padded = self._reference_stack(far_comp)
        mic_r, mic_i = mix_comp.reshape(2 * self.freq_bins, 1, frames).split(self.freq_bins, dim=0)

        if self.auto_delay_align:
            segment_lags = self.bulk_delay_align(mix_comp, far_comp)
        else:
            segment_lags = [self.default_lag] * self.num_segments

        # Stage 1: whole-block solve at the earliest segment lag, so the tap window still
        # spans every later segment's echo. Over-determined, so its residual is a reliable DTD.
        all_lags = torch.stack(segment_lags)
        anchor = all_lags.amin()
        latest = all_lags.amax()
        # A delay move guarantees stage 1 is wrong somewhere, but it is not the only way:
        # an echo-path gain or spectral change at constant delay needs tracking too. So
        # this is a hint that forces tracking on, never a gate that switches it off; the
        # residual-energy veto in `_path_select` is what rejects an unhelpful solve.
        block_moved = (latest != anchor).to(mix_comp.dtype).reshape(1, 1, 1)
        taps_r, taps_i = self._build_taps(padded, self.tap_window_global, anchor, frames)
        operands = self._gram_operands(taps_r, taps_i, mic_r, mic_i)
        # Tap 0 of the stacked operand is the reference at the dominant lag, which is what
        # the post-filter coherence term needs; one gather instead of two slices and a cat.
        anchor_far = torch.index_select(operands[1], 1, self.tap0_index)
        weight = None  # First IRLS round is unweighted, so the full-size multiply is skipped.
        filt = self.zero_filter
        is_zero = True
        last_round = len(self.global_cg) - 1
        for round_index, iterations in enumerate(self.global_cg):
            stacked, rhs, precond = self._normal_equations(operands, weight)
            filt = self._conjugate_gradient(stacked, rhs, precond, filt, iterations, is_zero)
            is_zero = False
            echo_r, echo_i = self._synthesize_echo(filt, taps_r, taps_i, frames)
            res_r, res_i = mic_r - echo_r, mic_i - echo_i
            # The final round's weights would only feed a solve that never happens.
            if round_index != last_round:
                weight = self._robust_weight(res_r, res_i, frames)
        dtd_mask, res_power, echo_power = self._double_talk_mask(res_r, res_i, echo_r, echo_i)
        reverb_decay = self._reverb_decay(filt) if ADAPTIVE_REVERB_DECAY else self.fixed_decay
        anchor_echo_r, anchor_echo_i = echo_r, echo_i
        # A delay move is not the only reason stage 1 can be wrong: an echo-path gain or
        # spectral change at constant delay needs tracking just as much. A time-invariant
        # fit leaves a residual that is unevenly distributed across the block, so an
        # uneven per-segment residual share is the evidence for that case.
        track = torch.maximum(
            block_moved, self._path_drift(res_power, echo_power, dtd_mask))

        # Stage 2: per-segment tracking solves at their own lag, warm-started and DT-masked.
        parts = []
        for index, (start, end, _, _) in enumerate(self.bounds):
            span = end - start
            taps_r, taps_i = self._build_taps(
                padded, getattr(self, f'tap_window_{index}'), segment_lags[index], span)
            seg_mic_r, seg_mic_i = mic_r[..., start:end], mic_i[..., start:end]
            seg_mask = dtd_mask[..., start:end]
            carried = filt
            seg = filt
            echo_c_r, echo_c_i = self._synthesize_echo(seg, taps_r, taps_i, span)
            # Stage 1's own echo, so the comparison spans both the filter and the lag the
            # taps were built at. Scoring the carried filter on the segment's taps instead
            # would make a wrong lag invisible: both candidates would share it.
            carried_error = self._path_error(
                seg_mic_r - anchor_echo_r[..., start:end],
                seg_mic_i - anchor_echo_i[..., start:end])
            weight = self._robust_weight(
                seg_mic_r - echo_c_r, seg_mic_i - echo_c_i, span) * seg_mask
            operands = self._gram_operands(taps_r, taps_i, seg_mic_r, seg_mic_i)
            last_segment_round = len(self.segment_cg) - 1
            for round_index, iterations in enumerate(self.segment_cg):
                stacked, rhs, precond = self._normal_equations(operands, weight)
                seg = self._conjugate_gradient(stacked, rhs, precond, seg, iterations, False)
                echo_s_r, echo_s_i = self._synthesize_echo(seg, taps_r, taps_i, span)
                if round_index != last_segment_round:
                    weight = self._robust_weight(
                        seg_mic_r - echo_s_r, seg_mic_i - echo_s_i, span) * seg_mask
            solved_error = self._path_error(seg_mic_r - echo_s_r, seg_mic_i - echo_s_i)
            prefer = self._path_select(carried_error, solved_error)
            take = track * prefer
            trust = self._adapt_trust(seg_mask) * take
            seg = carried + (seg - carried) * trust
            # Exploit linearity of synthesize_echo to avoid 3rd synthesize_echo call
            echo_r = echo_c_r + (echo_s_r - echo_c_r) * trust
            echo_i = echo_c_i + (echo_s_i - echo_c_i) * trust
            # The verdict also gates the echo itself, not just the filter: a wrong lag is
            # carried by the taps, so blending coefficients alone cannot undo it.
            echo_r = echo_r * take + anchor_echo_r[..., start:end] * (1.0 - take)
            echo_i = echo_i * take + anchor_echo_i[..., start:end] * (1.0 - take)
            filt = seg
            blend = getattr(self, f'blend_{index}')
            # Static zero-padding then a plain sum, so the crossfade accumulation lowers
            # to adds rather than a ScatterND per segment.
            pad = (start, frames - end)
            parts.append(F.pad(torch.cat([echo_r, echo_i], dim=0) * blend, pad))

        total = parts[0]
        for part in parts[1:]:
            total = total + part
        echo_est = total.reshape(batch, 2, self.freq_bins, frames)
        return echo_est, anchor_far, reverb_decay

    def _reverb_decay(self, filt):
        """Per-frame power decay of the room, fitted to the stage-1 filter's tail taps."""
        tap_energy = filt[:, :self.num_taps].square().sum(dim=(0, 2)).reshape(1, self.num_taps)
        slope = torch.matmul(torch.log(tap_energy[:, self.fit_start:] + 1e-20), self.decay_fit)
        return torch.clamp(torch.exp(slope), REVERB_DECAY_MIN, REVERB_DECAY_MAX).reshape(1, 1, 1)

    def forward(self, mix_comp, far_comp):
        shape = (self.model_batch, 2, self.freq_bins, self.signal_length)
        echo_est, anchor_far, reverb_decay = self.linear_stage(mix_comp, far_comp)
        clean_res = self.res_suppressor(
            mix_comp - echo_est, echo_est, anchor_far.reshape(shape), reverb_decay)
        return self.custom_istft.forward_packed(
            clean_res.reshape(self.model_batch, 2 * self.freq_bins, self.signal_length))


class RTAEC(nn.Module):
    """End-to-end RTAEC wrapper with a fixed 16 kHz model path and PCM scaling."""
    def __init__(self, rtaec_model, custom_stft, in_sample_rate=16000, out_sample_rate=16000,
                 use_batch_fold=False, fold_window=0, model_audio_length=0, output_audio_length=0):
        super().__init__()
        self.model = rtaec_model
        self.custom_stft = custom_stft
        self.use_batch_fold = use_batch_fold
        self.fold_window = fold_window
        self.model_audio_length = model_audio_length
        self.output_audio_length = output_audio_length
        self.model_batch = STATIC_MODEL_BATCH
        self.signal_length = STATIC_SIGNAL_LENGTH
        self.freq_bins = NFFT // 2 + 1
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.in_sample_rate_scale = in_sample_rate / MODEL_SAMPLE_RATE
        self.out_sample_rate_scale = out_sample_rate / MODEL_SAMPLE_RATE
        self.resample_before_centering = self.in_sample_rate_scale > 1.0
        self.resample_after_centering = self.in_sample_rate_scale < 1.0
        self.output_resample_before_pcm = self.out_sample_rate_scale < 1.0
        self.output_resample_after_pcm = self.out_sample_rate_scale > 1.0
        self.input_pcm_scale_folded = "int" in IN_AUDIO_DTYPE.lower() and custom_stft.input_scale != 1.0
        self.output_pcm_scale_folded = (
            "int" in OUT_AUDIO_DTYPE.lower()
            and hasattr(rtaec_model.custom_istft, 'inv_win_sum')
            and rtaec_model.custom_istft.output_scale != 1.0
        )
        self.register_buffer('inv_int16', torch.tensor(INV_INT16, dtype=torch.float32))
        self.register_buffer('pcm_scale', torch.tensor(32767.0, dtype=torch.float32))

    def forward(self, near_end_audio, far_end_audio):
        batch = self.model_batch
        audio_pair = torch.cat([near_end_audio, far_end_audio], dim=0).float()
        if self.use_batch_fold:
            audio_pair = audio_pair.reshape(2 * batch, 1, self.fold_window)
        if self.resample_before_centering:
            audio_pair = F.interpolate(audio_pair, size=self.model_audio_length,
                                       mode='linear', align_corners=False)
        if "int" in IN_AUDIO_DTYPE.lower() and not self.input_pcm_scale_folded:
            audio_pair = audio_pair * self.inv_int16
        audio_pair = audio_pair - audio_pair.mean(dim=2, keepdim=True)
        if self.resample_after_centering:
            audio_pair = F.interpolate(audio_pair, size=self.model_audio_length,
                                       mode='linear', align_corners=False)
        pair_comp = self.custom_stft(audio_pair).view(
            2 * batch, 2, self.freq_bins, self.signal_length)
        mix_comp, far_comp = pair_comp.split(batch, dim=0)

        audio = self.model(mix_comp, far_comp)
        if self.output_resample_before_pcm:
            audio = F.interpolate(audio, size=self.output_audio_length,
                                  mode='linear', align_corners=False)
        if "int" in OUT_AUDIO_DTYPE.lower() and not self.output_pcm_scale_folded:
            audio = audio * self.pcm_scale
        if self.output_resample_after_pcm:
            audio = F.interpolate(audio, size=self.output_audio_length,
                                  mode='linear', align_corners=False)
        if self.use_batch_fold:
            audio = audio.reshape(1, 1, -1)
        if "int" in OUT_AUDIO_DTYPE.lower():
            return audio.clamp(min=-32768.0, max=32767.0).to(torch.int16)
        if "32" in OUT_AUDIO_DTYPE:
            return audio
        return audio.to(torch.float16)


def _static_shape(value_info):
    shape = value_info.type.tensor_type.shape
    dims = []
    for dim in shape.dim:
        if not dim.HasField("dim_value"):
            return None
        dims.append(int(dim.dim_value))
    return tuple(dims)


def _torch_dtype_to_onnx(dtype):
    return {
        torch.float32: onnx.TensorProto.FLOAT,
        torch.float16: onnx.TensorProto.FLOAT16,
        torch.int16: onnx.TensorProto.INT16,
    }[dtype]


def _input_torch_dtype():
    if "32" in IN_AUDIO_DTYPE:
        return torch.float32
    if "int" in IN_AUDIO_DTYPE.lower():
        return torch.int16
    return torch.float16


def _validation_audio(dtype, length, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (1, 1, length)
    if dtype == torch.int16:
        return torch.randint(-30000, 30000, shape, dtype=dtype, generator=generator)
    return torch.randn(shape, dtype=torch.float32, generator=generator).to(dtype)


def _speech_like_validation_signal(length, seed, rms):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((1, 1, length), dtype=torch.float32, generator=generator)
    kernel = torch.tensor([0.08, 0.16, 0.24, 0.04, -0.12, -0.08], dtype=torch.float32).reshape(1, 1, -1)
    signal = F.conv1d(F.pad(noise, (kernel.shape[-1] - 1, 0)), kernel)
    envelope_frames = (length + HOP_LENGTH - 1) // HOP_LENGTH
    envelope = 0.15 + 0.85 * torch.rand((1, 1, envelope_frames), generator=generator)
    signal = signal * envelope.repeat_interleave(HOP_LENGTH, dim=-1)[..., :length]
    return signal * (rms / torch.sqrt(torch.mean(signal * signal)))


def _apply_echo_path(far, taps):
    """Convolve the far-end with a sparse room impulse response given as (delay_s, gain)."""
    echo = torch.zeros_like(far)
    for delay_seconds, gain in taps:
        delay = int(round(delay_seconds * IN_SAMPLE_RATE))
        if 0 < delay < far.shape[-1]:
            echo[..., delay:] += gain * far[..., :-delay]
    return echo


def _short_exponential_echo_path(first_delay, t60, num_taps=600, seed=3):
    """Short dense decay used by the existing in-window reverberation regression."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    taps = []
    for index in range(num_taps):
        offset = index / IN_SAMPLE_RATE * 2.0
        gain = float(torch.randn(1, generator=generator)) * float(np.exp(-3.0 * offset / t60)) * 0.3
        taps.append((first_delay + offset, gain))
    return taps


def _long_reverb_impulse_response(first_delay, t60, tail_seconds=0.8, seed=5):
    """Deterministic early reflections followed by an energy-normalized dense tail."""
    first_sample = int(round(first_delay * IN_SAMPLE_RATE))
    tail_start = int(round(0.050 * IN_SAMPLE_RATE))
    tail_samples = int(round(tail_seconds * IN_SAMPLE_RATE)) + 1
    impulse_response = torch.zeros(first_sample + tail_start + tail_samples, dtype=torch.float32)
    for offset_seconds, gain in ((0.0, 0.68), (0.0073, -0.21), (0.0218, 0.14), (0.0457, -0.08)):
        impulse_response[first_sample + int(round(offset_seconds * IN_SAMPLE_RATE))] += gain

    generator = torch.Generator(device="cpu").manual_seed(seed)
    time = torch.arange(tail_samples, dtype=torch.float32) / IN_SAMPLE_RATE
    tail = torch.randn(tail_samples, dtype=torch.float32, generator=generator)
    tail = tail * torch.exp(-3.0 * time / t60)
    tail = tail * (0.35 / torch.sqrt(torch.sum(tail * tail) + 1e-12))
    impulse_response[first_sample + tail_start:] += tail
    return impulse_response


def _apply_fir_echo_path(far, impulse_response):
    """Apply a dense validation-only FIR efficiently with zero-padded FFT convolution."""
    full_length = far.shape[-1] + impulse_response.numel() - 1
    fft_length = 1 << (full_length - 1).bit_length()
    far_spectrum = torch.fft.rfft(far.float(), n=fft_length)
    path_spectrum = torch.fft.rfft(
        impulse_response.reshape(1, 1, -1).to(far.device), n=fft_length)
    return torch.fft.irfft(far_spectrum * path_spectrum, n=fft_length)[..., :far.shape[-1]]


def _to_validation_input(audio, dtype):
    if dtype == torch.int16:
        return torch.round(audio * 32768.0).clamp(-32768.0, 32767.0).to(torch.int16)
    return audio.to(dtype)


def _normalized_validation_audio(audio, dtype):
    audio = audio.float()
    if dtype == torch.int16:
        audio = audio * INV_INT16
    return audio


def _resample_validation_target(audio):
    if audio.shape[-1] == OUTPUT_AUDIO_LENGTH:
        return audio
    return F.interpolate(audio, size=OUTPUT_AUDIO_LENGTH, mode="linear", align_corners=False)


def _si_sdr_db(estimate, target):
    estimate = estimate.float().reshape(-1)
    target = target.float().reshape(-1)
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    scale = torch.sum(estimate * target) / (torch.sum(target * target) + 1e-12)
    desired = scale * target
    return 10.0 * torch.log10(
        (torch.sum(desired * desired) + 1e-12)
        / (torch.sum((estimate - desired) * (estimate - desired)) + 1e-12)
    )


def _erle_db(reference_echo, output):
    return float(10.0 * torch.log10(
        (torch.sum(reference_echo * reference_echo) + 1e-12)
        / (torch.sum(output * output) + 1e-12)
    ))


def _linear_stage_erle_db(pipeline, mic_audio, far_audio, echo_audio):
    """
    ERLE of the LINEAR stage only, before the post-filter, measured in the time domain.

    The post-filter can mask a broken adaptive filter, so gating on the end-to-end number
    alone hides regressions in the part that actually identifies the echo path.

    The residual is reconstructed with the pipeline's own iSTFT before the ratio is taken.
    An STFT-domain ratio of the same residual reads ~19 dB *pessimistically* here, because
    the analysis is window-normalised while the synthesis restores window-sum gain: the
    ratio then mixes a genuine suppression figure with the analysis/synthesis normalisation
    and cannot rank configurations. Measuring in the domain the listener hears keeps this
    number comparable with the pipeline-level ERLE guards.
    """
    model = pipeline.model
    stft = pipeline.custom_stft
    freq_bins, frames = model.freq_bins, model.signal_length

    def analyze(signal):
        signal = signal.float()
        signal = signal - signal.mean(dim=2, keepdim=True)
        return stft(signal).view(1, 2, freq_bins, frames)

    mix_comp, far_comp = analyze(mic_audio), analyze(far_audio)
    if model.auto_delay_align:
        segment_lags = model.bulk_delay_align(mix_comp, far_comp)
    else:
        segment_lags = [model.default_lag] * model.num_segments

    echo_est, _, _ = model.linear_stage(mix_comp, far_comp)
    residual = mix_comp - echo_est
    reconstructed = model.custom_istft.forward_packed(
        residual.reshape(1, 2 * freq_bins, frames))
    reference = echo_audio.float()
    reference = reference - reference.mean(dim=2, keepdim=True)
    return _erle_db(reference, reconstructed.reshape_as(reference)), [int(lag) for lag in segment_lags]


def _validate_delay_confidence(pipeline, far, near):
    """Adversarial checks for unexcited and periodic lag surfaces."""
    model = pipeline.model
    stft = pipeline.custom_stft

    def analyze(signal):
        signal = signal.float()
        signal = signal - signal.mean(dim=2, keepdim=True)
        return stft(signal).view(1, 2, model.freq_bins, model.signal_length)

    silence_lags = model.bulk_delay_align(analyze(near), analyze(torch.zeros_like(far)))
    silence_lags = [int(lag) for lag in silence_lags]
    if any(lag != 0 for lag in silence_lags):
        raise RuntimeError(f"Unexcited delay tracker must retain zero lag: {silence_lags}")

    time = torch.arange(far.shape[-1], dtype=torch.float32) / IN_SAMPLE_RATE
    tone_far = (0.07 * torch.sin(2.0 * torch.pi * 440.0 * time)).reshape(1, 1, -1)
    tone_echo = torch.zeros_like(tone_far)
    tone_delay = int(round(0.160 * IN_SAMPLE_RATE))
    tone_echo[..., tone_delay:] = 0.7 * tone_far[..., :-tone_delay]
    tone_lags = model.bulk_delay_align(analyze(tone_echo), analyze(tone_far))
    tone_lags = [int(lag) for lag in tone_lags]
    if any(lag != tone_lags[0] for lag in tone_lags[1:]):
        raise RuntimeError(f"Periodic render caused false delay switches: {tone_lags}")
    print(f"Delay confidence validation: silence={silence_lags}, periodic={tone_lags}")


def _validate_aec_behavior(pipeline, input_dtype, output_dtype):
    far = _speech_like_validation_signal(EXPORT_AUDIO_LENGTH, seed=181, rms=0.07)
    near = _speech_like_validation_signal(EXPORT_AUDIO_LENGTH, seed=907, rms=0.04)
    echo = _apply_echo_path(far, ((0.1600, 0.68), (0.1673, -0.21), (0.1818, 0.14),
                                  (0.2057, -0.08), (0.2404, 0.05)))

    far_input = _to_validation_input(far, input_dtype)
    near_input = _to_validation_input(near, input_dtype)
    echo_input = _to_validation_input(echo, input_dtype)
    double_talk_input = _to_validation_input(echo + near, input_dtype)
    silence_input = torch.zeros_like(far_input)

    far_output = _normalized_validation_audio(pipeline(echo_input, far_input), output_dtype)
    near_output = _normalized_validation_audio(pipeline(near_input, silence_input), output_dtype)
    double_talk_output = _normalized_validation_audio(
        pipeline(double_talk_input, far_input), output_dtype)

    normalized_echo = _resample_validation_target(_normalized_validation_audio(echo_input, input_dtype))
    normalized_near = _resample_validation_target(_normalized_validation_audio(near_input, input_dtype))
    normalized_mix = _resample_validation_target(
        _normalized_validation_audio(double_talk_input, input_dtype))

    metrics = {
        "synthetic_erle_db": _erle_db(normalized_echo, far_output),
        "synthetic_near_sisdr_db": float(_si_sdr_db(near_output, normalized_near)),
        "synthetic_double_talk_gain_db": float(
            _si_sdr_db(double_talk_output, normalized_near)
            - _si_sdr_db(normalized_mix, normalized_near)),
        "synthetic_double_talk_keep_db": float(_si_sdr_db(double_talk_output, normalized_near)),
    }
    if AUTO_DELAY_ALIGN:
        _validate_delay_confidence(pipeline, far, near)

    # Keep the historical 74.9 ms decay and separately exercise a tail beyond the
    # 320 ms linear filter. These assertions guard the intended fixture distinction.
    short_path = _short_exponential_echo_path(0.160, 0.600)
    short_span_seconds = short_path[-1][0] - short_path[0][0]
    linear_span_seconds = BLOCK_TAPS * HOP_LENGTH / MODEL_SAMPLE_RATE
    if short_span_seconds >= linear_span_seconds:
        raise RuntimeError("Short-tail fixture unexpectedly exceeds the linear filter span")
    short_reverb_echo = _apply_echo_path(far, short_path)
    short_reverb_input = _to_validation_input(short_reverb_echo, input_dtype)
    short_reverb_output = _normalized_validation_audio(
        pipeline(short_reverb_input, far_input), output_dtype)
    metrics["synthetic_short_reverb_erle_db"] = _erle_db(
        _resample_validation_target(
            _normalized_validation_audio(short_reverb_input, input_dtype)),
        short_reverb_output)

    long_path = _long_reverb_impulse_response(0.160, 0.600)
    first_path_sample = int(round(0.160 * IN_SAMPLE_RATE))
    long_span_seconds = (long_path.numel() - 1 - first_path_sample) / IN_SAMPLE_RATE
    if long_span_seconds <= linear_span_seconds:
        raise RuntimeError("Long-tail fixture does not exceed the linear filter span")
    long_reverb_echo = _apply_fir_echo_path(far, long_path)
    long_reverb_input = _to_validation_input(long_reverb_echo, input_dtype)
    long_reverb_output = _normalized_validation_audio(
        pipeline(long_reverb_input, far_input), output_dtype)
    metrics["synthetic_long_reverb_erle_db"] = _erle_db(
        _resample_validation_target(
            _normalized_validation_audio(long_reverb_input, input_dtype)),
        long_reverb_output)

    # Echo-path gain step at constant delay: invisible to the delay tracker, so it is
    # the case that regresses if segment adaptation is gated on a delay move alone.
    gain_step_echo = _apply_echo_path(far, ((0.1600, 0.68), (0.1673, -0.21),
                                            (0.1818, 0.14), (0.2057, -0.08))).clone()
    gain_step_echo[..., EXPORT_AUDIO_LENGTH // 2:] *= 0.25
    gain_step_input = _to_validation_input(gain_step_echo, input_dtype)
    gain_step_output = _normalized_validation_audio(
        pipeline(gain_step_input, far_input), output_dtype)
    metrics["synthetic_gain_step_erle_db"] = _erle_db(
        _resample_validation_target(
            _normalized_validation_audio(gain_step_input, input_dtype)),
        gain_step_output)

    # Linear-stage ERLE, measured before the post-filter, on both a static and a
    # mid-block-moving echo path.
    linear_erle, static_lags = _linear_stage_erle_db(pipeline, echo, far, echo)
    metrics["linear_erle_db"] = linear_erle

    if AUTO_DELAY_ALIGN:
        split = EXPORT_AUDIO_LENGTH // 2
        near_path = _apply_echo_path(far, ((0.1600, 0.68), (0.1673, -0.21), (0.1818, 0.14)))
        far_path = _apply_echo_path(far, ((0.2400, 0.68), (0.2473, -0.21), (0.2618, 0.14)))
        moving_echo = torch.cat([near_path[..., :split], far_path[..., split:]], dim=-1)
        moving_input = _to_validation_input(moving_echo, input_dtype)
        moving_output = _normalized_validation_audio(pipeline(moving_input, far_input), output_dtype)
        metrics["synthetic_dynamic_delay_erle_db"] = _erle_db(
            _resample_validation_target(_normalized_validation_audio(moving_input, input_dtype)),
            moving_output)
        moving_linear, moving_lags = _linear_stage_erle_db(pipeline, moving_echo, far, moving_echo)
        metrics["linear_dynamic_delay_erle_db"] = moving_linear
        print(f"Delay tracking: static lags={static_lags} frames, "
              f"moving-path lags={moving_lags} frames (truth 16 -> 24)")

    print(
        "AEC behavior validation: "
        f"ERLE={metrics['synthetic_erle_db']:.2f} dB, "
        f"short_reverb_ERLE={metrics['synthetic_short_reverb_erle_db']:.2f} dB, "
        f"long_reverb_ERLE={metrics['synthetic_long_reverb_erle_db']:.2f} dB, "
        f"gain_step_ERLE={metrics['synthetic_gain_step_erle_db']:.2f} dB, "
        f"linear_ERLE={metrics['linear_erle_db']:.2f} dB, "
        f"near_SI-SDR={metrics['synthetic_near_sisdr_db']:.2f} dB, "
        f"double_talk_gain={metrics['synthetic_double_talk_gain_db']:.2f} dB, "
        f"double_talk_keep={metrics['synthetic_double_talk_keep_db']:.2f} dB"
        + (f", dynamic_ERLE={metrics['synthetic_dynamic_delay_erle_db']:.2f} dB"
           f", linear_dynamic_ERLE={metrics['linear_dynamic_delay_erle_db']:.2f} dB"
           if AUTO_DELAY_ALIGN else "")
    )

    thresholds = [
        ("synthetic_erle_db", MIN_SYNTHETIC_ERLE_DB, "Synthetic ERLE"),
        ("synthetic_short_reverb_erle_db", MIN_REVERB_ERLE_DB, "Short-reverberation ERLE"),
        ("synthetic_long_reverb_erle_db", MIN_LONG_REVERB_ERLE_DB, "Long-reverberation ERLE"),
        ("linear_erle_db", MIN_LINEAR_ERLE_DB, "Linear-stage ERLE"),
        ("synthetic_near_sisdr_db", MIN_NEAR_SISDR_DB, "Near-end SI-SDR"),
        ("synthetic_double_talk_gain_db", MIN_DOUBLE_TALK_GAIN_DB, "Double-talk SI-SDR improvement"),
        ("synthetic_double_talk_keep_db", MIN_DOUBLE_TALK_KEEP_DB, "Double-talk near-end preservation"),
        ("synthetic_gain_step_erle_db", MIN_GAIN_STEP_ERLE_DB, "Echo-path gain-step ERLE"),
    ]
    if AUTO_DELAY_ALIGN:
        thresholds.append(
            ("synthetic_dynamic_delay_erle_db", MIN_DYNAMIC_ERLE_DB, "Dynamic-delay ERLE"))
        thresholds.append(
            ("linear_dynamic_delay_erle_db", MIN_LINEAR_DYNAMIC_ERLE_DB,
             "Linear dynamic-delay ERLE"))
    for key, floor, label in thresholds:
        if metrics[key] < floor:
            message = f"{label} regressed below {floor:.1f} dB: {metrics[key]:.2f} dB"
            if STATIC_SIGNAL_LENGTH >= GUARD_CALIBRATION_FRAMES:
                raise RuntimeError(message)
            print(f"Warning: {message} (expected for blocks shorter than "
                  f"{GUARD_CALIBRATION_FRAMES} frames; quality improves with INPUT_AUDIO_LENGTH)")
    return metrics


def _export_to_memory(model, near_end_audio, far_end_audio):
    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        (near_end_audio, far_end_audio),
        buffer,
        input_names=['near_end_audio', 'far_end_audio'],
        output_names=['aec_audio'],
        dynamic_axes=None,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    return onnx.load_model_from_string(buffer.getvalue())


def _validate_graph_contract(model, input_dtype, output_dtype):
    expected_inputs = ["near_end_audio", "far_end_audio"]
    expected_outputs = ["aec_audio"]
    if [value.name for value in model.graph.input] != expected_inputs:
        raise RuntimeError("RTAEC input names/order changed during export")
    if [value.name for value in model.graph.output] != expected_outputs:
        raise RuntimeError("RTAEC output names/order changed during export")
    expected_input_shape = (1, 1, EXPORT_AUDIO_LENGTH)
    expected_output_shape = (1, 1, OUTPUT_AUDIO_LENGTH)
    for value in model.graph.input:
        if _static_shape(value) != expected_input_shape:
            raise RuntimeError(f"Unexpected RTAEC input shape for {value.name}: {_static_shape(value)}")
        if value.type.tensor_type.elem_type != _torch_dtype_to_onnx(input_dtype):
            raise RuntimeError(f"Unexpected RTAEC input dtype for {value.name}")
    output = model.graph.output[0]
    if _static_shape(output) != expected_output_shape:
        raise RuntimeError(f"Unexpected RTAEC output shape: {_static_shape(output)}")
    if output.type.tensor_type.elem_type != _torch_dtype_to_onnx(output_dtype):
        raise RuntimeError("Unexpected RTAEC output dtype")


def _validate_exported_model(model, pipeline, near_end_audio, far_end_audio, input_dtype, output_dtype):
    onnx.checker.check_model(model)
    _validate_graph_contract(model, input_dtype, output_dtype)
    with torch.inference_mode():
        torch_output = pipeline(near_end_audio, far_end_audio).detach().cpu().numpy()
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    ort_output = session.run(
        ["aec_audio"],
        {
            "near_end_audio": near_end_audio.cpu().numpy(),
            "far_end_audio": far_end_audio.cpu().numpy(),
        },
    )[0]
    if np.issubdtype(torch_output.dtype, np.integer):
        if not np.array_equal(torch_output, ort_output):
            raise RuntimeError("Integer ONNX output is not exactly equal to PyTorch output")
        max_abs = 0.0
        mean_abs = 0.0
    else:
        difference = np.abs(torch_output.astype(np.float64) - ort_output.astype(np.float64))
        max_abs = float(np.max(difference))
        mean_abs = float(np.mean(difference))
        # Tolerance tracks the output magnitude: float32 accumulation error grows with
        # signal scale, and the CG solve accumulates over K taps by T frames per bin.
        peak = float(np.max(np.abs(torch_output.astype(np.float64)))) + 1e-12
        resample_factor = 100.0 if (IN_SAMPLE_RATE != MODEL_SAMPLE_RATE
                                    or OUT_SAMPLE_RATE != MODEL_SAMPLE_RATE) else 1.0
        validation_atol = 2e-4 * max(1.0, peak) * resample_factor
        if max_abs > validation_atol:
            raise RuntimeError(
                f"ONNX/PyTorch mismatch: max_abs={max_abs:.8g}, mean_abs={mean_abs:.8g}, "
                f"atol={validation_atol:.8g}, output_peak={peak:.8g}"
            )
    print(f"ONNX validation: max_abs={max_abs:.8g}, mean_abs={mean_abs:.8g}, output_shape={ort_output.shape}")


def _embed_metadata(model, metadata):
    properties = {prop.key: prop for prop in model.metadata_props}
    for key, value in build_model_metadata(metadata).items():
        if key in properties:
            properties[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)


def _atomic_write_model(model, model_path):
    destination = Path(model_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = model.SerializeToString()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    for stale_path in (
        destination.with_name(destination.stem + "_Metadata.onnx"),
        Path(str(destination) + ".data"),
    ):
        if stale_path.exists():
            stale_path.unlink()


def _run_inference_demo():
    export_dir = Path(onnx_model_A).expanduser().resolve().parent
    inference_script = Path(__file__).resolve().with_name('Inference_RTAEC_ONNX.py')
    print(f"\nStart inference demo with {inference_script.name} using: {export_dir}\n")
    subprocess.run([sys.executable, str(inference_script), str(export_dir)], check=True)


def main():
    print('Export start ...')
    with torch.inference_mode():
        custom_stft = STFT_Process(
            model_type='stft_B', n_fft=NFFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH,
            max_frames=0, window_type=WINDOW_TYPE, center_pad=True, pad_mode="constant",
            input_scale=1.0,
            packed_output=True,
        ).eval()
        custom_istft = STFT_Process(
            model_type='istft_B', n_fft=NFFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH,
            max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, center_pad=True, pad_mode="constant",
            static_norm=not DYNAMIC_AXES,
            output_scale=1.0,
            output_length=FOLD_WINDOW_LENGTH if USE_BATCH_FOLD else int(round(EXPORT_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)),
            packed_output=True,
        ).eval()

        rtaec_model = RTAECModel(
            custom_istft,
            num_lags=BLOCK_LAGS,
            num_taps=BLOCK_TAPS,
            num_pre_taps=NUM_PRE_TAPS,
            num_segments=BLOCK_SEGMENTS,
            segment_overlap=SEGMENT_OVERLAP,
            segment_crossfade=SEGMENT_CROSSFADE,
            global_cg=GLOBAL_CG_SCHEDULE,
            segment_cg=SEGMENT_CG_SCHEDULE,
            output_length=MODEL_AUDIO_LENGTH,
            auto_delay_align=AUTO_DELAY_ALIGN,
            target_lag=TARGET_TDE_LAG,
            nonlinear_basis=NONLINEAR_BASIS,
        ).float().eval()
        rtaec = RTAEC(
            rtaec_model,
            custom_stft,
            IN_SAMPLE_RATE,
            OUT_SAMPLE_RATE,
            USE_BATCH_FOLD,
            FOLD_INPUT_LENGTH,
            MODEL_AUDIO_LENGTH,
            FOLD_OUTPUT_LENGTH if USE_BATCH_FOLD else OUTPUT_AUDIO_LENGTH,
        ).eval()

        input_torch_dtype = _input_torch_dtype()
        if "int" in OUT_AUDIO_DTYPE.lower():
            output_torch_dtype = torch.int16
        elif "32" in OUT_AUDIO_DTYPE:
            output_torch_dtype = torch.float32
        else:
            output_torch_dtype = torch.float16
        behavior_metrics = _validate_aec_behavior(rtaec, input_torch_dtype, output_torch_dtype)
        near_end_audio = _validation_audio(input_torch_dtype, EXPORT_AUDIO_LENGTH, seed=1729)
        far_end_audio = _validation_audio(input_torch_dtype, EXPORT_AUDIO_LENGTH, seed=1730)
        raw_model = _export_to_memory(rtaec, near_end_audio, far_end_audio)
        _validate_exported_model(
            raw_model,
            rtaec,
            near_end_audio,
            far_end_audio,
            input_torch_dtype,
            output_torch_dtype,
        )

    model_metadata = build_audio_metadata_from_globals(
        globals(), producer=Path(__file__).name, model_name="RTAEC", task="aec", model_family="rtaec",
        max_dynamic_audio_seconds=30, normalize_audio_default=False, input_channels=1, output_channels=1,
        num_audio_inputs=2, feature_kind="stft_block_wiener", center_pad=True, pad_mode="constant", extra={
            "processing_mode": "offline_block",
            "causal": False,
            "stateful": False,
            "required_input_samples": EXPORT_AUDIO_LENGTH,
            "minimum_buffering_seconds": EXPORT_AUDIO_LENGTH / IN_SAMPLE_RATE,
            "num_taps": BLOCK_TAPS,
            "num_pre_taps": NUM_PRE_TAPS,
            "num_lags": BLOCK_LAGS,
            "num_segments": BLOCK_SEGMENTS,
            "segment_overlap": SEGMENT_OVERLAP,
            "segment_crossfade": SEGMENT_CROSSFADE,
            "auto_delay_align": AUTO_DELAY_ALIGN,
            "target_tde_lag": TARGET_TDE_LAG,
            "tde_coarse_floor_ratio": TDE_COARSE_FLOOR_RATIO,
            "tde_coarse_second_ratio": TDE_COARSE_SECOND_RATIO,
            "tde_refined_floor_ratio": TDE_REFINED_FLOOR_RATIO,
            "tde_refined_second_ratio": TDE_REFINED_SECOND_RATIO,
            "tde_peak_exclusion": TDE_PEAK_EXCLUSION,
            "tde_refined_radius": TDE_REFINED_RADIUS,
            "global_cg_schedule": list(GLOBAL_CG_SCHEDULE),
            "segment_cg_schedule": list(SEGMENT_CG_SCHEDULE),
            "diag_load_db": DIAG_LOAD_DB,
            "robust_c": ROBUST_C,
            "irls_scale_mode": IRLS_SCALE_MODE,
            "dtd_solver_threshold": DTD_SOLVER_THRESHOLD,
            "dtd_res_threshold": DTD_RES_THRESHOLD,
            "dtd_hold": DTD_HOLD,
            "path_select_margin": PATH_SELECT_MARGIN,
            "path_drift_lo": PATH_DRIFT_LO,
            "path_drift_hi": PATH_DRIFT_HI,
            "path_drift_clean_lo": PATH_DRIFT_CLEAN_LO,
            "path_drift_clean_hi": PATH_DRIFT_CLEAN_HI,
            "nonlinear_basis": NONLINEAR_BASIS,
            "reverb_decay": REVERB_DECAY,
            "adaptive_reverb_decay": ADAPTIVE_REVERB_DECAY,
            "reverb_gain": REVERB_GAIN,
            "res_est_weight": RES_EST_WEIGHT,
            "res_coh_weight": RES_COH_WEIGHT,
            "complex_coherence": True,
            "res_min_gain_db": RES_MIN_GAIN_DB,
            **{key: round(value, 4) for key, value in behavior_metrics.items()},
        },
    )
    _embed_metadata(raw_model, model_metadata)
    onnx.checker.check_model(raw_model)
    _atomic_write_model(raw_model, onnx_model_A)
    print(f"Final artifact: {Path(onnx_model_A).expanduser().resolve()}")
    print('No metadata sidecar or external-data file was created.')
    print('\nExport done!')


if __name__ == '__main__':
    main()
    _run_inference_demo()
