"""
Dynamic near/far bulk-delay tracking shared by the AEC exporters.

`BulkDelayAlign` estimates one integer frame lag per tracking segment with sub-band
GCC-PHAT; it is the tracker shipped in RTAEC. `DelayWarp` applies those lags to a
far-end spectrum with per-segment shifts joined by a crossfade, for models that consume
an aligned reference instead of per-segment lags. `DynamicDelayAlign` bundles both with
parameters scaled from RTAEC's 16 kHz / 10 ms-hop tuning to another STFT grid.

Everything is static-shape tensor code inside `forward()`, so it exports as part of the
ONNX graph without Loop/Scan nodes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


TDE_SMOOTH               = 3
TDE_PRIOR_WEIGHT         = 0.35
TDE_COARSE_FLOOR_RATIO   = 1.8
TDE_COARSE_SECOND_RATIO  = 1.1
TDE_REFINED_FLOOR_RATIO  = 2.5
TDE_REFINED_SECOND_RATIO = 1.5
TDE_MIC_ACTIVITY_RATIO   = 1e-4

# RTAEC's tuning in physical units, so other STFT grids get the same behaviour.
MAX_DELAY_SECONDS        = 0.95
TDE_BAND_LO_HZ           = 62.5
TDE_BAND_HI_HZ           = 6875.0
SEGMENT_SECONDS          = 0.25
SEGMENT_OVERLAP_SECONDS  = 0.12
SEGMENT_CROSSFADE_SECONDS = 0.10
TDE_PRIOR_SIGMA_SECONDS  = 0.06
TDE_RADIUS_SECONDS       = 0.02


def _segment_bounds(signal_length, num_segments, overlap):
    """Core spans partition the block; analysis spans widen them by `overlap` on each side."""
    edges = [round(index * signal_length / num_segments) for index in range(num_segments + 1)]
    bounds = []
    for index in range(num_segments):
        core_start, core_end = edges[index], edges[index + 1]
        bounds.append((
            max(0, core_start - overlap),
            min(signal_length, core_end + overlap),
            core_start,
            core_end,
        ))
    return bounds


def _crossfade_profile(signal_length, num_segments, crossfade):
    """Partition of unity over the block: one raised ramp per segment, summing to one."""
    edges = [round(index * signal_length / num_segments) for index in range(num_segments + 1)]
    axis = torch.arange(signal_length, dtype=torch.float32)
    rows = []
    for index in range(num_segments):
        rising = (torch.clamp((axis - edges[index]) / crossfade + 0.5, 0.0, 1.0)
                  if index > 0 else torch.ones(signal_length))
        falling = (torch.clamp((edges[index + 1] - axis) / crossfade + 0.5, 0.0, 1.0)
                   if index < num_segments - 1 else torch.ones(signal_length))
        rows.append(rising * falling)
    profile = torch.stack(rows)
    return profile / profile.sum(dim=0, keepdim=True)


class BulkDelayAlign(nn.Module):
    """
    Sub-band GCC-PHAT bulk-delay tracker producing one integer lag per tracking segment.

    This module only *estimates* delay; it never warps the reference. Warping a shared
    reference mid-block makes the echo path time-varying, which a single time-invariant
    block-Wiener solve cannot represent: a one-frame step measured 3.4 dB linear ERLE
    against 22.1 dB for the correct constant lag. The solver therefore consumes these
    per-segment lags directly, giving each segment its own aligned, time-invariant solve.

    Correlation always reads real far-end left context rather than zero padding, so a
    segment starting mid-block still sees the samples its echo actually came from.

    Inputs are (batch, 2, freq_bins, signal_length) packed spectra; each batch row (for
    example a folded window) is tracked independently. Returns one (batch,) int32 lag
    tensor per segment, in frames.
    """
    def __init__(self, num_lags, target_lag, freq_bins, signal_length, lo, hi,
                 num_segments, overlap, prior_weight=TDE_PRIOR_WEIGHT, smooth=TDE_SMOOTH,
                 prior_sigma=6.0, batch=1,
                 coarse_floor_ratio=TDE_COARSE_FLOOR_RATIO,
                 coarse_second_ratio=TDE_COARSE_SECOND_RATIO,
                 refined_floor_ratio=TDE_REFINED_FLOOR_RATIO,
                 refined_second_ratio=TDE_REFINED_SECOND_RATIO,
                 peak_exclusion=2, refined_radius=2,
                 mic_activity_ratio=TDE_MIC_ACTIVITY_RATIO):
        super().__init__()
        if not 0 <= lo < hi <= freq_bins:
            raise ValueError(f"Invalid GCC-PHAT band [{lo}, {hi}) for {freq_bins} bins.")
        if num_segments < 1 or signal_length < num_segments:
            raise ValueError(f"Cannot split {signal_length} frames into {num_segments} segments.")
        self.num_lags = num_lags
        self.target_lag = target_lag
        self.freq_bins = freq_bins
        self.signal_length = signal_length
        self.lo = lo
        self.hi = hi
        self.bands = hi - lo
        self.rest_bins = freq_bins - hi
        self.batch = batch
        self.num_segments = num_segments
        self.prior_weight = prior_weight
        self.smooth = smooth
        self.coarse_floor_ratio = coarse_floor_ratio
        self.coarse_second_ratio = coarse_second_ratio
        self.refined_floor_ratio = refined_floor_ratio
        self.refined_second_ratio = refined_second_ratio
        self.peak_exclusion = peak_exclusion
        self.refined_radius = refined_radius
        self.mic_activity_ratio = mic_activity_ratio
        self.bounds = _segment_bounds(signal_length, num_segments, overlap)
        self.conf_bias = 1.8e-6
        self.register_buffer('lag_axis', torch.arange(num_lags, dtype=torch.float32).reshape(1, 1, num_lags))
        self.register_buffer('default_peak', torch.tensor([num_lags - 1], dtype=torch.int64))
        self.register_buffer('inv_prior_var', torch.tensor(1.0 / (2.0 * prior_sigma * prior_sigma),
                                                           dtype=torch.float32))

        # One 0/1 selector column per core span plus a global column, so all the
        # render-activity energies come out of a single matmul.
        selector = torch.zeros(signal_length, num_segments + 1, dtype=torch.float32)
        thresholds = torch.empty(num_segments + 1, dtype=torch.float32)
        for index, (_, _, core_start, core_end) in enumerate(self.bounds):
            selector[core_start:core_end, index] = 1.0
            thresholds[index] = 1e-5 * float(self.bands * (core_end - core_start))
        selector[:, num_segments] = 1.0
        thresholds[num_segments] = 1e-5 * float(self.bands * signal_length)
        self.register_buffer('energy_selector', selector)
        self.register_buffer('energy_thresholds', thresholds)

    def _surface(self, mic_norm, far_padded, start, end):
        """Complex lag correlations for mic[start:end], retaining real far-end left context."""
        span = end - start
        groups = self.batch * self.bands
        kernel = mic_norm[..., start:end].transpose(1, 2).reshape(2 * groups, 1, span)
        window = far_padded[..., start:end + self.num_lags - 1]
        stacked = window.transpose(0, 1).reshape(2, groups, span + self.num_lags - 1)
        product = F.conv1d(stacked, kernel, groups=groups).view(
            2, self.batch, self.bands, 2, self.num_lags)
        p0, p1 = product.split(1, dim=0)
        p00, p01 = p0.split(1, dim=3)
        p10, p11 = p1.split(1, dim=3)
        real = (p00 + p11).reshape(self.batch, self.bands, self.num_lags)
        imag = (p10 - p01).reshape(self.batch, self.bands, self.num_lags)
        return real, imag

    def _smooth_surface(self, real, imag):
        """Collapse complex band correlations into the smoothed GCC-PHAT lag surface."""
        surface = torch.sqrt(real * real + imag * imag + 1e-12).sum(dim=1, keepdim=True)
        return F.avg_pool1d(surface, self.smooth, stride=1, padding=self.smooth // 2,
                            count_include_pad=False)

    def _peak_confidence(self, surface):
        """Return the peak and continuous coarse/refined confidence masks."""
        batch = self.batch
        peak_value, peak_index = surface.max(dim=-1)
        distance = (self.lag_axis - peak_index.reshape(batch, 1, 1).float()).abs()
        second_peak = torch.where(
            distance > self.peak_exclusion, surface, torch.zeros_like(surface)
        ).amax(dim=-1)
        floor = surface.mean(dim=-1)
        coarse = ((peak_value > floor * self.coarse_floor_ratio + self.conf_bias)
                  & (peak_value > second_peak * self.coarse_second_ratio + self.conf_bias))
        refined = ((peak_value > floor * self.refined_floor_ratio + self.conf_bias)
                   & (peak_value > second_peak * self.refined_second_ratio + self.conf_bias))
        return peak_index.reshape(batch), coarse.reshape(batch), refined.reshape(batch)

    def forward(self, mix_comp, far_comp):
        batch = self.batch
        both = torch.cat([mix_comp, far_comp], dim=0)
        _, band, _ = both.split([self.lo, self.bands, self.rest_bins], dim=2)
        power = band.square().sum(dim=1, keepdim=True)
        normalized = band * torch.rsqrt(power + 1e-6)
        mic_norm, far_norm = normalized.split(batch, dim=0)
        far_padded = F.pad(far_norm, (self.num_lags - 1, 0))
        # Rows [0, batch) = mic, rows [batch, 2 * batch) = far-end energy per core span
        # (plus the global column).
        span_power = torch.matmul(power.reshape(2 * batch, -1, self.signal_length).sum(dim=1),
                                  self.energy_selector)
        mic_span, far_span = span_power.split(batch, dim=0)
        # A span whose echo has not reached the mic yet (bulk delay longer than the span
        # offset) holds render but no evidence, and its surface peak is noise.
        active_all = ((far_span > self.energy_thresholds)
                      & (mic_span > far_span * self.mic_activity_ratio)).unbind(dim=1)

        correlations = [self._surface(mic_norm, far_padded, core_start, core_end)
                        for _, _, core_start, core_end in self.bounds]
        global_real, global_imag = correlations[0]
        for real, imag in correlations[1:]:
            global_real = global_real + real
            global_imag = global_imag + imag
        global_surface = self._smooth_surface(global_real, global_imag)
        global_peak_value, global_peak = global_surface.max(dim=-1)
        global_peak = global_peak.reshape(batch)
        global_peak_value = global_peak_value.reshape(batch)
        global_floor = global_surface.mean(dim=-1).reshape(batch)
        global_coarse = global_peak_value > global_floor * self.coarse_floor_ratio + self.conf_bias
        global_active = active_all[self.num_segments]
        global_trusted = global_coarse & global_active
        previous = torch.where(global_trusted, global_peak, self.default_peak)
        has_trusted = global_trusted
        pending = previous
        pending_valid = torch.zeros_like(global_trusted)

        peaks = []
        for index, (real, imag) in enumerate(correlations):
            surface = self._smooth_surface(real, imag)
            if self.prior_weight > 0.0:
                distance = (self.lag_axis - previous.reshape(batch, 1, 1).float()).square() * self.inv_prior_var
                prior = self.prior_weight * (1.0 - torch.exp(-distance))
                surface = surface * (1.0 - has_trusted.reshape(batch, 1, 1).float() * prior)
            peak_index, coarse, refined = self._peak_confidence(surface)
            active = active_all[index]
            small_move = (peak_index - previous).abs() <= self.refined_radius
            refined_accept = refined if index == 0 else refined & small_move
            global_support = (global_trusted
                              & ((peak_index - global_peak).abs() <= self.refined_radius))
            adjacent_support = (pending_valid
                                & ((peak_index - pending).abs() <= self.refined_radius))
            accept = active & (refined_accept | (coarse & (global_support | adjacent_support)))
            previous = torch.where(accept, peak_index, previous)
            has_trusted = has_trusted | accept
            waiting = active & coarse & ~accept
            pending = torch.where(waiting, peak_index, pending)
            pending_valid = waiting
            peaks.append(previous)

        # Convert argmax position to a delay in frames: the surface is built in reversed
        # convolution order, so lag 0 sits at index num_lags - 1.
        lags = [self.num_lags - 1 - peak.to(torch.int32) for peak in peaks]
        if self.target_lag != 0:
            lags = [torch.clamp(lag + self.target_lag, 0, self.num_lags - 1) for lag in lags]
        return lags


class DelayWarp(nn.Module):
    """
    Shift a far-end spectrum by the per-segment lags of `BulkDelayAlign`.

    Each segment reads the reference at its own constant lag, and neighbouring segments
    are joined by a partition-of-unity crossfade, so a delay move becomes a short blend
    rather than a per-frame index ramp. `residual_delay` frames of the natural delay are
    kept, so the aligned reference still leads the echo slightly instead of being
    advanced past it; it never advances an unaligned (lag 0) reference.
    """
    def __init__(self, num_lags, signal_length, num_segments, crossfade, residual_delay=0, batch=1):
        super().__init__()
        self.pad_left = num_lags - 1
        self.signal_length = signal_length
        self.residual_delay = residual_delay
        self.batch = batch
        profile = _crossfade_profile(signal_length, num_segments, crossfade)
        self.bounds = []
        self.blend_names = []
        # Equal-valued buffers become Identity aliases in the exported graph, so share them.
        blend_lookup = {}
        for index in range(num_segments):
            support = torch.nonzero(profile[index] > 0.0).reshape(-1)
            start, end = int(support[0]), int(support[-1]) + 1
            self.bounds.append((start, end))
            self.register_buffer(f'window_{index}', torch.arange(
                start + self.pad_left, end + self.pad_left,
                dtype=torch.int32 if batch == 1 else torch.int64))
            blend = profile[index, start:end].reshape(1, 1, end - start)
            key = tuple(blend.reshape(-1).tolist())
            if key not in blend_lookup:
                blend_lookup[key] = f'blend_{index}'
                self.register_buffer(blend_lookup[key], blend)
            self.blend_names.append(blend_lookup[key])

    def forward(self, far_comp, lags):
        shape = far_comp.shape
        frames = self.signal_length
        rows = shape[1] * shape[2]
        padded = F.pad(far_comp, (self.pad_left, 0)).reshape(self.batch, rows, frames + self.pad_left)
        total = None
        for index, (start, end) in enumerate(self.bounds):
            shift = lags[index]
            if self.residual_delay > 0:
                shift = torch.clamp(shift - self.residual_delay, min=0)
            window = getattr(self, f'window_{index}')
            if self.batch == 1:
                line = torch.index_select(padded.reshape(rows, -1), -1,
                                          window - shift.reshape(()).to(window.dtype))
                line = line.reshape(1, rows, end - start)
            else:
                index_map = (window.reshape(1, 1, -1)
                             - shift.to(window.dtype).reshape(self.batch, 1, 1)).expand(
                    self.batch, rows, end - start)
                line = torch.gather(padded, 2, index_map)
            part = F.pad(line * getattr(self, self.blend_names[index]), (start, frames - end))
            total = part if total is None else total + part
        return total.reshape(shape)


class DynamicDelayAlign(nn.Module):
    """
    Estimate per-segment near/far bulk delay and return the aligned far-end spectrum.

    The tracker settings are RTAEC's, converted from physical units to this STFT grid.
    """
    def __init__(self, nfft, hop_length, sample_rate, signal_length, batch=1,
                 residual_delay=0, num_lags=None, estimate_only=False, crossfade=None,
                 **tracker_overrides):
        super().__init__()
        frame_seconds = hop_length / sample_rate
        bin_hz = sample_rate / nfft
        freq_bins = nfft // 2 + 1
        if num_lags is None:
            num_lags = int(MAX_DELAY_SECONDS / frame_seconds + 1e-9) + 1
        radius = max(1, round(TDE_RADIUS_SECONDS / frame_seconds))
        tracker_args = dict(
            target_lag=0,
            lo=max(2, round(TDE_BAND_LO_HZ / bin_hz)),
            hi=min(freq_bins, round(TDE_BAND_HI_HZ / bin_hz)),
            num_segments=max(1, min(8, round(signal_length * frame_seconds / SEGMENT_SECONDS))),
            overlap=max(1, round(SEGMENT_OVERLAP_SECONDS / frame_seconds)),
            prior_sigma=TDE_PRIOR_SIGMA_SECONDS / frame_seconds,
            peak_exclusion=radius,
            refined_radius=radius,
        )
        tracker_args.update(tracker_overrides)
        self.num_lags = num_lags
        self.num_segments = tracker_args['num_segments']
        self.estimate_only = estimate_only
        self.tracker = BulkDelayAlign(
            num_lags=num_lags, freq_bins=freq_bins, signal_length=signal_length,
            batch=batch, **tracker_args,
        )
        # Estimate-only callers (RTAEC) consume per-segment lags instead of a warped reference.
        if not estimate_only:
            self.warp = DelayWarp(
                num_lags=num_lags,
                signal_length=signal_length,
                num_segments=self.num_segments,
                crossfade=crossfade or max(2, round(SEGMENT_CROSSFADE_SECONDS / frame_seconds)),
                residual_delay=residual_delay,
                batch=batch,
            )

    def forward(self, mix_comp, far_comp):
        lags = self.tracker(mix_comp, far_comp)
        if self.estimate_only:
            return lags
        return self.warp(far_comp, lags)
