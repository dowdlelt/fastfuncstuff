"""Sonification: the same drives the swarm reads, made into gentle sound.

* Each ICA family is a voice on a minor-pentatonic scale -- any combination of
  notes is consonant, so whatever the data does, it never clashes. Pitch is
  the family's height in the head (superior sings higher), pan is its
  left/right centroid, and loudness is its live time course weighted by how
  salient its members are, so noise families murmur and real networks sing.
* A low drone (root and fifth) whose brightness follows the global signal.
* DVARS spikes -- the frames where the whole brain changes at once, usually
  motion -- are a soft filtered-noise swell.
* A small Schroeder reverb so it all sits in a room instead of a speaker.

Synthesis is block-based numpy; parameters ramp across each block so nothing
clicks when the data jumps.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from .features import Features

PENTATONIC = (0, 3, 5, 7, 10)  # minor pentatonic, semitones above the root


class Sonifier:
    def __init__(
        self, feats: Features, sample_rate: int = 44100, root_hz: float = 110.0, seed: int = 0
    ):
        self.sr = sample_rate
        self.f = feats
        fam_tcs = feats.family_tcs.float().cpu().numpy()  # (T, F)
        self.fam_tcs = fam_tcs
        n_fam = fam_tcs.shape[1]
        centre = feats.family_center.float().cpu().numpy()  # (F, 3) RAS, ~unit
        fam = feats.family.cpu().numpy()
        sal = feats.salience.float().cpu().numpy()
        weight = np.array([sal[fam == k].mean() if np.any(fam == k) else 0.0 for k in range(n_fam)])
        self.voice_gain = (weight / max(weight.max(), 1e-6)) ** 2
        # Height ranks to scale degrees across two and a half octaves.
        order = np.argsort(np.argsort(centre[:, 2]))
        degrees = np.round(order / max(n_fam - 1, 1) * 12).astype(int)
        semis = np.array([12 * (d // 5) + PENTATONIC[d % 5] for d in degrees])
        self.freqs = root_hz * 2.0 * 2.0 ** (semis / 12.0)  # voices start an octave above the drone
        # Most families are bilateral, so centroids sit near the midline;
        # stretch the asymmetry so a lateralised network is audibly off-centre.
        self.pan = np.clip(centre[:, 0] * 4.0, -1, 1)
        self.root = root_hz
        self.gs = feats.global_signal.float().cpu().numpy()
        self.dv = feats.dvars.float().cpu().numpy()
        self.phase = np.zeros(n_fam)
        self.drone_phase = np.zeros(4)
        self.amp = np.zeros(n_fam)
        self.drone_bright = 0.0
        self.noise_amp = 0.0
        self.noise_state = np.zeros(2)
        self.rng = np.random.default_rng(seed)
        self.volume = 0.6
        self.voices = 1.0
        self.drone = 0.5
        self.whoosh = 0.6
        self.reverb = 0.35
        self.gate = 1.0  # 0 fades everything out (pause) without a click
        self._gate_now = 1.0
        self._reverb = _Reverb(sample_rate)

    def _at(self, arr: np.ndarray, t: float) -> np.ndarray:
        n = arr.shape[0]
        i = int(math.floor(t)) % n
        j = (i + 1) % n
        u = t - math.floor(t)
        return arr[i] * (1 - u) + arr[j] * u

    def render(self, n: int, t1: float) -> np.ndarray:
        """(n, 2) float32 stereo ending at data time t1 (TRs); parameters ramp from the last call."""
        out = np.zeros((n, 2), dtype=np.float64)
        ramp = np.linspace(0, 1, n, endpoint=False)[:, None]
        sr = self.sr

        # Voices.
        tc = self._at(self.fam_tcs, t1)
        target = np.clip(tc - 0.3, 0, 3) / 3.0 * self.voice_gain * self.voices
        amp = self.amp[None, :] * (1 - ramp) + target[None, :] * ramp  # (n, F)
        inc = 2 * math.pi * self.freqs / sr
        ph = self.phase[None, :] + inc[None, :] * np.arange(1, n + 1)[:, None]
        self.phase = ph[-1] % (2 * math.pi)
        # Brighter timbre as a voice gets louder.
        tone = np.sin(ph) + (0.25 * amp) * np.sin(2 * ph) + (0.1 * amp) * np.sin(3 * ph)
        voice = tone * amp
        left = np.sqrt(0.5 * (1 - self.pan))
        right = np.sqrt(0.5 * (1 + self.pan))
        out[:, 0] += voice @ left * 0.22
        out[:, 1] += voice @ right * 0.22
        self.amp = target

        # Drone: root, fifth and their octaves; global signal opens it up.
        g = float(self._at(self.gs, t1))
        bright_t = 0.5 + 0.5 * math.tanh(g * 0.6)
        bright = self.drone_bright * (1 - ramp[:, 0]) + bright_t * ramp[:, 0]
        self.drone_bright = bright_t
        dfreq = np.array([self.root / 2, self.root * 0.75, self.root, self.root * 1.5])
        dgain = np.array([1.0, 0.6, 0.35, 0.2])
        dph = (
            self.drone_phase[None, :]
            + (2 * math.pi * dfreq / sr)[None, :] * np.arange(1, n + 1)[:, None]
        )
        self.drone_phase = dph[-1] % (2 * math.pi)
        layers = np.sin(dph) * dgain
        layers[:, 2:] *= bright[:, None]
        drone = layers.sum(1) * 0.05 * self.drone
        out += drone[:, None]

        # Whoosh on DVARS spikes.
        d = float(self._at(self.dv, t1))
        n_t = max(0.0, d - 2.5) * 0.25 * self.whoosh
        n_amp = self.noise_amp * (1 - ramp[:, 0]) + n_t * ramp[:, 0]
        self.noise_amp = n_amp[-1] * 0.92
        noise = self.rng.standard_normal((n, 2))
        a = 0.97  # one-pole lowpass: a soft swell, not a hiss
        for ch in range(2):
            y, zf = lfilter([1 - a], [1, -a], noise[:, ch], zi=self.noise_state[ch : ch + 1])
            self.noise_state[ch] = zf[0]
            out[:, ch] += y * n_amp * 6.0

        gate = self._gate_now * (1 - ramp[:, 0]) + self.gate * ramp[:, 0]
        self._gate_now = self.gate
        out = out * gate[:, None]
        out = self._reverb.process(out, self.reverb)
        out = np.tanh(out * self.volume * 1.5)
        return out.astype(np.float32)


class _Reverb:
    """Freeverb-style: four combs and two allpasses per channel.

    Every delay is longer than the processing sub-block, so each sub-block is a
    vectorised read from history -- no per-sample Python loop.
    """

    COMBS = (1116, 1188, 1277, 1356)
    ALLPASS = (556, 441)
    BLOCK = 400

    def __init__(self, sr: int):
        s = sr / 44100.0
        self.combs = [[_Delay(int(d * s) + 23 * ch) for d in self.COMBS] for ch in range(2)]
        self.allp = [[_Delay(int(d * s) + 11 * ch) for d in self.ALLPASS] for ch in range(2)]

    def process(self, x: np.ndarray, mix: float) -> np.ndarray:
        if mix <= 0:
            return x
        out = np.empty_like(x)
        for a in range(0, x.shape[0], self.BLOCK):
            blk = x[a : a + self.BLOCK]
            for ch in range(2):
                xin = blk[:, ch] * 0.15
                wet = np.zeros_like(xin)
                for c in self.combs[ch]:
                    wet += c.comb(xin, 0.84)
                for ap in self.allp[ch]:
                    wet = ap.allpass(wet, 0.5)
                out[a : a + self.BLOCK, ch] = blk[:, ch] * (1 - mix * 0.5) + wet * mix
        return out


class _Delay:
    def __init__(self, n: int):
        self.buf = np.zeros(n)
        self.pos = 0

    def _read(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        idx = (self.pos + np.arange(m)) % self.buf.size
        return self.buf[idx], idx

    def comb(self, x: np.ndarray, fb: float) -> np.ndarray:
        past, idx = self._read(x.size)
        y = x + fb * past
        self.buf[idx] = y
        self.pos = (self.pos + x.size) % self.buf.size
        return past

    def allpass(self, x: np.ndarray, g: float) -> np.ndarray:
        past, idx = self._read(x.size)
        y = -x + past
        self.buf[idx] = x + g * past
        self.pos = (self.pos + x.size) % self.buf.size
        return y


def render_track(
    feats: Features, n_frames: int, fps: float, speed: float, sample_rate: int = 44100
) -> np.ndarray:
    """Offline audio for a movie: (samples, 2) float32 matching ``n_frames``."""
    son = Sonifier(feats, sample_rate)
    per = sample_rate / fps
    chunks = []
    t = 0.0
    made = 0
    period = feats.n_time - 1
    for i in range(n_frames):
        want = int(round((i + 1) * per)) - made
        t1 = (t + speed / fps) % period
        chunks.append(son.render(want, t1))
        made += want
        t = t1
    return np.concatenate(chunks, 0)
