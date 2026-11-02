"""The ffs_dataviz window: a canvas, a hidden control panel, and live sound.

The canvas shows the swarm's low-resolution buffer scaled up with
nearest-neighbour sampling, so every simulated pixel stays a crisp square.
Everything tunable lives in a side panel that starts hidden (Tab or m).
"""

from __future__ import annotations

import math
import signal
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .features import Features
from .sound import Sonifier
from .swarm import PARAMS, Swarm

RESOLUTIONS = ((320, 180), (480, 270), (640, 360), (960, 540), (1280, 720), (1920, 1080))

KEYS_HELP = """\
space  play / pause        ← →  step 5 TRs (shift: 25)
tab m  controls            h    timeline
f      fullscreen          s    sound on / off
c      next colour source  l    anatomy ↔ function (animated)
r      reset camera        [ ]  slower / faster
drag   orbit               wheel  zoom
q esc  quit"""


class Canvas(QtWidgets.QWidget):
    def __init__(self, parent: Viz):
        super().__init__(parent)
        self.viz = parent
        self.image: QtGui.QImage | None = None
        self._buf: np.ndarray | None = None
        self.show_hud = True
        self._drag: QtCore.QPointF | None = None
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 180)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    def focusNextPrevChild(self, _next: bool) -> bool:
        # Tab toggles the controls; it must reach keyPressEvent, not move focus.
        return False

    def set_frame(self, rgb: np.ndarray) -> None:
        self._buf = np.ascontiguousarray(rgb)
        h, w, _ = self._buf.shape
        self.image = QtGui.QImage(self._buf.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        self.update()

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(3, 4, 8))  # matches the swarm's background
        if self.image is not None:
            iw, ih = self.image.width(), self.image.height()
            s = min(self.width() / iw, self.height() / ih)
            w, h = int(iw * s), int(ih * s)
            target = QtCore.QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)
            p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
            p.drawImage(target, self.image)
        if self.show_hud:
            self._draw_hud(p)
        p.end()

    def _draw_hud(self, p: QtGui.QPainter) -> None:
        viz = self.viz
        f = viz.feats
        w, h = self.width(), self.height()
        band = 34
        y0 = h - band - 10
        x0, x1 = 16, w - 16
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        gs = viz.gs_np
        dv = viz.dv_np
        n = gs.size
        xs = np.linspace(x0, x1, n)
        # Global signal as a faint line, DVARS spikes as ticks: the whole-brain
        # story of the run at a glance, with the playhead on it.
        path = QtGui.QPainterPath()
        mid = y0 + band / 2
        for i in range(n):
            y = mid - np.clip(gs[i], -3, 3) / 3 * (band / 2 - 2)
            if i == 0:
                path.moveTo(xs[i], y)
            else:
                path.lineTo(xs[i], y)
        p.setPen(QtGui.QPen(QtGui.QColor(140, 160, 220, 90), 1))
        p.drawPath(path)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 200, 120, 120), 1))
        for i in np.nonzero(dv > 3.0)[0]:
            p.drawLine(QtCore.QPointF(xs[i], y0 + band), QtCore.QPointF(xs[i], y0 + band - 6))
        for start in f.run_starts[1:]:
            p.setPen(QtGui.QPen(QtGui.QColor(120, 130, 160, 80), 1, QtCore.Qt.PenStyle.DashLine))
            p.drawLine(QtCore.QPointF(xs[start], y0), QtCore.QPointF(xs[start], y0 + band))
        px = x0 + (x1 - x0) * viz.swarm.t / max(n - 1, 1)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 170), 1))
        p.drawLine(QtCore.QPointF(px, y0 - 2), QtCore.QPointF(px, y0 + band + 2))
        p.setPen(QtGui.QColor(170, 180, 210, 150))
        font = p.font()
        font.setPointSizeF(8.5)
        p.setFont(font)
        t_s = viz.swarm.t * f.tr
        status = "" if viz.playing else "  ❚❚"
        sound = "  ♪" if viz.sound_on else ""
        p.drawText(
            x0,
            y0 - 6,
            f"{t_s:6.1f} s   TR {viz.swarm.t:5.1f}/{n}   ×{viz.swarm.p['speed']:.1f}{status}{sound}",
        )

    # camera interaction
    def mousePressEvent(self, e) -> None:
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            hud_top = self.height() - 50
            if self.show_hud and e.position().y() > hud_top:
                self._seek_to(e.position().x())
                return
            self._drag = e.position()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

    def _seek_to(self, x: float) -> None:
        frac = (x - 16) / max(1, self.width() - 32)
        self.viz.seek(np.clip(frac, 0, 1) * (self.viz.feats.n_time - 1))

    def mouseMoveEvent(self, e) -> None:
        if self._drag is None:
            if e.buttons() & QtCore.Qt.MouseButton.LeftButton and self.show_hud:
                self._seek_to(e.position().x())
            return
        d = e.position() - self._drag
        self._drag = e.position()
        sw = self.viz.swarm
        self.viz.set_param("yaw", sw.p["yaw"] + d.x() * 0.008)
        self.viz.set_param("pitch", float(np.clip(sw.p["pitch"] + d.y() * 0.008, -1.5, 1.5)))

    def mouseReleaseEvent(self, _e) -> None:
        self._drag = None
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, e) -> None:
        z = self.viz.swarm.p["zoom"] * (1.1 ** (e.angleDelta().y() / 120))
        self.viz.set_param("zoom", float(np.clip(z, 0.3, 4.0)))


class Controls(QtWidgets.QScrollArea):
    """Every swarm parameter as a slider (or a combo for a choice), grouped."""

    STEPS = 1000

    def __init__(self, viz: Viz):
        super().__init__()
        self.viz = viz
        self.setWidgetResizable(True)
        self.setMinimumWidth(300)
        body = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        self.widgets: dict[str, tuple[QtWidgets.QWidget, QtWidgets.QLabel | None]] = {}
        groups: dict[str, QtWidgets.QFormLayout] = {}
        for prm in PARAMS:
            if prm.group not in groups:
                box = QtWidgets.QGroupBox(prm.group)
                form = QtWidgets.QFormLayout(box)
                form.setContentsMargins(6, 4, 6, 4)
                groups[prm.group] = form
                lay.addWidget(box)
            form = groups[prm.group]
            if prm.choices:
                combo = QtWidgets.QComboBox()
                combo.addItems(list(prm.choices))
                combo.setCurrentIndex(int(viz.swarm.p[prm.name]))
                combo.setToolTip(prm.help)
                combo.currentIndexChanged.connect(lambda i, n=prm.name: viz.set_param(n, float(i)))
                form.addRow(prm.name, combo)
                self.widgets[prm.name] = (combo, None)
                continue
            row = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            sl.setRange(0, self.STEPS)
            sl.setToolTip(prm.help)
            val = QtWidgets.QLabel()
            val.setMinimumWidth(44)
            hl.addWidget(sl, 1)
            hl.addWidget(val)
            sl.valueChanged.connect(lambda v, n=prm.name: self._slid(n, v))
            form.addRow(prm.name, row)
            self.widgets[prm.name] = (sl, val)
            self.refresh(prm.name)

        sound = QtWidgets.QGroupBox("sound")
        form = QtWidgets.QFormLayout(sound)
        for name, lo, hi in (
            ("volume", 0, 1.5),
            ("voices", 0, 2),
            ("drone", 0, 2),
            ("whoosh", 0, 2),
            ("reverb", 0, 1),
        ):
            sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            sl.setRange(0, self.STEPS)
            sl.setValue(int((getattr(viz.son, name) - lo) / (hi - lo) * self.STEPS))
            sl.valueChanged.connect(
                lambda v, n=name, a=lo, b=hi: setattr(viz.son, n, a + (b - a) * v / self.STEPS)
            )
            form.addRow(name, sl)
        lay.addWidget(sound)

        res = QtWidgets.QComboBox()
        res.addItems([f"{w}×{h}" for w, h in RESOLUTIONS])
        res.setCurrentIndex(
            RESOLUTIONS.index((viz.swarm.width, viz.swarm.height))
            if (viz.swarm.width, viz.swarm.height) in RESOLUTIONS
            else 2
        )
        res.setToolTip("simulation pixels: fewer is chunkier")
        res.currentIndexChanged.connect(lambda i: viz.swarm.resize(*RESOLUTIONS[i]))
        box = QtWidgets.QGroupBox("pixels")
        QtWidgets.QVBoxLayout(box).addWidget(res)
        lay.addWidget(box)

        keys = QtWidgets.QLabel(KEYS_HELP)
        keys.setStyleSheet("font-family: monospace; font-size: 8pt; color: #8890a8;")
        lay.addWidget(keys)
        lay.addStretch(1)
        self.setWidget(body)

    def _slid(self, name: str, v: int) -> None:
        prm = next(p for p in PARAMS if p.name == name)
        value = prm.lo + (prm.hi - prm.lo) * v / self.STEPS
        if (
            float(prm.default).is_integer()
            and float(prm.lo).is_integer()
            and float(prm.hi).is_integer()
            and prm.hi - prm.lo <= 10
        ):
            value = round(value)
        self.viz.set_param(name, value, from_ui=True)
        self.widgets[name][1].setText(f"{value:.3g}")  # type: ignore[union-attr]

    def refresh(self, name: str) -> None:
        if name not in self.widgets:
            return
        w, label = self.widgets[name]
        prm = next(p for p in PARAMS if p.name == name)
        value = self.viz.swarm.p[name]
        w.blockSignals(True)
        if isinstance(w, QtWidgets.QComboBox):
            w.setCurrentIndex(int(value))
        else:
            w.setValue(int(round((value - prm.lo) / (prm.hi - prm.lo) * self.STEPS)))  # type: ignore[attr-defined]
            label.setText(f"{value:.3g}")  # type: ignore[union-attr]
        w.blockSignals(False)


class Viz(QtWidgets.QMainWindow):
    def __init__(
        self,
        feats: Features,
        title: str,
        *,
        size: tuple[int, int] = (640, 360),
        params: dict | None = None,
        sound: bool = True,
    ):
        super().__init__()
        self.feats = feats
        self.swarm = Swarm(feats, *size)
        for k, v in (params or {}).items():
            self.swarm.set_param(k, v)
        self.son = Sonifier(feats)
        self.gs_np = feats.global_signal.float().cpu().numpy()
        self.dv_np = feats.dvars.float().cpu().numpy()
        self.playing = True
        self.setWindowTitle(f"ffs_dataviz — {title}")
        self.setStyleSheet(
            "QMainWindow, QScrollArea, QWidget { background: #07080e; color: #b8c0d8; }"
            "QGroupBox { border: 1px solid #1c2030; margin-top: 10px; padding-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 6px; color: #7f88a8; }"
        )
        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)
        self.controls = Controls(self)
        self.dock = QtWidgets.QDockWidget("controls", self)
        self.dock.setWidget(self.controls)
        self.dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.dock.hide()
        self.resize(1280, 760)

        self._morph_target: float | None = None
        self._last = time.perf_counter()
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(0)

        self.sound_on = False
        self.sink = None
        self.audio_io = None
        if sound:
            self._start_audio()

    # ------------------------------------------------------------------ audio

    def _start_audio(self) -> None:
        try:
            from PySide6 import QtMultimedia
        except ImportError:
            return
        fmt = QtMultimedia.QAudioFormat()
        fmt.setSampleRate(self.son.sr)
        fmt.setChannelCount(2)
        fmt.setSampleFormat(QtMultimedia.QAudioFormat.SampleFormat.Float)
        dev = QtMultimedia.QMediaDevices.defaultAudioOutput()
        if dev.isNull() or not dev.isFormatSupported(fmt):
            return
        self.sink = QtMultimedia.QAudioSink(dev, fmt, self)
        self.sink.setBufferSize(int(self.son.sr * 8 * 0.12))
        self.audio_io = self.sink.start()
        self.sound_on = True

    def _feed_audio(self) -> None:
        if self.sink is None or self.audio_io is None:
            return
        free = self.sink.bytesFree() // 8
        if free < 256:
            return
        self.son.gate = 1.0 if (self.playing and self.sound_on) else 0.0
        block = self.son.render(free, self.swarm.t)
        self.audio_io.write(block.tobytes())

    # ------------------------------------------------------------------- loop

    def tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        sw = self.swarm
        if self._morph_target is not None:
            m = sw.p["morph"]
            step = math.copysign(
                min(abs(self._morph_target - m), dt * 0.35), self._morph_target - m
            )
            self.set_param("morph", m + step)
            if abs(self._morph_target - sw.p["morph"]) < 1e-4:
                self._morph_target = None
        if self.playing:
            sw.step(dt)
        else:
            # Paused: data time holds, the swarm keeps settling and the camera keeps orbiting.
            speed = sw.p["speed"]
            sw.p["speed"] = 0.0
            sw.step(dt)
            sw.p["speed"] = speed
        self.canvas.set_frame(sw.render())
        self._feed_audio()

    def seek(self, t: float) -> None:
        self.swarm.seek(t)
        self.canvas.update()

    def set_param(self, name: str, value: float, from_ui: bool = False) -> None:
        self.swarm.set_param(name, value)
        if not from_ui:
            self.controls.refresh(name)

    # ------------------------------------------------------------------- keys

    def keyPressEvent(self, e) -> None:
        k = e.key()
        K = QtCore.Qt.Key
        shift = bool(e.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        sw = self.swarm
        if k == K.Key_Space:
            self.playing = not self.playing
        elif k in (K.Key_Tab, K.Key_M):
            self.dock.setVisible(not self.dock.isVisible())
            self.canvas.setFocus()
        elif k == K.Key_H:
            self.canvas.show_hud = not self.canvas.show_hud
        elif k == K.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif k == K.Key_S:
            self.sound_on = not self.sound_on
        elif k == K.Key_Left:
            self.seek(sw.t - (25 if shift else 5))
        elif k == K.Key_Right:
            self.seek(sw.t + (25 if shift else 5))
        elif k == K.Key_C:
            n = len(next(p for p in PARAMS if p.name == "color").choices)
            self.set_param("color", (int(sw.p["color"]) + 1) % n)
        elif k == K.Key_L:
            self._morph_target = 0.0 if sw.p["morph"] > 0.5 else 1.0
        elif k == K.Key_R:
            for n in ("yaw", "pitch", "zoom", "orbit"):
                self.set_param(n, next(p for p in PARAMS if p.name == n).default)
        elif k == K.Key_BracketLeft:
            self.set_param("speed", max(0.1, sw.p["speed"] / 1.25))
        elif k == K.Key_BracketRight:
            self.set_param("speed", min(20.0, sw.p["speed"] * 1.25))
        elif k in (K.Key_Q, K.Key_Escape):
            if k == K.Key_Escape and self.isFullScreen():
                self.showNormal()
            else:
                self.close()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e) -> None:
        self.timer.stop()
        if self.sink is not None:
            self.sink.stop()
        super().closeEvent(e)


def launch(feats: Features, title: str, **kwargs) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Viz(feats, title, **kwargs)
    win.show()
    win.canvas.setFocus()
    # The render timer hands control back to Python every frame, so a plain
    # handler is enough for Ctrl+C in the terminal.
    signal.signal(signal.SIGINT, lambda *_: win.close())
    return app.exec()
