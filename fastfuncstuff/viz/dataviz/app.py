"""The ffs_dataviz window: a canvas, a hidden control panel, and live sound.

The canvas shows the scene's low-resolution buffer scaled up with
nearest-neighbour sampling, so every simulated pixel stays a crisp square.
Everything tunable lives in a side panel that starts hidden (Tab or m); it
shows the shared groups plus the active scene's own.
"""

from __future__ import annotations

import math
import signal
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .engine import PARAM_BY_NAME, PARAMS, Visualizer
from .features import Features
from .scenes import SCENE_NAMES, SETS
from .sound import Sonifier

RESOLUTIONS = ((320, 180), (480, 270), (640, 360), (960, 540), (1280, 720), (1920, 1080))

#: The wheel's zoom parameter in each scene.
ZOOM_PARAM = {
    "brain": "zoom",
    "mandala": "mzoom",
    "flow": "fzoom",
    "milkdrop": "isize",
    "galaxies": "gzoom",
    "constellation": "czoom",
}

KEYS_HELP = """\
1-9 0  scene in the current set   g  next set
space  play / pause        ← →  step 5 TRs (shift: 25)
tab m  controls            h    hide everything but the picture (again: back)
f      fullscreen          s    sound on / off
c      next colour source  [ ]  slower / faster
click  probe a voxel       i inspector  p z/PSC  o raw/pooled
l      brain: anatomy ↔ correlation space (animated)
r      reset view          drag orbit (brain)   wheel zoom
q esc  quit"""


class Canvas(QtWidgets.QWidget):
    def __init__(self, parent: Viz):
        super().__init__(parent)
        self.viz = parent
        self.image: QtGui.QImage | None = None
        self._buf: np.ndarray | None = None
        self.show_hud = True
        self._drag: QtCore.QPointF | None = None
        self._press: QtCore.QPointF | None = None
        self._scrub = False
        self._target = QtCore.QRectF()
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 180)

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
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0))
        if self.image is not None:
            iw, ih = self.image.width(), self.image.height()
            s = min(self.width() / iw, self.height() / ih)
            w, h = int(iw * s), int(ih * s)
            target = QtCore.QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)
            self._target = QtCore.QRectF(target)
            p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
            p.drawImage(target, self.image)
            self._draw_probe(p)
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
        gs, dv = viz.gs_np, viz.dv_np
        n = gs.size
        xs = np.linspace(x0, x1, n)
        # Global signal as a faint line, DVARS spikes as ticks: the whole-brain
        # story of the run at a glance, with the playhead on it.
        mid = y0 + band / 2
        ys = mid - np.clip(gs, -3, 3) / 3 * (band / 2 - 2)
        path = QtGui.QPainterPath()
        path.moveTo(xs[0], ys[0])
        for i in range(1, n):
            path.lineTo(xs[i], ys[i])
        p.setPen(QtGui.QPen(QtGui.QColor(140, 160, 220, 90), 1))
        p.drawPath(path)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 200, 120, 120), 1))
        for i in np.nonzero(dv > 3.0)[0]:
            p.drawLine(QtCore.QPointF(xs[i], y0 + band), QtCore.QPointF(xs[i], y0 + band - 6))
        p.setPen(QtGui.QPen(QtGui.QColor(120, 130, 160, 80), 1, QtCore.Qt.PenStyle.DashLine))
        for start in f.run_starts[1:]:
            p.drawLine(QtCore.QPointF(xs[start], y0), QtCore.QPointF(xs[start], y0 + band))
        px = x0 + (x1 - x0) * viz.vis.t / max(n - 1, 1)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 170), 1))
        p.drawLine(QtCore.QPointF(px, y0 - 2), QtCore.QPointF(px, y0 + band + 2))
        p.setPen(QtGui.QColor(170, 180, 210, 150))
        font = p.font()
        font.setPointSizeF(8.5)
        p.setFont(font)
        self._draw_sets(p, x0, y0 - 22)
        status = "" if viz.playing else "  ❚❚"
        sound = "  ♪" if viz.sound_on else ""
        t = viz.vis.t
        p.drawText(
            x0,
            y0 - 6,
            f"{viz.vis.scene_name}   {t * f.tr:6.1f} s   TR {t:5.1f}/{n}   ×{viz.vis.p['speed']:.1f}{status}{sound}",
        )

    def to_scene(self, pos: QtCore.QPointF) -> tuple[float, float]:
        t = self._target
        vis = self.viz.vis
        return (pos.x() - t.left()) / max(t.width(), 1) * vis.width, (pos.y() - t.top()) / max(
            t.height(), 1
        ) * vis.height

    def _draw_probe(self, p: QtGui.QPainter) -> None:
        insp = self.viz.inspector
        if not insp.isVisible():
            return
        at = self.viz.vis.scene.locate(insp.probe)
        if at is None:
            return
        t = self._target
        vis = self.viz.vis
        x = t.left() + at[0] / vis.width * t.width()
        y = t.top() + at[1] / vis.height * t.height()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1.2))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawEllipse(QtCore.QPointF(x, y), 7, 7)

    def resizeEvent(self, e) -> None:
        self.viz.place_inspector()
        super().resizeEvent(e)

    def _draw_sets(self, p: QtGui.QPainter, x: float, y: float) -> None:
        """The current set's scenes with their keys, the current one lit."""
        viz = self.viz
        name, names = SETS[viz.scene_set]
        fm = p.fontMetrics()
        p.setPen(QtGui.QColor(120, 128, 160, 150))
        label = f"set {viz.scene_set + 1}/{len(SETS)} {name} (g)  "
        p.drawText(QtCore.QPointF(x, y), label)
        x += fm.horizontalAdvance(label)
        for i, scene in enumerate(names):
            text = f"{(i + 1) % 10} {scene}"
            lit = scene == viz.vis.scene_name
            p.setPen(QtGui.QColor(235, 215, 160, 230) if lit else QtGui.QColor(150, 158, 190, 140))
            p.drawText(QtCore.QPointF(x, y), text)
            x += fm.horizontalAdvance(text + "   ")

    def _in_hud(self, y: float) -> bool:
        return self.show_hud and y > self.height() - 50

    def _seek_to(self, x: float) -> None:
        frac = (x - 16) / max(1, self.width() - 32)
        self.viz.seek(float(np.clip(frac, 0, 1)) * (self.viz.feats.n_time - 1))

    def mousePressEvent(self, e) -> None:
        if e.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if self._in_hud(e.position().y()):
            self._scrub = True
            self._seek_to(e.position().x())
        else:
            self._drag = e.position()
            self._press = e.position()

    def mouseMoveEvent(self, e) -> None:
        if self._scrub:
            self._seek_to(e.position().x())
            return
        if self._drag is None:
            return
        d = e.position() - self._drag
        self._drag = e.position()
        if self.viz.vis.scene_name == "brain":
            p = self.viz.vis.p
            self.viz.set_param("yaw", p["yaw"] + d.x() * 0.008)
            self.viz.set_param("pitch", float(np.clip(p["pitch"] + d.y() * 0.008, -1.5, 1.5)))

    def mouseReleaseEvent(self, e) -> None:
        # A click that did not drag probes the voxel under it.
        if self._press is not None and (e.position() - self._press).manhattanLength() < 4:
            x, y = self.to_scene(e.position())
            radius = 6.0 * self.viz.vis.width / max(self._target.width(), 1) + 3
            v = self.viz.vis.scene.pick(x, y, radius)
            if v is not None:
                self.viz.place_inspector()
                self.viz.inspector.set_probe(v)
                self.viz.inspector.show()
        self._press = None
        self._drag = None
        self._scrub = False

    def wheelEvent(self, e) -> None:
        name = ZOOM_PARAM.get(self.viz.vis.scene_name)
        if name is None:
            return
        prm = PARAM_BY_NAME[name]
        z = self.viz.vis.p[name] * (1.1 ** (e.angleDelta().y() / 120))
        self.viz.set_param(name, float(np.clip(z, prm.lo, prm.hi)))


class Controls(QtWidgets.QScrollArea):
    """Every parameter as a slider (or a combo for a choice), grouped."""

    STEPS = 1000

    def __init__(self, viz: Viz):
        super().__init__()
        self.viz = viz
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(310)
        body = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        self.widgets: dict[str, tuple[QtWidgets.QWidget, QtWidgets.QLabel | None]] = {}
        self.scene_boxes: dict[str, QtWidgets.QGroupBox] = {}
        self.blurb = QtWidgets.QLabel()
        self.blurb.setWordWrap(True)
        self.blurb.setStyleSheet("color: #8890a8; font-size: 8pt;")
        groups: dict[str, QtWidgets.QFormLayout] = {}
        for prm in PARAMS:
            if prm.group not in groups:
                box = QtWidgets.QGroupBox(prm.group)
                form = QtWidgets.QFormLayout(box)
                form.setContentsMargins(6, 4, 6, 4)
                groups[prm.group] = form
                lay.addWidget(box)
                if prm.scene is not None:
                    self.scene_boxes[prm.scene] = box
                if prm.group == "scene":
                    form.addRow(self.blurb)
            self._add(groups[prm.group], prm)

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
        size = (viz.vis.width, viz.vis.height)
        res.setCurrentIndex(RESOLUTIONS.index(size) if size in RESOLUTIONS else 2)
        res.setToolTip("simulation pixels: fewer is chunkier")
        res.currentIndexChanged.connect(lambda i: viz.vis.resize(*RESOLUTIONS[i]))
        box = QtWidgets.QGroupBox("pixels")
        QtWidgets.QVBoxLayout(box).addWidget(res)
        lay.addWidget(box)

        keys = QtWidgets.QLabel(KEYS_HELP)
        keys.setStyleSheet("font-family: monospace; font-size: 8pt; color: #8890a8;")
        lay.addWidget(keys)
        lay.addStretch(1)
        self.setWidget(body)
        self.show_scene()

    def _add(self, form: QtWidgets.QFormLayout, prm) -> None:
        viz = self.viz
        if prm.choices:
            combo = QtWidgets.QComboBox()
            combo.addItems(list(prm.choices))
            combo.setCurrentIndex(int(viz.vis.p[prm.name]))
            combo.setToolTip(prm.help)
            combo.currentIndexChanged.connect(
                lambda i, n=prm.name: viz.set_param(n, float(i), from_ui=True)
            )
            form.addRow(prm.name, combo)
            self.widgets[prm.name] = (combo, None)
            return
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
        label = QtWidgets.QLabel(prm.name)
        label.setToolTip(prm.help)
        form.addRow(label, row)
        self.widgets[prm.name] = (sl, val)
        self.refresh(prm.name)

    def show_scene(self) -> None:
        from .scenes import SCENES

        name = self.viz.vis.scene_name
        for scene, box in self.scene_boxes.items():
            box.setVisible(scene == name)
        self.blurb.setText(SCENES[name].blurb)

    def _slid(self, name: str, v: int) -> None:
        prm = PARAM_BY_NAME[name]
        value = prm.lo + (prm.hi - prm.lo) * v / self.STEPS
        if prm.integer:
            value = round(value)
        self.viz.set_param(name, value, from_ui=True)
        label = self.widgets[name][1]
        if label is not None:
            label.setText(f"{value:.3g}")

    def refresh(self, name: str) -> None:
        if name not in self.widgets:
            return
        w, label = self.widgets[name]
        prm = PARAM_BY_NAME[name]
        value = self.viz.vis.p[name]
        w.blockSignals(True)
        if isinstance(w, QtWidgets.QComboBox):
            w.setCurrentIndex(int(value))
        elif isinstance(w, QtWidgets.QSlider):
            w.setValue(int(round((value - prm.lo) / (prm.hi - prm.lo) * self.STEPS)))
        if label is not None:
            label.setText(f"{value:.3g}")
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
        start: float = 0.0,
    ):
        super().__init__()
        self.feats = feats
        self.vis = Visualizer(feats, *size)
        for k, v in (params or {}).items():
            self.vis.set_param(k, v)
        self.vis.seek(start)
        self.son = Sonifier(feats)
        self.gs_np = feats.global_signal.float().cpu().numpy()
        self.dv_np = feats.dvars.float().cpu().numpy()
        self.playing = True
        self.scene_set = next(
            i for i, (_, names) in enumerate(SETS) if self.vis.scene_name in names
        )
        self._set_memory = {i: names[0] for i, (_, names) in enumerate(SETS)}
        self._hidden: dict | None = None
        self.setWindowTitle(f"ffs_dataviz — {title}")
        self.setStyleSheet(
            "QMainWindow, QScrollArea, QWidget { background: #07080e; color: #b8c0d8; }"
            "QGroupBox { border: 1px solid #1c2030; margin-top: 10px; padding-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 6px; color: #7f88a8; }"
        )
        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)
        from .inspector import Inspector

        self.inspector = Inspector(self, self.canvas)
        self.inspector.hide()
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
        self.audio_io.write(self.son.render(free, self.vis.t).tobytes())

    # ------------------------------------------------------------------- loop

    def tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if self._morph_target is not None:
            m = self.vis.p["morph"]
            step = math.copysign(
                min(abs(self._morph_target - m), dt * 0.35), self._morph_target - m
            )
            self.set_param("morph", m + step)
            if abs(self._morph_target - self.vis.p["morph"]) < 1e-4:
                self._morph_target = None
        # Paused, data time holds but the scene keeps moving: the swarm settles,
        # the camera orbits, the tunnel keeps flowing.
        self.vis.step(dt, self.playing)
        self.canvas.set_frame(self.vis.render())
        if self.inspector.isVisible():
            self.inspector.update()
        self._feed_audio()

    def seek(self, t: float) -> None:
        self.vis.seek(t)
        self.canvas.update()

    def select_scene(self, name: str) -> None:
        self.set_param("scene", float(SCENE_NAMES.index(name)))

    def toggle_hidden(self) -> None:
        """Hide everything but the picture; the second press restores it."""
        if self._hidden is None:
            self._hidden = {
                "hud": self.canvas.show_hud,
                "inspector": self.inspector.isVisible(),
                "dock": self.dock.isVisible(),
            }
            self.canvas.show_hud = False
            self.inspector.hide()
            self.dock.hide()
            self.canvas.setCursor(QtCore.Qt.CursorShape.BlankCursor)
        else:
            self.canvas.show_hud = self._hidden["hud"]
            self.inspector.setVisible(self._hidden["inspector"])
            self.dock.setVisible(self._hidden["dock"])
            self.canvas.unsetCursor()
            self._hidden = None
        self.canvas.setFocus()

    def place_inspector(self) -> None:
        if not hasattr(self, "inspector"):
            return
        c = self.canvas
        w = int(min(560, max(360, c.width() * 0.42)))
        h = int(min(420, max(260, c.height() * 0.5)))
        self.inspector.setGeometry(c.width() - w - 12, c.height() - h - 60, w, h)

    def set_param(self, name: str, value: float, from_ui: bool = False) -> None:
        self.vis.set_param(name, value)
        if name == "families":
            # New families, new voices; the inspector's family line follows.
            gate, vol = self.son.gate, self.son.volume
            self.son = Sonifier(self.feats)
            self.son.gate, self.son.volume = gate, vol
            self.inspector.set_probe(self.inspector.probe)
        if not from_ui:
            self.controls.refresh(name)
        if name == "scene":
            self.controls.show_scene()
            scene = SCENE_NAMES[int(value)]
            for i, (_, names) in enumerate(SETS):
                if scene in names:
                    self.scene_set = i
                    self._set_memory[i] = scene

    # ------------------------------------------------------------------- keys

    def keyPressEvent(self, e) -> None:
        k = e.key()
        K = QtCore.Qt.Key
        shift = bool(e.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        vis = self.vis
        if K.Key_0 <= k <= K.Key_9:
            names = SETS[self.scene_set][1]
            i = (k - K.Key_1) % 10
            if i < len(names):
                self.select_scene(names[i])
        elif k == K.Key_G:
            self.scene_set = (self.scene_set + 1) % len(SETS)
            self.select_scene(self._set_memory[self.scene_set])
        elif k == K.Key_Space:
            self.playing = not self.playing
        elif k in (K.Key_Tab, K.Key_M):
            self.dock.setVisible(not self.dock.isVisible())
            self.canvas.setFocus()
        elif k == K.Key_H:
            self.toggle_hidden()
        elif k == K.Key_I:
            self.place_inspector()
            self.inspector.setVisible(not self.inspector.isVisible())
        elif k == K.Key_P:
            self.inspector.psc = not self.inspector.psc
            self.inspector.update()
        elif k == K.Key_O:
            self.inspector.overlay_shown = not self.inspector.overlay_shown
            self.inspector.update()
        elif k == K.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif k == K.Key_S:
            self.sound_on = not self.sound_on
        elif k == K.Key_Left:
            self.seek(vis.t - (25 if shift else 5))
        elif k == K.Key_Right:
            self.seek(vis.t + (25 if shift else 5))
        elif k == K.Key_C:
            n = len(PARAM_BY_NAME["color"].choices)
            self.set_param("color", (int(vis.p["color"]) + 1) % n)
        elif k == K.Key_L:
            self._morph_target = 0.0 if vis.p["morph"] > 0.5 else 1.0
        elif k == K.Key_R:
            for n in ZOOM_PARAM.values():
                self.set_param(n, PARAM_BY_NAME[n].default)
            for n in ("yaw", "pitch", "orbit"):
                self.set_param(n, PARAM_BY_NAME[n].default)
        elif k == K.Key_BracketLeft:
            self.set_param("speed", max(0.1, vis.p["speed"] / 1.25))
        elif k == K.Key_BracketRight:
            self.set_param("speed", min(20.0, vis.p["speed"] * 1.25))
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
