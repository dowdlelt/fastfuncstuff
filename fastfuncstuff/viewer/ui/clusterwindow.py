"""The cluster window: a thresholded map as a list of things you can point at.

Three jobs, and the third is the one that makes the other two worth having:

* **Report.** Size, volume in mm3, peak and its coordinates, centre of mass,
  and -- when the dataset carries its own ClustSim tables -- the corrected
  alpha each cluster earned. That is the table people paste into a paper.
* **Navigate.** Clicking a row puts the crosshair on that cluster's peak, which
  is the gesture the whole panel exists for.
* **Promote.** MAKE ROIS turns the table into an ROI layer, at which point the
  clusters are an atlas: the matrix can correlate them, the readout names them,
  a seed can come from one. Nothing downstream learns what a cluster is.

Clustering one volume is milliseconds, so this rebuilds as the threshold moves
rather than being a button you remember to press. It is the one built window
that is not on the worker, and that is a measurement rather than an oversight.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.clusters import ClusterTable
from fastfuncstuff.viewer.commands import Command
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp, keep_keys_for_shortcuts
from fastfuncstuff.viewer.viewports import Viewport

COLUMNS = ("#", "voxels", "mm³", "peak", "x", "y", "z", "α")
#: Height of the selected cluster's mean time course, in pixels.
TRACE_HEIGHT = 64
BARE_WIDTH = 260


class TraceStrip(QtWidgets.QWidget):
    """The selected cluster's mean time course.

    A cluster's own average, not the peak voxel's: the peak is the most extreme
    voxel and therefore the one whose time course is most selected-for, which
    is the classic way to make a plot look better than the effect.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: np.ndarray | None = None
        self._color = (200, 200, 200)
        self._label = ""
        self.setFixedHeight(TRACE_HEIGHT)

    def set_series(self, series: np.ndarray | None, color, label: str = "") -> None:
        self._series = None if series is None or series.size < 2 else np.asarray(series)
        self._color = tuple(int(c) for c in color)
        self._label = label
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        if self._series is None:
            p.setPen(QtGui.QColor(c.faint))
            p.drawText(
                self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "pick a cluster for its mean"
            )
            p.end()
            return
        series = self._series - float(self._series.mean())
        span = float(np.abs(series).max()) or 1.0
        path = QtGui.QPainterPath()
        for i, value in enumerate(series):
            x = i / max(len(series) - 1, 1) * self.width()
            y = self.height() * (0.5 - 0.42 * float(value) / span)
            path.lineTo(x, y) if i else path.moveTo(x, y)
        pen = QtGui.QPen(QtGui.QColor(*self._color))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawPath(path)
        if self._label:
            p.setPen(QtGui.QColor(c.faint))
            p.drawText(4, 12, self._label)
        p.end()


class ClusterWindow(QtWidgets.QWidget):
    """A floating cluster report."""

    closed = QtCore.Signal(str)
    #: (i, j, k) of a clicked cluster's peak.
    located = QtCore.Signal(int, int, int)
    #: Asks the controller to turn this table into an ROI layer.
    rois_requested = QtCore.Signal(str)
    rebuild_requested = QtCore.Signal(str)

    def __init__(
        self,
        vid: str,
        session,
        dispatch: Callable[[Command], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.vid = vid
        self.session = session
        self._dispatch = dispatch
        self._table: ClusterTable | None = None
        self._source_key = ""
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.resize(560, 460)
        self.setMinimumSize(150, 140)
        self.setStyleSheet(theme.stylesheet())

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(7, 6, 7, 7)
        v.setSpacing(5)

        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(5)
        bar.addWidget(QtWidgets.QLabel("NN"))
        self.nn_box = QtWidgets.QComboBox()
        for n in (1, 2, 3):
            self.nn_box.addItem(f"NN{n}", userData=n)
        self.nn_box.setToolTip(
            "Which voxels count as touching: NN1 faces, NN2 edges, NN3 corners. "
            "It has to match the ClustSim table the alphas come from."
        )
        self.nn_box.activated.connect(lambda _: self.rebuild_requested.emit(self.vid))
        bar.addWidget(self.nn_box)

        bar.addWidget(QtWidgets.QLabel("MIN"))
        self.min_spin = QtWidgets.QSpinBox()
        self.min_spin.setRange(1, 100_000)
        self.min_spin.setValue(1)
        self.min_spin.setMaximumWidth(80)
        self.min_spin.setToolTip("Drop clusters below this many voxels")
        self.min_spin.valueChanged.connect(lambda _: self.rebuild_requested.emit(self.vid))
        bar.addWidget(self.min_spin)

        bar.addStretch(1)
        self.rois_button = QtWidgets.QPushButton(theme.key_label("MAKE ROIS", "m"))
        self.rois_button.setToolTip(
            "Add these clusters to the stack as an ROI layer, usable as matrix "
            "nodes, as a seed, or as a named readout (m)"
        )
        self.rois_button.clicked.connect(lambda: self.rois_requested.emit(self.vid))
        bar.addWidget(self.rois_button)
        self.controls = QtWidgets.QWidget()
        self.controls.setLayout(bar)
        v.addWidget(self.controls)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.currentCellChanged.connect(lambda row, *_: self._select(row))
        v.addWidget(self.table, 1)

        self.trace = TraceStrip()
        v.addWidget(self.trace)

        self.info = QtWidgets.QLabel("")
        self.info.setObjectName("value")
        self.info.setWordWrap(True)
        v.addWidget(self.info)

        self.help = ShortcutHelp(self, f"clusters · {vid}")
        self.help.apply(
            [
                Binding(
                    "m", "make an ROI layer of these", self.rois_button.click, group="clusters"
                ),
                Binding(
                    "r",
                    "recompute",
                    lambda: self.rebuild_requested.emit(self.vid),
                    group="clusters",
                ),
                Binding("click", "jump to that cluster's peak", None, group="clusters"),
                Binding("h", "this list", self.help.toggle, group="window"),
                Binding("w", "close this window", self.close, group="window"),
            ]
        )
        keep_keys_for_shortcuts(self)

    # -- settings the controller reads ---------------------------------
    @property
    def nn(self) -> int:
        return int(self.nn_box.currentData() or 1)

    @property
    def min_voxels(self) -> int:
        return int(self.min_spin.value())

    # -- input ---------------------------------------------------------
    def _select(self, row: int) -> None:
        if self._table is None or not (0 <= row < len(self._table)):
            return
        cluster = self._table.clusters[row]
        self.located.emit(*cluster.peak_ijk)
        self._show_trace(cluster)

    def _show_trace(self, cluster) -> None:
        """Average the cluster's voxels in whichever run is loaded under it."""
        from fastfuncstuff.viewer.rois import roi_color

        series = self.session.cluster_series(self._table, cluster.index)
        self.trace.set_series(
            series,
            roi_color(cluster.index - 1),
            f"C{cluster.index}  ·  {cluster.n_voxels} voxels",
        )

    def _viewport(self) -> Viewport | None:
        return self.session.state.viewports.find(self.vid)

    # -- output --------------------------------------------------------
    def apply(self, viewport: Viewport) -> None:
        self.setWindowTitle(viewport.title)

    def show_table(self, source_key: str, table: ClusterTable | None, message: str = "") -> None:
        self._table = table
        self._source_key = source_key
        self.trace.set_series(None, (0, 0, 0))
        self.table.setRowCount(0 if table is None else len(table))
        if table is None:
            self.info.setText(message or "nothing to cluster")
            return
        for row, cluster in enumerate(table):
            cells = (
                str(cluster.index),
                f"{cluster.n_voxels:,}",
                f"{cluster.volume_mm3:,.0f}",
                f"{cluster.peak:+.3g}",
                f"{cluster.peak_xyz[0]:.1f}",
                f"{cluster.peak_xyz[1]:.1f}",
                f"{cluster.peak_xyz[2]:.1f}",
                self._alpha_text(table, cluster),
            )
            for column, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        note = f"   · {table.note}" if table.note else ""
        self.info.setText(message or f"{table.summary()}{note}")

    @staticmethod
    def _alpha_text(table: ClusterTable, cluster) -> str:
        """An alpha at the edge of the simulated range is a bound, and says so.

        Printing 0.01 for a cluster ten times bigger than anything simulated
        would understate it, and printing an extrapolated 1e-6 would invent a
        number. "<0.01" is the thing that is actually true.
        """
        if cluster.alpha is None:
            return "--"
        simulated = table.alpha_range
        if simulated is None:
            return f"{cluster.alpha:.3g}"
        strictest, loosest = simulated
        if cluster.alpha <= strictest:
            return f"<{strictest:g}"
        if cluster.alpha >= loosest:
            return f">{loosest:g}"
        return f"{cluster.alpha:.3g}"

    def set_busy(self, busy: bool) -> None:
        self.controls.setEnabled(not busy)

    def mark_stale(self) -> None:
        """Never stale: this recomputes whenever the threshold moves."""

    def refresh(self) -> None:
        """The table follows the threshold, which the controller pushes in."""

    def restyle(self) -> None:
        self.setStyleSheet(theme.stylesheet())
        self.trace.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 (Qt)
        narrow = event.size().width() < BARE_WIDTH
        self.controls.setVisible(not narrow)
        self.trace.setVisible(event.size().height() > 220)
        super().resizeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.closed.emit(self.vid)
        super().closeEvent(event)


__all__ = ["COLUMNS", "ClusterWindow", "TraceStrip"]
