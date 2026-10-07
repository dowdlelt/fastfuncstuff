"""The mesh list window: which white/pial meshes are loaded, drawn, and in use.

One row per mesh, top to bottom in list order; per hemisphere and type the
top-most row is the one edits, depth sampling and depth maps act on, and is
marked ``●``. Every change goes through a recorded command except Bak, which
is a safety copy and not part of what a script rebuilds.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.commands import Command
from fastfuncstuff.viewer.meshlist import KINDS, infer_label
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.vocab import LoadMesh, RemoveMesh, SaveMesh, SetMesh, UseMesh

if TYPE_CHECKING:
    from fastfuncstuff.viewer.session import ViewerSession

#: Table columns.
SHOW, COLOUR, NAME, HEMI, KIND = range(5)
#: What the type column says for each kind: "wm" because smoothwm is one too.
KIND_TEXT = {"white": "wm", "pial": "pial"}


class MeshWindow(QtWidgets.QWidget):
    """A top-level list of the loaded meshes, rebuilt when the list changes."""

    def __init__(
        self,
        session: Callable[[], ViewerSession],
        dispatch: Callable[[Command], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.setWindowTitle("meshes")
        self.setStyleSheet(theme.stylesheet())
        self._session = session
        self._dispatch = dispatch
        self._built = (-1, -1)
        self._pending = False

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "", "mesh", "hemi", "type"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(NAME, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for col in (SHOW, COLOUR, HEMI, KIND):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.cellDoubleClicked.connect(lambda row, _col: self._use(row))
        self.table.itemSelectionChanged.connect(self._sync_buttons)
        v.addWidget(self.table, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(4)
        self.use_button = self._make_button(
            "USE",
            "Put the selected mesh in use for its hemisphere and type: it moves\n"
            "to the top of its group, and edits and depth sampling act on it.\n"
            "The one it replaces stays listed. (Or double-click a row.)",
            lambda: self._use(self.table.currentRow()),
        )
        self.load_button = self._make_button(
            "LOAD…",
            "Add a white or pial surface file to the bottom of the list, shown\n"
            "for comparison. Hemi and type are read from the filename.",
            self._load,
        )
        self.save_button = self._make_button(
            "SAVE AS…", "Write the selected mesh to a file of your choosing.", self._save
        )
        self.remove_button = self._make_button(
            "REMOVE", "Drop the selected mesh from the list (not one in use).", self._remove
        )
        self.bak_button = self._make_button(
            "BAK",
            "Back up every mesh in use now: ?h.<type>.bak-<time> beside the\n"
            "subject's files, plus the edit log. Load one back to fall back to it.",
            self._backup,
        )
        for b in (
            self.use_button,
            self.load_button,
            self.save_button,
            self.remove_button,
            self.bak_button,
        ):
            buttons.addWidget(b)
        buttons.addStretch(1)
        v.addLayout(buttons)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("value")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        self.resize(460, 260)
        self.rebuild()

    def _make_button(self, text: str, tip: str, slot) -> QtWidgets.QPushButton:
        b = QtWidgets.QPushButton(text)
        b.setToolTip(tip)
        b.clicked.connect(slot)
        return b

    @property
    def surfaces(self):
        return self._session().surfaces

    def restyle(self) -> None:
        self.setStyleSheet(theme.stylesheet())

    # -- drawing ---------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the rows if the list changed -- on the next turn of the event loop.

        Deferred because the change usually comes from one of the row's own
        widgets, and replacing a widget inside its own signal is how Qt
        crashes.
        """
        if not self._pending:
            self._pending = True
            QtCore.QTimer.singleShot(0, self.rebuild)

    def rebuild(self) -> None:
        """Rebuild the rows now, if the list changed since they were built."""
        self._pending = False
        surfaces = self.surfaces
        stamp = (id(surfaces), surfaces.meshes_version)
        if stamp == self._built:
            return
        self._built = stamp
        keep = self._selected_key()
        hemis = sorted(surfaces.hemis)
        self.table.blockSignals(True)
        self.table.setRowCount(len(surfaces.meshes))
        for i, row in enumerate(surfaces.meshes):
            active = surfaces.is_active(row)
            show = QtWidgets.QCheckBox()
            show.setChecked(row.shown)
            show.toggled.connect(lambda on, k=row.key: self._do(SetMesh(k, shown=int(on))))
            self.table.setCellWidget(i, SHOW, _centred(show))

            swatch = QtWidgets.QPushButton()
            swatch.setFixedSize(22, 16)
            swatch.setToolTip("colour of this mesh's outlines")
            swatch.setStyleSheet(f"background-color: {QtGui.QColor.fromRgbF(*row.rgb).name()};")
            swatch.clicked.connect(lambda _=False, k=row.key: self._pick_colour(k))
            self.table.setCellWidget(i, COLOUR, _centred(swatch))

            item = QtWidgets.QTableWidgetItem(("● " if active else "   ") + row.label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, row.key)
            tip = str(row.path) if row.path else "in memory"
            item.setToolTip(f"{tip}\n{'in use' if active else 'comparison'}  [{row.key}]")
            if active:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.table.setItem(i, NAME, item)

            # A row in use cannot be relabelled: its hemi and type would be
            # left with no mesh at all.
            hemi = _combo(hemis, row.hemi, enabled=not active)
            hemi.currentTextChanged.connect(lambda h, k=row.key: self._do(SetMesh(k, hemi=h)))
            self.table.setCellWidget(i, HEMI, hemi)
            kind = _combo([KIND_TEXT[k] for k in KINDS], KIND_TEXT[row.kind], enabled=not active)
            kind.currentIndexChanged.connect(
                lambda n, k=row.key: self._do(SetMesh(k, kind=KINDS[n]))
            )
            self.table.setCellWidget(i, KIND, kind)
            if row.key == keep:
                self.table.selectRow(i)
        self.table.blockSignals(False)
        self._sync_buttons()

    def _selected_key(self) -> str | None:
        item = self.table.item(self.table.currentRow(), NAME)
        return None if item is None else item.data(QtCore.Qt.ItemDataRole.UserRole)

    def _selected(self):
        key = self._selected_key()
        return None if key is None else self.surfaces.mesh(key)

    def _sync_buttons(self) -> None:
        row = self._selected()
        surfaces = self.surfaces
        active = row is not None and surfaces.is_active(row)
        self.use_button.setEnabled(row is not None and not active)
        self.remove_button.setEnabled(row is not None and not active)
        self.save_button.setEnabled(row is not None)
        self.load_button.setEnabled(bool(surfaces.hemis))
        self.bak_button.setEnabled(bool(surfaces.hemis))

    # -- actions -----------------------------------------------------------
    def _do(self, command: Command) -> bool:
        try:
            self._dispatch(command)
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(str(exc))
            self._built = (-1, -1)  # put the widgets back to what the list says
            self.refresh()
            return False
        self.status.setText("")
        self.refresh()
        return True

    def _use(self, index: int) -> None:
        item = self.table.item(index, NAME)
        if item is None:
            return
        key = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if self._do(UseMesh(key)):
            row = self.surfaces.mesh(key)
            self.status.setText(f"{row.hemi} {KIND_TEXT[row.kind]}: now using {row.name}")

    def _pick_colour(self, key: str) -> None:
        row = self.surfaces.mesh(key)
        colour = QtWidgets.QColorDialog.getColor(QtGui.QColor.fromRgbF(*row.rgb), self, row.name)
        if colour.isValid():
            rgb = f"{colour.redF():.3f},{colour.greenF():.3f},{colour.blueF():.3f}"
            self._do(SetMesh(key, rgb=rgb))

    def _start_dir(self) -> str:
        surfaces = self.surfaces
        if surfaces.subject is not None:
            return str(Path(surfaces.subject) / "surf")
        return str(Path.cwd())

    def _load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load a white or pial surface", self._start_dir()
        )
        if path:
            self.load_path(path)

    def load_path(self, path: str) -> None:
        """Load a file, asking only for what its name does not say."""
        hemi, kind = infer_label(path)
        hemis = sorted(self.surfaces.hemis)
        if hemi not in hemis:
            if len(hemis) == 1:
                hemi = hemis[0]
            else:
                hemi, ok = QtWidgets.QInputDialog.getItem(
                    self, "Hemisphere", f"Which hemisphere is {Path(path).name}?", hemis, 0, False
                )
                if not ok:
                    return
        if kind is None:
            text, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Type",
                f"Is {Path(path).name} a pial or a wm surface?",
                ["pial", "wm"],
                0,
                False,
            )
            if not ok:
                return
            kind = "pial" if text == "pial" else "white"
        if self._do(LoadMesh(path, hemi, kind)):
            self.status.setText(f"loaded {Path(path).name} as {hemi} {KIND_TEXT[kind]}")

    def _save(self) -> None:
        row = self._selected()
        if row is None:
            return
        start = Path(self._start_dir()) / f"{row.hemi}.{row.kind}.ffsedit"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, f"Save {row.name}", str(start))
        if not path:
            return
        # The dialog has already asked about replacing an existing file.
        if self._do(SaveMesh(row.key, path, overwrite=True)):
            self.status.setText(f"wrote {path}")

    def _remove(self) -> None:
        row = self._selected()
        if row is not None:
            self._do(RemoveMesh(row.key))

    def _backup(self) -> None:
        try:
            written = self.surfaces.backup()
        except (OSError, ValueError) as exc:
            self.status.setText(f"backup failed: {exc}")
            return
        surfaces = [p for p in written if not p.name.endswith(".json")]
        if not surfaces:
            self.status.setText(
                "nothing backed up (a hemisphere whose topology changed is saved whole: use save)"
            )
            return
        self.status.setText(
            f"backed up {', '.join(p.name for p in surfaces)} in {surfaces[0].parent}"
        )


def _centred(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    holder = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(holder)
    lay.setContentsMargins(4, 0, 4, 0)
    lay.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(widget)
    return holder


def _combo(options: list[str], current: str, *, enabled: bool) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox()
    combo.addItems(options)
    combo.setCurrentText(current)
    combo.setEnabled(enabled)
    return combo


__all__ = ["MeshWindow"]
