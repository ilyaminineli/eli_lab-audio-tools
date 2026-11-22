"""PySide6 interface for pitch-folder-first Eugene:RE restoration."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal, Qt, QUrl
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .io import read_wav
from .pitch_studio import (
    analyze_pitch_folder, default_profile, discover_pitch_layers,
    pitch_name, preview_sample, process_pitch_folder, suggest_reference_match,
)


class TaskWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self, task: Callable, parent=None):
        super().__init__(parent)
        self.task = task

    def run(self):
        try:
            result = self.task(
                lambda done, total, label: self.progress.emit(done, total, label)
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class EnvelopePlot(QWidget):
    """Small dependency-free spectral-envelope plot."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frequencies: list[float] = []
        self.values: list[float] = []
        self.low: list[float] = []
        self.high: list[float] = []
        self.setMinimumHeight(190)

    def set_data(self, envelope: dict | None):
        envelope = envelope or {}
        self.frequencies = list(envelope.get("frequency_hz", []))
        self.values = list(envelope.get("median_relative_db", []))
        self.low = list(envelope.get("p10_relative_db", []))
        self.high = list(envelope.get("p90_relative_db", []))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fbfaf7"))
        left, right, top, bottom = 52, self.width() - 18, 14, self.height() - 30
        if right <= left or bottom <= top:
            return
        painter.setPen(QPen(QColor("#e5e2db"), 1))
        for db in (-12, -6, 0, 6, 12):
            y = bottom - (db + 12) / 24 * (bottom - top)
            painter.drawLine(left, int(y), right, int(y))
            painter.setPen(QColor("#77746d"))
            painter.drawText(6, int(y + 4), f"{db:+d} dB")
            painter.setPen(QPen(QColor("#e5e2db"), 1))
        painter.setPen(QColor("#77746d"))
        for hz in (100, 300, 1000, 3000, 10000):
            x = left + (np.log10(hz) - 2.0) / 2.0 * (right - left)
            painter.drawText(int(x - 18), self.height() - 8, f"{hz:g}")
            painter.setPen(QPen(QColor("#eeeae3"), 1))
            painter.drawLine(int(x), top, int(x), bottom)
            painter.setPen(QColor("#77746d"))
        if len(self.frequencies) < 2 or len(self.values) != len(self.frequencies):
            painter.drawText(left + 12, top + 22, "Analyse a pitch layer to inspect its median spectral envelope.")
            return

        def point(index, value):
            x = left + (np.log10(max(self.frequencies[index], 100.0)) - 2.0) / 2.0 * (right - left)
            y = bottom - (float(np.clip(value, -12, 12)) + 12) / 24 * (bottom - top)
            return int(x), int(y)

        if len(self.low) == len(self.frequencies) and len(self.high) == len(self.frequencies):
            area = QPainterPath()
            area.moveTo(*point(0, self.high[0]))
            for i in range(1, len(self.frequencies)):
                area.lineTo(*point(i, self.high[i]))
            for i in range(len(self.frequencies) - 1, -1, -1):
                area.lineTo(*point(i, self.low[i]))
            area.closeSubpath()
            painter.fillPath(area, QColor(100, 110, 116, 30))
        line = QPainterPath()
        line.moveTo(*point(0, self.values[0]))
        for i in range(1, len(self.frequencies)):
            line.lineTo(*point(i, self.values[i]))
        painter.setPen(QPen(QColor("#252927"), 2))
        painter.drawPath(line)


class PitchStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eugene:RE — Pitch Restoration Studio")
        self.resize(1240, 820)
        self.layers: list[dict] = []
        self.profiles: dict[str, dict] = {}
        self.analyses: dict[str, dict] = {}
        self.worker: TaskWorker | None = None
        self.preview_directory = tempfile.TemporaryDirectory(prefix="eugene_pitch_preview_")
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.9)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("EUGENE:RE  /  PITCH RESTORATION")
        title.setObjectName("pageTitle")
        header.addWidget(title)
        header.addStretch()
        self.status_label = QLabel("Select a voicebank folder to begin.")
        self.status_label.setObjectName("muted")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 12, 0)
        bank_group = QGroupBox("VOICEBANK")
        bank_layout = QVBoxLayout(bank_group)
        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("Choose the voicebank root or a pitch folder…")
        bank_layout.addWidget(self.root_edit)
        root_buttons = QHBoxLayout()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.choose_root)
        scan = QPushButton("Scan pitch folders")
        scan.clicked.connect(self.scan_layers)
        root_buttons.addWidget(browse)
        root_buttons.addWidget(scan)
        bank_layout.addLayout(root_buttons)
        left_layout.addWidget(bank_group)

        layer_group = QGroupBox("PITCH LAYERS  /  checked = process")
        layer_layout = QVBoxLayout(layer_group)
        self.layer_list = QListWidget()
        self.layer_list.currentItemChanged.connect(self.current_layer_changed)
        layer_layout.addWidget(self.layer_list, 1)
        selection_buttons = QHBoxLayout()
        select_current = QPushButton("Select current")
        select_current.clicked.connect(self.select_current_layer)
        select_all = QPushButton("Select all")
        select_all.clicked.connect(self.select_all_layers)
        clear_selection = QPushButton("Clear")
        clear_selection.clicked.connect(self.clear_layer_selection)
        selection_buttons.addWidget(select_current)
        selection_buttons.addWidget(select_all)
        selection_buttons.addWidget(clear_selection)
        layer_layout.addLayout(selection_buttons)
        left_layout.addWidget(layer_group, 1)

        output_group = QGroupBox("OUTPUT")
        output_layout = QVBoxLayout(output_group)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Separate output directory…")
        output_layout.addWidget(self.output_edit)
        output_buttons = QHBoxLayout()
        output_browse = QPushButton("Browse…")
        output_browse.clicked.connect(self.choose_output)
        output_open = QPushButton("Open output")
        output_open.clicked.connect(self.open_output)
        output_buttons.addWidget(output_browse)
        output_buttons.addWidget(output_open)
        output_layout.addLayout(output_buttons)
        left_layout.addWidget(output_group)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 0, 0, 0)
        current_header = QHBoxLayout()
        self.current_title = QLabel("No pitch selected")
        self.current_title.setObjectName("sectionTitle")
        current_header.addWidget(self.current_title)
        current_header.addStretch()
        self.analyse_button = QPushButton("Analyse checked")
        self.analyse_button.clicked.connect(self.analyse_checked)
        current_header.addWidget(self.analyse_button)
        right_layout.addLayout(current_header)

        self.metrics_label = QLabel("Analysis has not been run.")
        self.metrics_label.setObjectName("muted")
        self.metrics_label.setWordWrap(True)
        right_layout.addWidget(self.metrics_label)

        envelope_group = QGroupBox("MEDIAN SPECTRAL ENVELOPE  /  ANALYSED SAMPLES")
        envelope_layout = QVBoxLayout(envelope_group)
        self.envelope_plot = EnvelopePlot()
        envelope_layout.addWidget(self.envelope_plot)
        right_layout.addWidget(envelope_group)

        profile_group = QGroupBox("PITCH-SPECIFIC TONE PROFILE")
        profile_layout = QVBoxLayout(profile_group)
        hint = QLabel("Adjust each pitch independently. Zero gain means no change; use preview before processing.")
        hint.setObjectName("muted")
        profile_layout.addWidget(hint)
        self.eq_table = QTableWidget(5, 4)
        self.eq_table.setHorizontalHeaderLabels(["Band", "Centre (Hz)", "Gain (dB)", "Q"])
        self.eq_table.verticalHeader().setVisible(False)
        self.eq_table.horizontalHeader().setStretchLastSection(True)
        self.eq_table.setMaximumHeight(220)
        self.band_controls = []
        for row, default in enumerate(default_profile()["bands"]):
            name = QTableWidgetItem(default["name"])
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.eq_table.setItem(row, 0, name)
            frequency = QDoubleSpinBox()
            frequency.setRange(40.0, 18000.0)
            frequency.setDecimals(0)
            frequency.setValue(default["frequency_hz"])
            gain = QDoubleSpinBox()
            gain.setRange(-6.0, 6.0)
            gain.setDecimals(1)
            gain.setSingleStep(0.1)
            gain.setValue(0.0)
            q_value = QDoubleSpinBox()
            q_value.setRange(0.1, 5.0)
            q_value.setDecimals(2)
            q_value.setSingleStep(0.05)
            q_value.setValue(default["q"])
            for widget in (frequency, gain, q_value):
                widget.valueChanged.connect(self.save_current_profile)
            self.eq_table.setCellWidget(row, 1, frequency)
            self.eq_table.setCellWidget(row, 2, gain)
            self.eq_table.setCellWidget(row, 3, q_value)
            self.band_controls.append((frequency, gain, q_value))
        profile_layout.addWidget(self.eq_table)

        match_row = QHBoxLayout()
        match_row.addWidget(QLabel("Reference layer"))
        self.reference_combo = QComboBox()
        self.reference_combo.addItem("Analyse layers first")
        match_row.addWidget(self.reference_combo, 1)
        self.match_button = QPushButton("Suggest envelope match")
        self.match_button.clicked.connect(self.generate_match)
        match_row.addWidget(self.match_button)
        profile_layout.addLayout(match_row)
        profile_io_row = QHBoxLayout()
        load_profiles = QPushButton("Load profiles…")
        load_profiles.clicked.connect(self.load_profiles)
        save_profiles = QPushButton("Save profiles…")
        save_profiles.clicked.connect(self.save_profiles)
        profile_io_row.addWidget(load_profiles)
        profile_io_row.addWidget(save_profiles)
        profile_io_row.addStretch()
        profile_layout.addLayout(profile_io_row)
        right_layout.addWidget(profile_group)

        preview_group = QGroupBox("AUDITION  /  CURRENT SAMPLE")
        preview_layout = QGridLayout(preview_group)
        self.sample_combo = QComboBox()
        preview_layout.addWidget(QLabel("Sample"), 0, 0)
        preview_layout.addWidget(self.sample_combo, 0, 1, 1, 3)
        self.original_button = QPushButton("Play original")
        self.original_button.clicked.connect(lambda: self.play_preview(processed=False))
        self.processed_button = QPushButton("Play processed preview")
        self.processed_button.clicked.connect(lambda: self.play_preview(processed=True))
        preview_layout.addWidget(self.original_button, 1, 0, 1, 2)
        preview_layout.addWidget(self.processed_button, 1, 2, 1, 2)
        right_layout.addWidget(preview_group)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Wet blend"))
        self.wet_slider = QSlider(Qt.Orientation.Horizontal)
        self.wet_slider.setRange(0, 100)
        self.wet_slider.setValue(100)
        self.wet_value = QLabel("100%")
        self.wet_slider.valueChanged.connect(lambda value: self.wet_value.setText(f"{value}%"))
        controls.addWidget(self.wet_slider, 1)
        controls.addWidget(self.wet_value)
        self.oto_checkbox = QCheckBox("Protect fixed OTO regions (skip unmatched samples)")
        self.oto_checkbox.setChecked(False)
        controls.addWidget(self.oto_checkbox)
        right_layout.addLayout(controls)

        self.process_button = QPushButton("Process checked pitch folders →")
        self.process_button.setObjectName("primary")
        self.process_button.clicked.connect(self.process_checked)
        right_layout.addWidget(self.process_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        right_layout.addWidget(self.progress)
        splitter.addWidget(right)
        splitter.setSizes([310, 900])

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f3f2ee; color: #252927; font-size: 13px; }
            #pageTitle { font-size: 15px; font-weight: 700; letter-spacing: 2px; }
            #sectionTitle { font-size: 19px; font-weight: 600; }
            #muted { color: #777a74; }
            QGroupBox { border: 1px solid #d9d7d0; border-radius: 5px; margin-top: 10px; padding: 10px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLineEdit, QListWidget, QComboBox, QDoubleSpinBox, QTableWidget { background: #fbfaf7; border: 1px solid #d8d6cf; border-radius: 3px; padding: 5px; }
            QPushButton { background: #e8e6df; border: 1px solid #d1cfc7; border-radius: 3px; padding: 7px 11px; }
            QPushButton:hover { background: #dedcd4; }
            QPushButton:disabled { color: #a5a39c; }
            QPushButton#primary { background: #29312e; color: #faf9f5; border: 1px solid #29312e; font-weight: 600; padding: 10px; }
            QPushButton#primary:hover { background: #414b46; }
            QHeaderView::section { background: #eceae4; padding: 5px; border: 0; }
            QProgressBar { border: 1px solid #d8d6cf; border-radius: 3px; text-align: center; background: #fbfaf7; height: 12px; }
            QProgressBar::chunk { background: #59645e; }
            QCheckBox { spacing: 7px; }
        """)

    def choose_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select voicebank or pitch folder")
        if path:
            self.root_edit.setText(path)
            self.scan_layers()

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select separate output folder")
        if path:
            self.output_edit.setText(path)

    def open_output(self):
        path = Path(self.output_edit.text()).expanduser()
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def scan_layers(self):
        try:
            layers = discover_pitch_layers(self.root_edit.text())
        except Exception as exc:
            QMessageBox.warning(self, "Cannot scan folder", str(exc))
            return
        self.layers = layers
        self.layer_list.clear()
        self.profiles = {item["name"]: self.profiles.get(item["name"], default_profile(item["pitch"])) for item in layers}
        for index, layer in enumerate(layers):
            item = QListWidgetItem(f'{layer["name"]}   ·   {layer["wav_count"]} WAV')
            item.setData(Qt.ItemDataRole.UserRole, layer)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.layer_list.addItem(item)
        self.status_label.setText(f"{len(layers)} pitch layer(s) found.")
        if layers:
            self.layer_list.setCurrentRow(0)
            self.reference_combo.clear()
            self.reference_combo.addItem("Select reference after analysis…")
            self.reference_combo.addItems([item["pitch"] for item in layers])
        else:
            self.current_title.setText("No pitch folders detected")
            self.metrics_label.setText("Expected folder names containing a pitch token, e.g. '6. B2', '4. F3', '5. G4'.")

    def select_current_layer(self):
        item = self.layer_list.currentItem()
        if item:
            item.setCheckState(Qt.CheckState.Checked)

    def select_all_layers(self):
        for index in range(self.layer_list.count()):
            self.layer_list.item(index).setCheckState(Qt.CheckState.Checked)

    def clear_layer_selection(self):
        for index in range(self.layer_list.count()):
            self.layer_list.item(index).setCheckState(Qt.CheckState.Unchecked)

    def checked_layers(self) -> list[dict]:
        selected = []
        for index in range(self.layer_list.count()):
            item = self.layer_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.data(Qt.ItemDataRole.UserRole))
        return selected

    def current_layer(self) -> dict | None:
        item = self.layer_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def current_layer_changed(self, current, previous):
        layer = current.data(Qt.ItemDataRole.UserRole) if current else None
        if not layer:
            return
        name = layer["name"]
        self.current_title.setText(f'{layer["pitch"]}  /  {name}')
        self._load_profile(name)
        self._load_samples(layer)
        self._show_analysis(name)

    def _load_profile(self, name: str):
        profile = self.profiles.setdefault(name, default_profile(name))
        for row, controls in enumerate(self.band_controls):
            band = profile["bands"][row] if row < len(profile["bands"]) else DEFAULT_ZERO_BAND
            controls[0].blockSignals(True)
            controls[1].blockSignals(True)
            controls[2].blockSignals(True)
            controls[0].setValue(float(band["frequency_hz"]))
            controls[1].setValue(float(band["gain_db"]))
            controls[2].setValue(float(band.get("q", 0.7)))
            controls[0].blockSignals(False)
            controls[1].blockSignals(False)
            controls[2].blockSignals(False)

    def save_current_profile(self, *_):
        layer = self.current_layer()
        if not layer:
            return
        profile = default_profile(layer["pitch"])
        profile["bands"] = []
        for row, controls in enumerate(self.band_controls):
            label = self.eq_table.item(row, 0).text()
            frequency, gain, q_value = (control.value() for control in controls)
            profile["bands"].append({
                "name": label, "frequency_hz": frequency,
                "gain_db": gain, "q": q_value,
            })
        self.profiles[layer["name"]] = profile

    def _load_samples(self, layer: dict):
        self.sample_combo.clear()
        folder = Path(layer["path"])
        samples = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.casefold() == ".wav")
        for sample in samples:
            self.sample_combo.addItem(sample.relative_to(folder).as_posix(), str(sample))

    def _show_analysis(self, name: str):
        analysis = self.analyses.get(name)
        if not analysis:
            self.metrics_label.setText("Not analysed yet. Select the layer and run Analyse checked.")
            self.envelope_plot.set_data(None)
            return
        summary = analysis["summary"]
        self.metrics_label.setText(
            f'{analysis["analysed_count"]}/{analysis["sample_count"]} samples analysed  ·  '
            f'Median F0 {summary["f0_median_hz"] or 0:.1f} Hz  ·  '
            f'Centroid {summary["spectral_centroid_median_hz"] or 0:.0f} Hz  ·  '
            f'H/R proxy {summary["harmonic_to_residual_median_db"] or 0:.1f} dB  ·  '
            f'{analysis["failed_count"]} failed'
        )
        self.envelope_plot.set_data(analysis["spectral_envelope"])

    def _set_busy(self, busy: bool):
        for widget in (self.analyse_button, self.process_button):
            widget.setDisabled(busy)
        self.progress.setValue(0)
        self.status_label.setText("Working…" if busy else "Ready.")

    def _start_worker(self, task: Callable, on_complete: Callable):
        if self.worker and self.worker.isRunning():
            return
        self._set_busy(True)
        self.worker = TaskWorker(task, self)
        self.worker.progress.connect(self._progress_update)
        self.worker.completed.connect(lambda result: (self._set_busy(False), on_complete(result)))
        self.worker.failed.connect(self._task_failed)
        self.worker.start()

    def _progress_update(self, done: int, total: int, label: str):
        self.progress.setValue(int(done / max(total, 1) * 100))
        self.status_label.setText(f"{done}/{total}  ·  {label}")

    def _task_failed(self, message: str):
        self._set_busy(False)
        QMessageBox.critical(self, "Task failed", message)

    def analyse_checked(self):
        selected = self.checked_layers()
        if not selected:
            QMessageBox.information(self, "No pitch folders", "Scan a voicebank and select at least one pitch layer.")
            return
        total = max(1, sum(item["wav_count"] for item in selected))

        def task(progress):
            results, completed = {}, 0
            for layer in selected:
                def update(done, count, label, base=completed):
                    progress(base + done, total, f'{layer["pitch"]}: {label}')
                results[layer["name"]] = analyze_pitch_folder(layer["path"], progress=update)
                completed += layer["wav_count"]
            return results

        def complete(results):
            self.analyses.update(results)
            self.reference_combo.clear()
            self.reference_combo.addItem("Choose reference layer…")
            self.reference_combo.addItems([
                layer["pitch"] for layer in self.layers if layer["name"] in self.analyses
            ])
            layer = self.current_layer()
            if layer:
                self._show_analysis(layer["name"])
            self.status_label.setText(f'Analysis complete: {sum(x["analysed_count"] for x in results.values())} WAV(s).')

        self._start_worker(task, complete)

    def generate_match(self):
        layer = self.current_layer()
        if not layer:
            return
        current = self.analyses.get(layer["name"])
        if not current:
            QMessageBox.information(self, "Analyse first", "Run analysis for this layer and the reference layer first.")
            return
        reference_pitch = self.reference_combo.currentText()
        reference_layer = next((x for x in self.layers if x["pitch"] == reference_pitch), None)
        reference = self.analyses.get(reference_layer["name"]) if reference_layer else None
        if not reference:
            QMessageBox.information(self, "Reference unavailable", "Choose a reference pitch that has already been analysed.")
            return
        if current["pitch"] == reference["pitch"]:
            QMessageBox.information(self, "Same pitch", "Choose a different pitch as the reference.")
            return
        suggested = suggest_reference_match(current, reference, strength=0.6, max_gain_db=2.5)
        self.profiles[layer["name"]] = suggested
        self._load_profile(layer["name"])
        self.status_label.setText(
            f'Suggested a bounded tonal-balance match: {current["pitch"]} → {reference["pitch"]}. Audition before processing.'
        )

    def save_profiles(self):
        layer = self.current_layer()
        if layer:
            self.save_current_profile()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save pitch profiles", "eugene_pitch_profiles.json",
            "JSON profile files (*.json)",
        )
        if not path:
            return
        payload = {
            "format": "eugene_pitch_profiles",
            "version": 1,
            "voicebank_root": self.root_edit.text().strip(),
            "profiles": self.profiles,
        }
        try:
            Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            self.status_label.setText(f"Saved {len(self.profiles)} pitch profile(s).")
        except Exception as exc:
            QMessageBox.critical(self, "Could not save profiles", str(exc))

    def load_profiles(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load pitch profiles", "",
            "JSON profile files (*.json)",
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload.get("format") != "eugene_pitch_profiles":
                raise ValueError("This file is not an Eugene pitch-profile preset.")
            profiles = payload.get("profiles")
            if not isinstance(profiles, dict):
                raise ValueError("The profile file does not contain a profiles object.")
            from .pitch_studio import _to_profile
            checked = {}
            for name, spec in profiles.items():
                if not isinstance(spec, dict) or len(spec.get("bands", [])) != 5:
                    raise ValueError(f"Profile '{name}' must contain exactly five EQ bands.")
                _to_profile(spec, name=str(name))
                checked[str(name)] = spec
            self.profiles.update(checked)
            layer = self.current_layer()
            if layer and layer["name"] in self.profiles:
                self._load_profile(layer["name"])
            self.status_label.setText(f"Loaded {len(checked)} pitch profile(s).")
        except Exception as exc:
            QMessageBox.critical(self, "Could not load profiles", str(exc))

    def play_preview(self, processed: bool):
        layer = self.current_layer()
        if not layer or self.sample_combo.currentIndex() < 0:
            return
        path = Path(self.sample_combo.currentData())
        try:
            if processed:
                self.save_current_profile()
                audio, sample_rate, _ = preview_sample(
                    path, self.profiles[layer["name"]],
                    wet=self.wet_slider.value() / 100.0,
                    protect_oto=self.oto_checkbox.isChecked(),
                    source_root=Path(layer["path"]),
                )
            else:
                audio, sample_rate, _ = read_wav(path)
            if np.max(np.abs(audio)) > 1.0:
                raise ValueError("Preview would clip. Reduce EQ gain or wet blend.")
            target = Path(self.preview_directory.name) / ("processed.wav" if processed else "original.wav")
            sf.write(str(target), audio, sample_rate, subtype="PCM_16")
            self.player.stop()
            self.player.setSource(QUrl.fromLocalFile(str(target.resolve())))
            self.player.play()
            self.status_label.setText(("Processed preview: " if processed else "Original preview: ") + path.name)
        except Exception as exc:
            QMessageBox.warning(self, "Preview unavailable", str(exc))

    def process_checked(self):
        selected = self.checked_layers()
        if not selected:
            QMessageBox.information(self, "No pitch folders", "Select at least one pitch layer to process.")
            return
        output_root_text = self.output_edit.text().strip()
        if not output_root_text:
            QMessageBox.information(self, "Choose output", "Select a separate output directory first.")
            return
        output_root = Path(output_root_text).expanduser()
        for layer in selected:
            source = Path(layer["path"]).resolve()
            destination = (output_root / layer["name"]).resolve()
            if destination == source or source in destination.parents or destination in source.parents:
                QMessageBox.warning(self, "Unsafe output path", f"Output overlaps source folder: {source}")
                return

        profiles = {}
        for layer in selected:
            if self.current_layer() and layer["name"] == self.current_layer()["name"]:
                self.save_current_profile()
            profiles[layer["name"]] = self.profiles.get(layer["name"], default_profile(layer["pitch"]))
        wet = self.wet_slider.value() / 100.0
        protect = self.oto_checkbox.isChecked()
        total = max(1, sum(item["wav_count"] for item in selected))

        def task(progress):
            reports, completed = {}, 0
            for layer in selected:
                def update(done, count, label, base=completed):
                    progress(base + done, total, f'{layer["pitch"]}: {label}')
                reports[layer["name"]] = process_pitch_folder(
                    layer["path"], output_root, profiles[layer["name"]],
                    wet=wet, protect_oto=protect, progress=update,
                )
                completed += layer["wav_count"]
            return reports

        def complete(reports):
            processed = sum(report["summary"]["processed"] for report in reports.values())
            unchanged = sum(report["summary"]["unchanged"] for report in reports.values())
            skipped = sum(report["summary"]["skipped_oto"] for report in reports.values())
            failed = sum(report["summary"]["failed"] for report in reports.values())
            self.status_label.setText(
                f"Complete · {processed} changed · {unchanged} unchanged · {skipped} OTO-skipped · {failed} failed"
            )
            QMessageBox.information(
                self, "Pitch restoration complete",
                f"Processed only {len(reports)} selected pitch folder(s).\n\n"
                f"Changed WAVs: {processed}\nUnchanged: {unchanged}\n"
                f"Skipped by OTO protection: {skipped}\nFailed: {failed}\n\n"
                f"Output: {output_root}",
            )

        self._start_worker(task, complete)

    def closeEvent(self, event):
        self.player.stop()
        self.preview_directory.cleanup()
        super().closeEvent(event)


DEFAULT_ZERO_BAND = {
    "name": "Band", "frequency_hz": 1000.0, "gain_db": 0.0, "q": 0.7,
}


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Eugene:RE Pitch Restoration Studio")
    window = PitchStudioWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
