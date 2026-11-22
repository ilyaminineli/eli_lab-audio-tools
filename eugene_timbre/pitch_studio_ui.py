"""PySide6 interface for pitch-folder-first Eugene:RE restoration."""
from __future__ import annotations

import json
import math
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
    QTableWidgetItem, QVBoxLayout, QWidget, QTabWidget, QTreeWidget,
    QTreeWidgetItem, QScrollArea, QSizePolicy,
)

from .io import read_wav
from .moresampler import render_moresampler_ab
from .oto_analysis import OtoEntry, find_nearest_oto, match_entries, parse_oto
from .pitch_studio import (
    analyze_pitch_folder, default_profile, discover_pitch_layers,
    pitch_name, preview_sample, process_pitch_folder,
    inventory_voicebank,
)
from .recommendations import (
    get_eq_approach_labels, recommend_profiles, moresampler_flag_templates,
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frequencies: list[float] = []
        self.values: list[float] = []
        self.low: list[float] = []
        self.high: list[float] = []
        self.profile: dict = {}
        self.wet = 1.0
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, envelope: dict | None):
        envelope = envelope or {}
        self.frequencies = list(envelope.get("frequency_hz", []))
        self.values = list(envelope.get("median_relative_db", []))
        self.low = list(envelope.get("p10_relative_db", []))
        self.high = list(envelope.get("p90_relative_db", []))
        self.update()

    def set_profile(self, profile: dict | None, wet: float = 1.0):
        self.profile = dict(profile or {})
        self.wet = float(np.clip(wet, 0.0, 1.0))
        self.update()

    @staticmethod
    def _profile_curve(profile: dict, frequencies: np.ndarray) -> np.ndarray:
        """Calculate the plotted biquad/Butterworth response from current controls."""
        response = np.zeros_like(frequencies, dtype=np.float64)
        highpass = float(profile.get("highpass_hz", 0.0) or 0.0)
        order = int(profile.get("highpass_slope", 2) or 2)
        if highpass > 0:
            safe_f = np.maximum(frequencies, 1.0)
            response += -10.0 * np.log10(1.0 + (highpass / safe_f) ** (2 * order))
        sample_rate = 44100.0
        for band in profile.get("bands", []):
            centre = float(band.get("frequency_hz", 0.0) or 0.0)
            gain = float(band.get("gain_db", 0.0) or 0.0)
            q = float(band.get("q", 0.7) or 0.7)
            if centre <= 0 or abs(gain) < 1e-9 or centre >= sample_rate / 2:
                continue
            w0 = 2.0 * np.pi * centre / sample_rate
            alpha = np.sin(w0) / (2.0 * q)
            cosine = np.cos(w0)
            amplitude = 10.0 ** (gain / 40.0)
            b0, b1, b2 = 1 + alpha * amplitude, -2 * cosine, 1 - alpha * amplitude
            a0, a1, a2 = 1 + alpha / amplitude, -2 * cosine, 1 - alpha / amplitude
            omega = 2.0 * np.pi * frequencies / sample_rate
            z1 = np.exp(-1j * omega)
            z2 = z1 * z1
            numerator = b0 + b1 * z1 + b2 * z2
            denominator = a0 + a1 * z1 + a2 * z2
            response += 20.0 * np.log10(np.maximum(np.abs(numerator / denominator), 1e-9))
        return response

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fbfaf7"))
        left, right, top, bottom = 62, self.width() - 22, 24, self.height() - 42
        if right <= left or bottom <= top:
            return

        def x_for(hz):
            return left + (np.log10(max(float(hz), 100.0)) - 2.0) / 2.0 * (right - left)

        def y_for(db):
            value = float(np.clip(db, -12.0, 12.0))
            return bottom - (value + 12.0) / 24.0 * (bottom - top)

        painter.setPen(QPen(QColor("#e5e2db"), 1))
        for db in (-12, -6, 0, 6, 12):
            y = y_for(db)
            painter.drawLine(left, int(y), right, int(y))
            painter.setPen(QColor("#77746d"))
            painter.drawText(7, int(y + 4), f"{db:+d} dB")
            painter.setPen(QPen(QColor("#e5e2db"), 1))
        painter.setPen(QColor("#77746d"))
        for hz in (100, 300, 1000, 3000, 10000):
            x = x_for(hz)
            painter.drawText(int(x - 18), self.height() - 9, f"{hz:g}")
            painter.setPen(QPen(QColor("#eeeae3"), 1))
            painter.drawLine(int(x), top, int(x), bottom)
            painter.setPen(QColor("#77746d"))

        if len(self.frequencies) >= 2 and len(self.values) == len(self.frequencies):
            def envelope_point(index, value):
                return int(x_for(self.frequencies[index])), int(y_for(value))

            if len(self.low) == len(self.frequencies) and len(self.high) == len(self.frequencies):
                area = QPainterPath()
                area.moveTo(*envelope_point(0, self.high[0]))
                for i in range(1, len(self.frequencies)):
                    area.lineTo(*envelope_point(i, self.high[i]))
                for i in range(len(self.frequencies) - 1, -1, -1):
                    area.lineTo(*envelope_point(i, self.low[i]))
                area.closeSubpath()
                painter.fillPath(area, QColor(100, 110, 116, 28))
            line = QPainterPath()
            line.moveTo(*envelope_point(0, self.values[0]))
            for i in range(1, len(self.frequencies)):
                line.lineTo(*envelope_point(i, self.values[i]))
            painter.setPen(QPen(QColor("#252927"), 2))
            painter.drawPath(line)
        else:
            painter.setPen(QColor("#77746d"))
            painter.drawText(left + 12, top + 28, "Analyse a pitch layer to see its measured spectral envelope.")

        frequencies = np.geomspace(100.0, 10000.0, 144)
        response = self._profile_curve(self.profile, frequencies) * self.wet
        profile_line = QPainterPath()
        profile_line.moveTo(int(x_for(frequencies[0])), int(y_for(response[0])))
        for index in range(1, len(frequencies)):
            profile_line.lineTo(int(x_for(frequencies[index])), int(y_for(response[index])))
        painter.setPen(QPen(QColor("#e7795b"), 3))
        painter.drawPath(profile_line)

        painter.setPen(QColor("#252927"))
        painter.drawText(left + 4, 15, "MEDIAN SPECTRAL ENVELOPE")
        painter.setPen(QColor("#e7795b"))
        painter.drawText(right - 176, 15, "CURRENT EQ RESPONSE")


class PitchStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eugene:RE — Pitch Restoration Studio")
        self.resize(1480, 1040)
        self.setMinimumSize(1200, 820)
        self.inventory: dict | None = None
        self.layers: list[dict] = []
        self.profiles: dict[str, dict] = {}
        self.analyses: dict[str, dict] = {}
        self.profile_recommendations: dict[str, list[dict]] = {}
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
        left_layout.setSpacing(10)

        bank_group = QGroupBox("VOICEBANK ROOT")
        bank_layout = QVBoxLayout(bank_group)
        self.root_edit = QLineEdit(
            "C:/Users/photo/OneDrive/Документы/UtauV/Data/Singers/EugeneRE"
        )
        self.root_edit.setPlaceholderText("Choose the voicebank root…")
        bank_layout.addWidget(self.root_edit)
        root_buttons = QHBoxLayout()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.choose_root)
        scan = QPushButton("Scan whole bank")
        scan.clicked.connect(self.scan_layers)
        root_buttons.addWidget(browse)
        root_buttons.addWidget(scan)
        bank_layout.addLayout(root_buttons)
        left_layout.addWidget(bank_group)

        self.left_tabs = QTabWidget()
        pitch_page = QWidget()
        pitch_page_layout = QVBoxLayout(pitch_page)
        pitch_page_layout.setContentsMargins(7, 8, 7, 7)
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
        pitch_page_layout.addWidget(layer_group, 1)
        self.left_tabs.addTab(pitch_page, "PITCH LAYERS")

        bank_page = QWidget()
        bank_page_layout = QVBoxLayout(bank_page)
        bank_page_layout.setContentsMargins(7, 8, 7, 7)
        self.inventory_summary_label = QLabel("Run Scan whole bank to index all folders and files.")
        self.inventory_summary_label.setObjectName("muted")
        self.inventory_summary_label.setWordWrap(True)
        bank_page_layout.addWidget(self.inventory_summary_label)
        self.bank_tree = QTreeWidget()
        self.bank_tree.setHeaderLabels(["VOICEBANK HIERARCHY", "TYPE / CONTENT"])
        self.bank_tree.setColumnWidth(0, 240)
        self.bank_tree.setAlternatingRowColors(True)
        self.bank_tree.setUniformRowHeights(True)
        self.bank_tree.setExpandsOnDoubleClick(True)
        bank_page_layout.addWidget(self.bank_tree, 1)
        self.left_tabs.addTab(bank_page, "BANK STRUCTURE")
        left_layout.addWidget(self.left_tabs, 1)

        output_group = QGroupBox("OUTPUT")
        output_layout = QVBoxLayout(output_group)
        self.output_edit = QLineEdit("C:/EugeneRE_Restoration")
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
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(9)
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

        self.detail_tabs = QTabWidget()
        right_layout.addWidget(self.detail_tabs, 1)

        tone_scroll = QScrollArea()
        tone_scroll.setWidgetResizable(True)
        tone_page = QWidget()
        tone_layout = QVBoxLayout(tone_page)
        tone_layout.setContentsMargins(8, 8, 8, 8)
        tone_layout.setSpacing(10)

        envelope_group = QGroupBox("MEASURED ENVELOPE + LIVE EQ RESPONSE")
        envelope_layout = QVBoxLayout(envelope_group)
        self.envelope_plot = EnvelopePlot()
        envelope_layout.addWidget(self.envelope_plot, 1)
        tone_layout.addWidget(envelope_group, 2)

        profile_group = QGroupBox("PITCH-SPECIFIC FILTER / EQ")
        profile_layout = QVBoxLayout(profile_group)
        hint = QLabel(
            "The graph's orange line is the current filter response. "
            "The black line is the analysed median spectral envelope."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        profile_layout.addWidget(hint)
        self.profile_state_label = QLabel("NEUTRAL — no EQ or high-pass filter is active.")
        self.profile_state_label.setObjectName("profileState")
        self.profile_state_label.setWordWrap(True)
        profile_layout.addWidget(self.profile_state_label)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("High-pass (Hz)"))
        self.highpass_spin = QDoubleSpinBox()
        self.highpass_spin.setRange(0, 3000)
        self.highpass_spin.setDecimals(0)
        self.highpass_spin.setSingleStep(10)
        self.highpass_spin.setValue(0)
        self.highpass_spin.setSpecialValueText("OFF")
        filter_row.addWidget(self.highpass_spin)
        filter_row.addWidget(QLabel("Slope"))
        self.highpass_slope_combo = QComboBox()
        self.highpass_slope_combo.addItems(["12 dB/oct", "24 dB/oct", "36 dB/oct"])
        filter_row.addWidget(self.highpass_slope_combo)
        filter_row.addStretch()
        self.highpass_spin.valueChanged.connect(self.save_current_profile)
        self.highpass_slope_combo.currentIndexChanged.connect(self.save_current_profile)
        profile_layout.addLayout(filter_row)

        self.eq_table = QTableWidget(5, 4)
        self.eq_table.setHorizontalHeaderLabels(["Band", "Centre (Hz)", "Gain (dB)", "Q"])
        self.eq_table.verticalHeader().setVisible(False)
        self.eq_table.horizontalHeader().setStretchLastSection(False)
        self.eq_table.setMinimumHeight(260)
        self.eq_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.eq_table.verticalHeader().setDefaultSectionSize(40)
        self.eq_table.setColumnWidth(0, 170)
        self.eq_table.setColumnWidth(1, 150)
        self.eq_table.setColumnWidth(2, 140)
        self.eq_table.setColumnWidth(3, 110)
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
            gain.setRange(-12.0, 12.0)
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

        approach_row = QHBoxLayout()
        approach_row.addWidget(QLabel("Restoration approach"))
        self.approach_combo = QComboBox()
        for key, label in get_eq_approach_labels():
            self.approach_combo.addItem(label, key)
        self.approach_combo.setMinimumHeight(38)
        approach_row.addWidget(self.approach_combo, 1)
        apply_current = QPushButton("Apply to current")
        apply_current.clicked.connect(self.apply_recommended_current)
        apply_checked = QPushButton("Apply to checked")
        apply_checked.clicked.connect(self.apply_recommended_checked)
        approach_row.addWidget(apply_current)
        approach_row.addWidget(apply_checked)
        profile_layout.addLayout(approach_row)

        self.recommendation_hint = QLabel(
            "After analysis, choose an approach. Suggestions are generated from the measured layer; "
            "they remain previews until you apply one."
        )
        self.recommendation_hint.setObjectName("muted")
        self.recommendation_hint.setWordWrap(True)
        profile_layout.addWidget(self.recommendation_hint)

        match_row = QHBoxLayout()
        match_row.addWidget(QLabel("Reference pitch for envelope matching"))
        self.reference_combo = QComboBox()
        self.reference_combo.addItem("Analyse another pitch layer first")
        self.reference_combo.currentIndexChanged.connect(self.refresh_recommendation_hint)
        match_row.addWidget(self.reference_combo, 1)
        profile_layout.addLayout(match_row)
        self.approach_combo.currentIndexChanged.connect(self.refresh_recommendation_hint)
        profile_io_row = QHBoxLayout()
        reset_profile = QPushButton("Reset to neutral")
        reset_profile.clicked.connect(self.reset_profile)
        load_profiles = QPushButton("Load profiles…")
        load_profiles.clicked.connect(self.load_profiles)
        save_profiles = QPushButton("Save profiles…")
        save_profiles.clicked.connect(self.save_profiles)
        profile_io_row.addWidget(reset_profile)
        profile_io_row.addWidget(load_profiles)
        profile_io_row.addWidget(save_profiles)
        profile_io_row.addStretch()
        profile_layout.addLayout(profile_io_row)
        tone_layout.addWidget(profile_group, 2)

        preview_group = QGroupBox("AUDITION / CURRENT SAMPLE")
        preview_layout = QGridLayout(preview_group)
        self.sample_combo = QComboBox()
        self.sample_combo.setMinimumHeight(38)
        self.sample_combo.currentIndexChanged.connect(self._refresh_oto_aliases)
        preview_layout.addWidget(QLabel("Sample"), 0, 0)
        preview_layout.addWidget(self.sample_combo, 0, 1, 1, 3)
        self.original_button = QPushButton("Play original")
        self.original_button.clicked.connect(lambda: self.play_preview(processed=False))
        self.processed_button = QPushButton("Play processed preview")
        self.processed_button.clicked.connect(lambda: self.play_preview(processed=True))
        preview_layout.addWidget(self.original_button, 1, 0, 1, 2)
        preview_layout.addWidget(self.processed_button, 1, 2, 1, 2)
        self.preview_delta_label = QLabel("Choose a sample and preview it to measure the actual sample delta.")
        self.preview_delta_label.setObjectName("muted")
        self.preview_delta_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_delta_label, 2, 0, 1, 4)
        tone_layout.addWidget(preview_group)
        tone_layout.addStretch(1)
        tone_scroll.setWidget(tone_page)
        self.detail_tabs.addTab(tone_scroll, "ANALYSIS + EQ")

        more_scroll = QScrollArea()
        more_scroll.setWidgetResizable(True)
        more_page = QWidget()
        more_page_layout = QVBoxLayout(more_page)
        more_page_layout.setContentsMargins(8, 8, 8, 8)
        more_page_layout.setSpacing(12)
        moresampler_group = QGroupBox("MORESAMPLER / OTO PHONE A/B RENDER")
        moresampler_layout = QGridLayout(moresampler_group)
        moresampler_layout.setHorizontalSpacing(12)
        moresampler_layout.setVerticalSpacing(12)
        self.moresampler_edit = QLineEdit(
            "C:/Users/photo/OneDrive/Документы/UtauV/Data/Resamplers/moresampler.exe"
        )
        self.moresampler_edit.setPlaceholderText("Path to moresampler.exe…")
        moresampler_browse = QPushButton("Browse…")
        moresampler_browse.clicked.connect(self.choose_moresampler)
        moresampler_layout.addWidget(QLabel("Executable"), 0, 0)
        moresampler_layout.addWidget(self.moresampler_edit, 0, 1, 1, 3)
        moresampler_layout.addWidget(moresampler_browse, 0, 4)
        self.alias_combo = QComboBox()
        self.alias_combo.setMinimumWidth(280)
        self.alias_combo.setMinimumHeight(38)
        moresampler_layout.addWidget(QLabel("OTO alias"), 1, 0)
        moresampler_layout.addWidget(self.alias_combo, 1, 1, 1, 2)
        self.target_note_spin = QSpinBox()
        self.target_note_spin.setRange(24, 96)
        self.target_note_spin.setValue(60)
        self.target_note_spin.setMinimumHeight(38)
        moresampler_layout.addWidget(QLabel("Target MIDI note"), 1, 3)
        moresampler_layout.addWidget(self.target_note_spin, 1, 4)
        self.render_duration_spin = QSpinBox()
        self.render_duration_spin.setRange(100, 5000)
        self.render_duration_spin.setSingleStep(50)
        self.render_duration_spin.setValue(800)
        self.render_duration_spin.setMinimumHeight(38)
        self.render_flags_edit = QLineEdit()
        self.render_flags_edit.setPlaceholderText("Optional Moresampler flags, e.g. Mt+8Mb+6")
        self.render_flags_edit.setMinimumHeight(38)
        moresampler_layout.addWidget(QLabel("Duration (ms)"), 2, 0)
        moresampler_layout.addWidget(self.render_duration_spin, 2, 1)
        moresampler_layout.addWidget(QLabel("Renderer flags"), 2, 2)
        moresampler_layout.addWidget(self.render_flags_edit, 2, 3, 1, 2)

        self.moresampler_style_combo = QComboBox()
        self.moresampler_style_combo.setMinimumHeight(38)
        for template in moresampler_flag_templates():
            self.moresampler_style_combo.addItem(template["label"], template)
        self.moresampler_style_combo.currentIndexChanged.connect(self.refresh_moresampler_style_hint)
        moresampler_layout.addWidget(QLabel("Moresampler style template"), 3, 0)
        moresampler_layout.addWidget(self.moresampler_style_combo, 3, 1, 1, 3)
        apply_moresampler_style = QPushButton("Use template")
        apply_moresampler_style.clicked.connect(self.apply_moresampler_template)
        moresampler_layout.addWidget(apply_moresampler_style, 3, 4)
        self.moresampler_style_hint = QLabel(
            "Renderer flags are separate from source-WAV EQ. They affect Moresampler synthesis only."
        )
        self.moresampler_style_hint.setObjectName("muted")
        self.moresampler_style_hint.setWordWrap(True)
        moresampler_layout.addWidget(self.moresampler_style_hint, 4, 0, 1, 4)
        self.refresh_moresampler_style_hint()

        self.renderer_flags_compare_checkbox = QCheckBox(
            "A/B test selected Moresampler flags against neutral flags (renderer-only comparison)"
        )
        self.renderer_flags_compare_checkbox.setChecked(False)
        moresampler_layout.addWidget(self.renderer_flags_compare_checkbox, 5, 0, 1, 5)

        self.moresampler_hint = QLabel(
            "Select a pitch sample with a matching OTO alias. "
            "This control still renders one OTO phone, not a complete UST phrase."
        )
        self.moresampler_hint.setObjectName("muted")
        self.moresampler_hint.setWordWrap(True)
        moresampler_layout.addWidget(self.moresampler_hint, 6, 0, 1, 4)
        self.moresampler_button = QPushButton("Render original + restored through Moresampler →")
        self.moresampler_button.setMinimumHeight(46)
        self.moresampler_button.clicked.connect(self.render_moresampler)
        moresampler_layout.addWidget(self.moresampler_button, 6, 4)
        more_page_layout.addWidget(moresampler_group)
        more_page_layout.addStretch(1)
        more_scroll.setWidget(more_page)
        self.detail_tabs.addTab(more_scroll, "MORESAMPLER A/B")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Wet blend"))
        self.wet_slider = QSlider(Qt.Orientation.Horizontal)
        self.wet_slider.setRange(0, 100)
        self.wet_slider.setValue(100)
        self.wet_value = QLabel("100%")
        self.wet_slider.valueChanged.connect(lambda value: self.wet_value.setText(f"{value}%"))
        self.wet_slider.valueChanged.connect(self.update_processing_state)
        controls.addWidget(self.wet_slider, 1)
        controls.addWidget(self.wet_value)
        self.oto_checkbox = QCheckBox("Apply EQ only inside OTO stretchable body regions")
        self.oto_checkbox.setChecked(False)
        self.oto_checkbox.toggled.connect(self.update_processing_state)
        controls.addWidget(self.oto_checkbox)
        right_layout.addLayout(controls)

        self.processing_state_label = QLabel(
            "NO FILTER CHANGE — all EQ gains are 0 dB and high-pass is OFF. "
            "No mask is active; the full sample is passed through."
        )
        self.processing_state_label.setObjectName("processingState")
        self.processing_state_label.setWordWrap(True)
        right_layout.addWidget(self.processing_state_label)

        self.process_button = QPushButton("Process checked pitch folders →")
        self.process_button.setObjectName("primary")
        self.process_button.setMinimumHeight(48)
        self.process_button.clicked.connect(self.process_checked)
        right_layout.addWidget(self.process_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        right_layout.addWidget(self.progress)
        splitter.addWidget(right)
        splitter.setSizes([365, 1040])

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f3f2ee; color: #252927; font-size: 14px; }
            #pageTitle { font-size: 16px; font-weight: 700; letter-spacing: 2px; }
            #sectionTitle { font-size: 20px; font-weight: 600; }
            #muted { color: #777a74; }
            #profileState { font-size: 14px; font-weight: 700; color: #b45135; padding: 5px 2px; }
            #processingState { font-size: 13px; color: #39433e; padding: 5px 2px; }
            QGroupBox { border: 1px solid #d9d7d0; border-radius: 6px; margin-top: 12px; padding: 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
            QLineEdit, QListWidget, QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget, QTreeWidget {
                background: #fbfaf7; border: 1px solid #d8d6cf; border-radius: 4px; padding: 6px;
            }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox { min-height: 28px; }
            QTreeWidget::item { min-height: 25px; }
            QTreeWidget::item:selected, QListWidget::item:selected { background: #e5e1d8; color: #252927; }
            QPushButton { background: #e8e6df; border: 1px solid #d1cfc7; border-radius: 4px; padding: 9px 13px; min-height: 28px; }
            QPushButton:hover { background: #dedcd4; }
            QPushButton:disabled { color: #a5a39c; }
            QPushButton#primary { background: #29312e; color: #faf9f5; border: 1px solid #29312e; font-weight: 700; padding: 12px; }
            QPushButton#primary:hover { background: #414b46; }
            QHeaderView::section { background: #eceae4; padding: 8px; border: 0; }
            QTabWidget::pane { border: 1px solid #d9d7d0; background: #f3f2ee; }
            QTabBar::tab { background: #e8e6df; padding: 10px 16px; margin-right: 3px; border: 1px solid #d1cfc7; }
            QTabBar::tab:selected { background: #fbfaf7; }
            QProgressBar { border: 1px solid #d8d6cf; border-radius: 4px; text-align: center; background: #fbfaf7; height: 16px; }
            QProgressBar::chunk { background: #59645e; }
            QCheckBox { spacing: 8px; }
            QSlider::groove:horizontal { height: 7px; background: #d7d4cc; border-radius: 3px; }
            QSlider::handle:horizontal { background: #29312e; width: 17px; margin: -6px 0; border-radius: 8px; }
        """)

        if Path(self.root_edit.text()).expanduser().is_dir():
            self.scan_layers()

    def choose_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select voicebank or pitch folder")
        if path:
            self.root_edit.setText(path)
            self.scan_layers()

    def choose_moresampler(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Moresampler executable", "", "Moresampler executable (moresampler*.exe)"
        )
        if path:
            self.moresampler_edit.setText(path)

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
            root = Path(self.root_edit.text()).expanduser().resolve()
            inventory = inventory_voicebank(root)
            layers = discover_pitch_layers(root)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot scan voicebank", str(exc))
            return
        self.inventory = inventory
        self.layers = layers
        self.layer_list.clear()
        self.bank_tree.clear()
        self.profiles = {
            item["name"]: self.profiles.get(item["name"], default_profile(item["pitch"]))
            for item in layers
        }
        for layer in layers:
            item = QListWidgetItem(
                f'{layer["name"]}   ·   {layer["wav_count"]} WAV samples'
            )
            item.setData(Qt.ItemDataRole.UserRole, layer)
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(Qt.CheckState.Unchecked)
            self.layer_list.addItem(item)

        stats = inventory["stats"]
        self.inventory_summary_label.setText(
            f'{inventory["root_name"]}  ·  {stats["folders"]} folders  ·  '
            f'{stats["wav"]} WAV  ·  {stats["frq"]} FRQ  ·  {stats["llsm"]} LLSM  ·  '
            f'{stats["oto"]} oto.ini  ·  {inventory["pitch_layer_count"]} pitch layers  ·  '
            f'{inventory["addon_folder_count"]} add-on folders'
        )
        root_item = QTreeWidgetItem([
            inventory["root_name"],
            f'{stats["folders"]} folders / {stats["wav"]} WAV / {stats["oto"]} OTO',
        ])
        root_item.setToolTip(0, inventory["root_path"])
        self.bank_tree.addTopLevelItem(root_item)

        def append_nodes(parent_item, nodes, depth=0):
            for node in nodes:
                counts = node["stats"]
                if node["type"] == "directory":
                    details = (
                        f'{counts["wav"]} WAV · {counts["frq"]} FRQ · '
                        f'{counts["llsm"]} LLSM · {counts["oto"]} OTO'
                    )
                    label = node["name"]
                    if depth == 0:
                        label = f'{label}  [{node["category"]}]'
                    child_item = QTreeWidgetItem([label, details])
                    child_item.setToolTip(0, node["path"])
                    child_item.setData(0, Qt.ItemDataRole.UserRole, node)
                    parent_item.addChild(child_item)
                    append_nodes(child_item, node["children"], depth + 1)
                else:
                    child_item = QTreeWidgetItem([node["name"], node["category"]])
                    child_item.setToolTip(0, node["path"])
                    child_item.setData(0, Qt.ItemDataRole.UserRole, node)
                    parent_item.addChild(child_item)

        append_nodes(root_item, inventory["children"])
        root_item.setExpanded(True)
        for index in range(root_item.childCount()):
            child = root_item.child(index)
            node = child.data(0, Qt.ItemDataRole.UserRole)
            if node and node["category"] in {
                "Pitch layer", "R English subbank", "Consonant releases",
                "Bonus phonetic extras", "L phonetic extension",
                "V phonetic extension", "TH phonetic extension",
            }:
                child.setExpanded(False)

        self.status_label.setText(
            f'Scanned {stats["wav"]} WAVs across {stats["folders"]} folders; '
            f'{len(layers)} pitch layers ready for pitch-only processing.'
        )
        self.reference_combo.clear()
        self.reference_combo.addItem("Select reference after analysis…")
        self.reference_combo.addItems([item["pitch"] for item in layers])
        if layers:
            self.layer_list.setCurrentRow(0)
        else:
            self.current_title.setText("No pitch folders detected")
            self.metrics_label.setText(
                "No pitch folders were found. The BANK STRUCTURE tab still shows the full indexed hierarchy."
            )

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
        self.refresh_recommendation_hint()

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
        self.highpass_spin.blockSignals(True)
        self.highpass_slope_combo.blockSignals(True)
        self.highpass_spin.setValue(float(profile.get("highpass_hz", 0.0)))
        self.highpass_slope_combo.setCurrentIndex({2: 0, 4: 1, 6: 2}.get(int(profile.get("highpass_slope", 2)), 0))
        self.highpass_spin.blockSignals(False)
        self.highpass_slope_combo.blockSignals(False)
        self.update_processing_state()

    def save_current_profile(self, *_):
        layer = self.current_layer()
        if not layer:
            return
        previous = self.profiles.get(layer["name"], {})
        profile = default_profile(layer["pitch"])
        profile["description"] = previous.get("description", "Pitch-folder restoration profile.")
        for key in ("approach", "generated_from_analysis", "analysis_pitch",
                    "recommendation_reasons", "recommendation_limit",
                    "reference_pitch", "match_method", "match_strength", "max_gain_db"):
            if key in previous:
                profile[key] = previous[key]
        profile["highpass_hz"] = float(self.highpass_spin.value())
        profile["highpass_slope"] = (2, 4, 6)[self.highpass_slope_combo.currentIndex()]
        profile["bands"] = []
        for row, controls in enumerate(self.band_controls):
            label = self.eq_table.item(row, 0).text()
            frequency, gain, q_value = (control.value() for control in controls)
            profile["bands"].append({
                "name": label, "frequency_hz": frequency,
                "gain_db": gain, "q": q_value,
            })
        self.profiles[layer["name"]] = profile
        self.update_processing_state()

    @staticmethod
    def _profile_is_active(profile: dict | None) -> bool:
        if not profile:
            return False
        return (
            float(profile.get("highpass_hz", 0.0) or 0.0) > 0.0
            or any(abs(float(band.get("gain_db", 0.0) or 0.0)) > 1e-6 for band in profile.get("bands", []))
        )

    def update_processing_state(self, *_):
        layer = self.current_layer()
        profile = self.profiles.get(layer["name"]) if layer else default_profile()
        if not profile:
            profile = default_profile()
        gains = [float(band.get("gain_db", 0.0) or 0.0) for band in profile.get("bands", [])]
        highpass = float(profile.get("highpass_hz", 0.0) or 0.0)
        wet = self.wet_slider.value() / 100.0 if hasattr(self, "wet_slider") else 1.0
        mask_active = self.oto_checkbox.isChecked() if hasattr(self, "oto_checkbox") else False
        active = self._profile_is_active(profile) and wet > 0.0
        self.envelope_plot.set_profile(profile, wet=wet)
        if active:
            nonzero = [f'{gain:+.1f} dB @ {band["frequency_hz"]:.0f} Hz' for gain, band in zip(gains, profile.get("bands", [])) if abs(gain) > 1e-6]
            parts = nonzero[:3]
            if highpass > 0:
                slope = int(profile.get("highpass_slope", 2)) * 6
                parts.append(f"HPF {highpass:.0f} Hz / {slope} dB/oct")
            description = ", ".join(parts) if parts else "High-pass filter active"
            self.profile_state_label.setText(f"ACTIVE FILTER PROFILE  ·  {description}")
            self.profile_state_label.setStyleSheet("color: #376b53; font-weight: 700;")
        else:
            reason = "Wet blend is 0%" if wet <= 0 else "all EQ gains are 0 dB and high-pass is OFF"
            self.profile_state_label.setText(f"NEUTRAL — {reason}. The processed preview should null against the original.")
            self.profile_state_label.setStyleSheet("color: #b45135; font-weight: 700;")
        mask_text = (
            "OTO mask ON — only matched stretchable body regions are eligible; unmatched files are skipped."
            if mask_active else
            "OTO mask OFF — the filter is applied to the full selected WAV."
        )
        state_text = (
            f"{'FILTER ACTIVE' if active else 'NO FILTER CHANGE'}  ·  Wet {self.wet_slider.value()}%  ·  {mask_text}"
        )
        if not active:
            state_text += "  EQ is neutral until you edit a gain, set a high-pass frequency, or use Suggest envelope match."
        self.processing_state_label.setText(state_text)

    def reset_profile(self):
        layer = self.current_layer()
        if not layer:
            return
        self.profiles[layer["name"]] = default_profile(layer["pitch"])
        self._load_profile(layer["name"])
        self.status_label.setText(f'{layer["pitch"]}: reset to neutral; no filters are active.')

    def _load_samples(self, layer: dict):
        self.sample_combo.clear()
        folder = Path(layer["path"])
        samples = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.casefold() == ".wav")
        for sample in samples:
            self.sample_combo.addItem(sample.relative_to(folder).as_posix(), str(sample))

    def _refresh_oto_aliases(self, *_):
        self.alias_combo.clear()
        layer = self.current_layer()
        if not layer or self.sample_combo.currentIndex() < 0:
            self.moresampler_button.setEnabled(False)
            return
        wav_path = Path(self.sample_combo.currentData())
        try:
            oto_path = find_nearest_oto(wav_path, Path(layer["path"]))
            if oto_path is None:
                raise FileNotFoundError("No oto.ini found in this pitch folder.")
            parsed = parse_oto(oto_path)
            try:
                relative = wav_path.relative_to(oto_path.parent).as_posix()
            except ValueError:
                relative = wav_path.name
            entries = match_entries(parsed.index(), relative, wav_path.name)
            for entry in entries:
                self.alias_combo.addItem(f"{entry.alias}  ·  line {entry.line_number}", entry)
            self.moresampler_button.setEnabled(bool(entries))
            self.moresampler_hint.setText(
                f"{len(entries)} OTO alias(es) available from {oto_path.name}. "
                "Moresampler will receive the selected alias's timing values."
                if entries else "No matching OTO alias was found for this WAV."
            )
            self._suggest_target_note(layer, wav_path)
        except Exception as exc:
            self.moresampler_button.setEnabled(False)
            self.moresampler_hint.setText(str(exc))

    def _suggest_target_note(self, layer: dict, wav_path: Path):
        analysis = self.analyses.get(layer["name"])
        if not analysis:
            return
        relative = wav_path.relative_to(Path(layer["path"])).as_posix()
        row = next((item for item in analysis.get("files", []) if item.get("file") == relative), None)
        f0 = row.get("f0_median_hz") if row else analysis.get("summary", {}).get("f0_median_hz")
        if isinstance(f0, (int, float)) and f0 > 0:
            midi_note = int(round(69 + 12 * math.log2(float(f0) / 440.0)))
            self.target_note_spin.setValue(max(24, min(96, midi_note)))

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
        self.update_processing_state()
        self.refresh_recommendation_hint()
        if self.current_layer() and self.sample_combo.currentIndex() >= 0:
            self._suggest_target_note(self.current_layer(), Path(self.sample_combo.currentData()))

    def _set_busy(self, busy: bool):
        for widget in (self.analyse_button, self.process_button, self.moresampler_button):
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
            for name, report in results.items():
                self.profile_recommendations[name] = recommend_profiles(report)
            self.reference_combo.clear()
            self.reference_combo.addItem("Choose reference layer…")
            self.reference_combo.addItems([
                layer["pitch"] for layer in self.layers if layer["name"] in self.analyses
            ])
            layer = self.current_layer()
            if layer:
                self._show_analysis(layer["name"])
            self.refresh_recommendation_hint()
            self.status_label.setText(
                f'Analysis complete: {sum(x["analysed_count"] for x in results.values())} WAV(s); '
                f'{len(results) * 4} non-neutral EQ starting points generated.'
            )

        self._start_worker(task, complete)

    def _reference_analysis(self, current_layer: dict) -> dict | None:
        reference_pitch = self.reference_combo.currentData()
        if not reference_pitch:
            reference_pitch = self.reference_combo.currentText()
        reference_layer = next(
            (item for item in self.layers if item["pitch"] == reference_pitch),
            None,
        )
        if not reference_layer or reference_layer["name"] == current_layer["name"]:
            return None
        return self.analyses.get(reference_layer["name"])

    def _recommendation_for(self, layer: dict, approach: str) -> dict | None:
        analysis = self.analyses.get(layer["name"])
        if not analysis:
            return None
        reference = self._reference_analysis(layer) if approach == "reference_match" else None
        recommendations = recommend_profiles(analysis, reference)
        self.profile_recommendations[layer["name"]] = recommendations
        return next((item for item in recommendations if item["key"] == approach), None)

    def refresh_recommendation_hint(self, *_):
        layer = self.current_layer()
        if not layer:
            self.recommendation_hint.setText("Analyse a pitch layer to generate profile suggestions.")
            return
        analysis = self.analyses.get(layer["name"])
        if not analysis:
            self.recommendation_hint.setText("Analyse this pitch layer first. Analysis will generate neutral, clarity, warmth, air and soft suggestions automatically.")
            return
        approach = self.approach_combo.currentData()
        recommendation = self._recommendation_for(layer, approach)
        if not recommendation or not recommendation.get("available"):
            reason = recommendation.get("reason") if recommendation else "Select an approach."
            self.recommendation_hint.setText(reason or "Select an approach.")
            return
        profile = recommendation["profile"]
        gains = [
            f'{band["name"]} {band["gain_db"]:+.2f} dB'
            for band in profile["bands"] if abs(float(band["gain_db"])) > 0.01
        ]
        gain_text = ", ".join(gains) if gains else "neutral / no change"
        reasons = " ".join(profile.get("recommendation_reasons", []))
        self.recommendation_hint.setText(
            f'{recommendation["label"]}: {gain_text}. {reasons} '
            "Apply it to audition/export; generation itself does not modify WAVs."
        )

    def apply_recommended_current(self):
        layer = self.current_layer()
        if not layer:
            return
        approach = self.approach_combo.currentData()
        recommendation = self._recommendation_for(layer, approach)
        if not recommendation or not recommendation.get("available"):
            QMessageBox.information(
                self, "Recommendation unavailable",
                recommendation.get("reason") if recommendation else "Analyse this pitch layer first.",
            )
            return
        self.profiles[layer["name"]] = dict(recommendation["profile"])
        self._load_profile(layer["name"])
        self.status_label.setText(
            f'{recommendation["label"]} applied to the {layer["pitch"]} pitch profile. '
            "Audition before processing."
        )

    def apply_recommended_checked(self):
        selected = self.checked_layers()
        if not selected:
            QMessageBox.information(self, "No pitch layers checked", "Check the pitch folders that should receive this approach.")
            return
        approach = self.approach_combo.currentData()
        applied, missing, unavailable = [], [], []
        for layer in selected:
            if layer["name"] not in self.analyses:
                missing.append(layer["pitch"])
                continue
            recommendation = self._recommendation_for(layer, approach)
            if not recommendation or not recommendation.get("available"):
                unavailable.append(layer["pitch"])
                continue
            self.profiles[layer["name"]] = dict(recommendation["profile"])
            applied.append(layer)
        current = self.current_layer()
        if current and any(item["name"] == current["name"] for item in applied):
            self._load_profile(current["name"])
        parts = [f"Applied {approach} to {len(applied)} checked pitch layer(s)."]
        if missing:
            parts.append("Not analysed: " + ", ".join(missing) + ".")
        if unavailable:
            parts.append("Reference/template unavailable: " + ", ".join(unavailable) + ".")
        parts.append("Next, audition the selected pitch profiles and press Process checked pitch folders when ready.")
        message = " ".join(parts)
        self.status_label.setText(message)
        QMessageBox.information(self, "Pitch profiles generated", message)

    def refresh_moresampler_style_hint(self, *_):
        template = self.moresampler_style_combo.currentData()
        if not template:
            return
        self.moresampler_style_hint.setText(
            f'{template["description"]}  {template["caution"]}'
        )

    def apply_moresampler_template(self):
        template = self.moresampler_style_combo.currentData()
        if not template:
            return
        self.render_flags_edit.setText(template["flags"])
        self.moresampler_style_hint.setText(
            f'Loaded flags "{template["flags"] or "(none)"}". {template["description"]}  {template["caution"]}'
        )
        self.detail_tabs.setCurrentIndex(1)

    def generate_match(self):
        self.approach_combo.setCurrentIndex(
            max(0, self.approach_combo.findData("reference_match"))
        )
        self.apply_recommended_current()

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
            original_audio, original_sr, _ = read_wav(path)
            if processed:
                self.save_current_profile()
                audio, sample_rate, _ = preview_sample(
                    path, self.profiles[layer["name"]],
                    wet=self.wet_slider.value() / 100.0,
                    protect_oto=self.oto_checkbox.isChecked(),
                    source_root=Path(layer["path"]),
                )
                if sample_rate != original_sr or audio.shape != original_audio.shape:
                    raise ValueError("Processed preview changed sample rate or channel/frame shape.")
                delta = np.asarray(audio, dtype=np.float64) - np.asarray(original_audio, dtype=np.float64)
                peak_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
                changed_fraction = float(np.mean(np.abs(delta) > 1e-7)) if delta.size else 0.0
                rms_delta = float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0
                if peak_delta <= 1e-7:
                    self.preview_delta_label.setText(
                        "NO SAMPLE DIFFERENCE — current settings are neutral, wet is 0%, or the OTO mask contains no changed region."
                    )
                else:
                    self.preview_delta_label.setText(
                        f"Measured preview delta: peak {peak_delta:.6f}  ·  "
                        f"changed samples {changed_fraction * 100:.1f}%  ·  "
                        f"delta RMS {20 * np.log10(max(rms_delta, 1e-12)):.1f} dBFS"
                    )
            else:
                audio, sample_rate = original_audio, original_sr
                self.preview_delta_label.setText("Playing original source WAV. Press Play processed preview to measure the DSP delta.")
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

    def render_moresampler(self):
        layer = self.current_layer()
        if not layer or self.sample_combo.currentIndex() < 0:
            QMessageBox.information(self, "Select a sample", "Choose a pitch layer and a sample first.")
            return
        executable_text = self.moresampler_edit.text().strip()
        if not executable_text:
            QMessageBox.information(self, "Choose Moresampler", "Select your moresampler.exe first.")
            return
        oto = self.alias_combo.currentData()
        if not isinstance(oto, OtoEntry):
            QMessageBox.information(self, "No OTO alias", "Choose a sample with a matching OTO alias.")
            return
        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.information(self, "Choose output", "Select a separate output directory first.")
            return
        layer_analysis = self.analyses.get(layer["name"])
        if not layer_analysis:
            QMessageBox.information(self, "Analyse first", "Run Analyse checked for this pitch layer before Moresampler testing.")
            return
        wav_path = Path(self.sample_combo.currentData()).resolve()
        relative = wav_path.relative_to(Path(layer["path"])).as_posix()
        sample_analysis = next(
            (item for item in layer_analysis.get("files", []) if item.get("file") == relative),
            None,
        )
        reference_f0 = (
            sample_analysis.get("f0_median_hz") if sample_analysis
            else layer_analysis.get("summary", {}).get("f0_median_hz")
        )
        if not isinstance(reference_f0, (int, float)) or reference_f0 <= 0:
            QMessageBox.warning(self, "No measured F0", "This pitch layer has no usable median F0 for Moresampler analysis.")
            return

        try:
            output_root = Path(output_text).expanduser().resolve()
            source_root = Path(self.root_edit.text()).expanduser().resolve()
            if output_root == source_root or source_root in output_root.parents or output_root in source_root.parents:
                raise ValueError("Moresampler test output must be outside the source voicebank.")
            self.save_current_profile()
            profile = dict(self.profiles[layer["name"]])
            wet = self.wet_slider.value() / 100.0
            compare_renderer_flags = self.renderer_flags_compare_checkbox.isChecked()
            if compare_renderer_flags:
                if not flags:
                    raise ValueError("Choose a non-neutral Moresampler style template or enter renderer flags first.")
            elif wet <= 0 or not self._profile_is_active(profile):
                raise ValueError("The current EQ profile is neutral (or Wet is 0%). Change an EQ gain/high-pass setting for a source-WAV A/B, or enable renderer-only flag comparison.")
            protect_oto = self.oto_checkbox.isChecked()
            executable = Path(executable_text).expanduser().resolve()
            target_note = self.target_note_spin.value()
            duration_ms = float(self.render_duration_spin.value())
            flags = self.render_flags_edit.text().strip()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid render settings", str(exc))
            return

        def task(progress):
            progress(0, 2, f"{layer['pitch']}: preparing A/B source")
            if compare_renderer_flags:
                restored_audio, sr, metadata = read_wav(wav_path)
            else:
                restored_audio, sr, _ = preview_sample(
                    wav_path, profile, wet=wet, protect_oto=protect_oto,
                    source_root=Path(layer["path"]),
                )
                _, _, metadata = read_wav(wav_path)
            progress(1, 2, f"{layer['pitch']}: rendering through Moresampler")
            report = render_moresampler_ab(
                source_wav=wav_path,
                restored_audio=restored_audio,
                sample_rate=sr,
                subtype=metadata["subtype"],
                oto=oto,
                executable=executable,
                output_root=output_root,
                source_bank_root=source_root,
                pitch_layer=layer["name"],
                reference_f0_hz=float(reference_f0),
                target_note=target_note,
                duration_ms=duration_ms,
                velocity=100,
                flags=flags,
                original_flags="" if compare_renderer_flags else None,
                volume=100,
                modulation=0,
                tempo=120.0,
            )
            progress(2, 2, "Moresampler A/B render complete")
            return report

        def complete(report):
            output_folder = Path(report["output_folder"])
            self.status_label.setText(f"Moresampler A/B render saved: {output_folder}")
            QMessageBox.information(
                self,
                "Moresampler comparison complete",
                (
                    "Renderer-only comparison: A used neutral flags and B used the selected Moresampler flags.\n\n"
                    if compare_renderer_flags else
                    "Both versions were rendered with the same OTO timing, note, duration and flags.\n\n"
                )
                f"Original: {report['outputs']['original']}\n"
                f"Restored: {report['outputs']['restored']}\n\n"
                f"Output: {output_folder}\n\n"
                "Listen to the two rendered phone samples. The report's spectral measurements are diagnostics, not a naturalness score.",
            )
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_folder.resolve())))

        self._start_worker(task, complete)

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
        if wet <= 0 or not any(self._profile_is_active(profiles[name]) for name in profiles):
            QMessageBox.information(
                self, "Neutral profile",
                "All selected pitch profiles are neutral (0 dB EQ, high-pass OFF), or Wet is 0%. No WAV samples would change. Edit the filters or use Suggest envelope match first."
            )
            return
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
