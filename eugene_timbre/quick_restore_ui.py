"""Simple one-pass Eugene:RE voicebank restoration interface.

The interface intentionally exposes only bank selection, preview, and export.
All routing/OTO safety decisions stay in processing_chain.py.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal, Qt, QUrl, QTimer
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QComboBox, QPushButton,
    QProgressBar, QVBoxLayout, QWidget,
)

from .auto_routing import identify_family
from .dsp import apply_profile_masked
from .io import read_wav
from .oto_analysis import match_entries, parse_oto
from .processing_chain import (
    _alias_audit,
    _match_oto,
    _profile,
    _route,
    build_oto_body_mask,
    load_processing_config,
    process_auto,
)


DEFAULT_INPUT = "C:/Users/photo/OneDrive/Документы/UtauV/Data/Singers/EugeneRE"
DEFAULT_OUTPUT = "C:/EugeneRE_Restoration"
DEFAULT_WET = 0.5


class Worker(QThread):
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


def prepare_preview(root: Path, relative_path: str, wet: float = DEFAULT_WET) -> dict:
    """Create an OTO-aware before/after preview for exactly one source WAV."""
    root = root.expanduser().resolve()
    source = (root / Path(relative_path)).resolve()
    if root not in source.parents or not source.is_file():
        raise ValueError("Preview sample is not inside the selected voicebank.")

    audio, sample_rate, metadata = read_wav(source)
    config = load_processing_config()
    family = identify_family(Path(relative_path), root)
    oto_path, entries, parsed = _match_oto(source, root, None, {})
    route = _route(family, source.name, entries, config)

    unchanged = audio.astype(np.float32, copy=True)
    output = unchanged
    status = "bypassed"
    reason = route.get("reason", "No processing route is configured.")
    mask_info = {"mask_coverage_percent": 0.0, "eligible_stretch_regions": 0}
    profile_description = None

    if route.get("enabled"):
        audit = _alias_audit(entries)
        if not entries:
            status, reason = "manual_review", "No matching OTO alias; this sample is not modified."
        elif audit["has_duplicate_aliases"]:
            status, reason = "manual_review", "Duplicate OTO aliases; this sample is not modified."
        elif audit["has_unclassified_tokens"]:
            status, reason = "manual_review", "Unclassified OTO alias tokens; this sample is not modified."
        else:
            mask_options = config.get("oto_mask", {})
            mask, mask_info = build_oto_body_mask(
                len(audio), sample_rate, entries, mask_options
            )
            minimum_coverage = float(mask_options.get("minimum_process_coverage_percent", 5.0))
            if mask_info.get("timing_warnings"):
                status, reason = "manual_review", "OTO timing warnings; this sample is not modified."
            elif not np.any(mask > 0):
                status, reason = "manual_review", "No safe stretchable OTO body region; sample is not modified."
            elif mask_info["mask_coverage_percent"] < minimum_coverage:
                status, reason = "partial", (
                    f"Only {mask_info['mask_coverage_percent']:.2f}% of the WAV is eligible; "
                    "below the safe processing threshold."
                )
            else:
                spec = route.get("profile_spec", {})
                profile = _profile(spec, route.get("profile_name", "profile"))
                if profile.bands or profile.highpass_hz > 0:
                    output = apply_profile_masked(
                        audio, sample_rate, profile, mask,
                        wet=wet,
                        context_padding_ms=float(mask_options.get("context_padding_ms", 80.0)),
                    )
                    status = "preview_ready"
                    reason = (
                        f"Preview uses {route.get('profile_name')} inside "
                        f"{mask_info['mask_coverage_percent']:.1f}% OTO-approved body regions."
                    )
                    profile_description = {
                        "name": profile.name,
                        "bands": [
                            {"frequency_hz": b.frequency_hz, "gain_db": b.gain_db, "q": b.q}
                            for b in profile.bands
                        ],
                        "highpass_hz": profile.highpass_hz,
                    }
                else:
                    status, reason = "bypassed", "This family is configured for neutral pass-through."

    return {
        "source": str(source),
        "relative_path": relative_path,
        "sample_rate": sample_rate,
        "subtype": metadata["subtype"],
        "original": unchanged,
        "processed": np.asarray(output, dtype=np.float32),
        "status": status,
        "reason": reason,
        "profile": profile_description,
        "mask": mask_info,
        "oto_path": str(oto_path) if oto_path else None,
        "oto_parse_errors": len(parsed.errors) if parsed else 0,
        "same_audio": np.array_equal(unchanged, output),
    }


class QuickRestoreWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eugene:RE — Quick Voicebank Restore")
        self.resize(960, 700)
        self.setMinimumSize(800, 600)

        self.scan_report: dict | None = None
        self.preview_data: dict | None = None
        self.worker: Worker | None = None
        self.preview_directory = tempfile.TemporaryDirectory(prefix="eugene_quick_preview_")

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.9)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title = QLabel("EUGENE:RE  /  VOICEBANK PREPARATION")
        title.setObjectName("title")
        subtitle = QLabel(
            "Choose the bank → listen to a preview → create a replacement-ready copy."
        )
        subtitle.setObjectName("muted")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        paths = QGroupBox("VOICEBANK")
        path_grid = QGridLayout(paths)
        self.root_edit = QLineEdit(DEFAULT_INPUT)
        self.output_edit = QLineEdit(DEFAULT_OUTPUT)
        self.root_browse = QPushButton("Browse…")
        self.output_browse = QPushButton("Browse…")
        self.root_browse.clicked.connect(self.choose_root)
        self.output_browse.clicked.connect(self.choose_output)
        path_grid.addWidget(QLabel("Input bank"), 0, 0)
        path_grid.addWidget(self.root_edit, 0, 1)
        path_grid.addWidget(self.root_browse, 0, 2)
        path_grid.addWidget(QLabel("Output copy"), 1, 0)
        path_grid.addWidget(self.output_edit, 1, 1)
        path_grid.addWidget(self.output_browse, 1, 2)
        layout.addWidget(paths)

        self.scan_button = QPushButton("1. Scan bank and prepare preview")
        self.scan_button.setMinimumHeight(44)
        self.scan_button.clicked.connect(self.scan_bank)
        layout.addWidget(self.scan_button)

        self.summary_group = QGroupBox("SCAN RESULT")
        summary_layout = QVBoxLayout(self.summary_group)
        self.summary_label = QLabel(
            "The bank has not been scanned yet. The scan reads WAVs and OTO metadata without modifying the source."
        )
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("summary")
        summary_layout.addWidget(self.summary_label)
        self.sample_combo = QComboBox()
        self.sample_combo.setMinimumHeight(38)
        self.sample_combo.currentIndexChanged.connect(self.change_preview_sample)
        summary_layout.addWidget(QLabel("Preview sample"))
        summary_layout.addWidget(self.sample_combo)
        self.preview_note = QLabel(
            "The preview will use the same OTO-aware processing rules as the full-bank export."
        )
        self.preview_note.setWordWrap(True)
        self.preview_note.setObjectName("muted")
        summary_layout.addWidget(self.preview_note)
        layout.addWidget(self.summary_group, 1)

        audition = QHBoxLayout()
        self.play_original_button = QPushButton("Play original")
        self.play_processed_button = QPushButton("Play processed preview")
        self.play_original_button.setMinimumHeight(40)
        self.play_processed_button.setMinimumHeight(40)
        self.play_original_button.clicked.connect(lambda: self.play_audio("original"))
        self.play_processed_button.clicked.connect(lambda: self.play_audio("processed"))
        self.play_original_button.setEnabled(False)
        self.play_processed_button.setEnabled(False)
        audition.addWidget(self.play_original_button)
        audition.addWidget(self.play_processed_button)
        layout.addLayout(audition)

        self.export_button = QPushButton("2. Create processed voicebank copy →")
        self.export_button.setObjectName("primary")
        self.export_button.setMinimumHeight(54)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_bank)
        layout.addWidget(self.export_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        footer = QHBoxLayout()
        self.status_label = QLabel("Original voicebank remains untouched.")
        self.status_label.setObjectName("muted")
        footer.addWidget(self.status_label, 1)
        self.open_output_button = QPushButton("Open output folder")
        self.open_output_button.clicked.connect(self.open_output)
        footer.addWidget(self.open_output_button)
        layout.addLayout(footer)

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f3f2ee; color: #252927; font-size: 14px; }
            #title { font-size: 19px; font-weight: 700; letter-spacing: 2px; }
            #muted { color: #73766f; }
            #summary { font-size: 14px; }
            QGroupBox { border: 1px solid #d8d6cf; border-radius: 6px; margin-top: 12px; padding: 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
            QLineEdit, QComboBox { background: #fbfaf7; border: 1px solid #d8d6cf; border-radius: 4px; padding: 8px; min-height: 28px; }
            QPushButton { background: #e8e6df; border: 1px solid #d1cfc7; border-radius: 4px; padding: 10px 14px; min-height: 26px; }
            QPushButton:hover { background: #dedcd4; }
            QPushButton:disabled { color: #a5a39c; }
            QPushButton#primary { background: #29312e; color: #faf9f5; border: 1px solid #29312e; font-weight: 700; padding: 14px; }
            QPushButton#primary:hover { background: #414b46; }
            QProgressBar { border: 1px solid #d8d6cf; border-radius: 4px; text-align: center; background: #fbfaf7; height: 17px; }
            QProgressBar::chunk { background: #59645e; }
        """)

        if Path(self.root_edit.text()).expanduser().is_dir():
            QTimer.singleShot(350, self.scan_bank)

    def choose_root(self):
        selected = QFileDialog.getExistingDirectory(
            self, "Choose Eugene:RE voicebank", self.root_edit.text()
        )
        if selected:
            self.root_edit.setText(selected)
            self.scan_bank()

    def choose_output(self):
        selected = QFileDialog.getExistingDirectory(
            self, "Choose output voicebank folder", self.output_edit.text()
        )
        if selected:
            self.output_edit.setText(selected)

    def _set_busy(self, busy: bool, label: str):
        self.scan_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy and self.scan_report is not None)
        self.root_browse.setEnabled(not busy)
        self.output_browse.setEnabled(not busy)
        self.sample_combo.setEnabled(not busy)
        self.status_label.setText(label)

    def _start_worker(self, task, completed):
        if self.worker and self.worker.isRunning():
            return
        self.worker = Worker(task, self)
        self.worker.progress.connect(self.update_progress)
        self.worker.failed.connect(self.worker_failed)
        self.worker.completed.connect(completed)
        self.worker.start()

    def update_progress(self, done: int, total: int, label: str):
        total = max(1, total)
        self.progress.setValue(int(100 * done / total))
        self.status_label.setText(label)

    def worker_failed(self, message: str):
        self._set_busy(False, "Operation failed.")
        QMessageBox.critical(self, "Voicebank operation failed", message)

    def scan_bank(self):
        if self.worker and self.worker.isRunning():
            return
        root_text = self.root_edit.text().strip()
        if not root_text:
            QMessageBox.information(self, "Choose voicebank", "Select your existing voicebank folder.")
            return
        root = Path(root_text).expanduser().resolve()
        if not root.is_dir():
            QMessageBox.warning(self, "Voicebank not found", str(root))
            return
        self.scan_report = None
        self.preview_data = None
        self.sample_combo.clear()
        self.play_original_button.setEnabled(False)
        self.play_processed_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress.setValue(0)
        self._set_busy(True, "Scanning WAVs, OTO entries and safe processing regions…")

        def task(progress):
            with tempfile.TemporaryDirectory(prefix="eugene_bank_preflight_") as temporary:
                report = process_auto(
                    root, Path(temporary) / "preflight",
                    wet=DEFAULT_WET, dry_run=True, progress=progress,
                )
            ready = [
                item for item in report["files"]
                if item["status"] == "ready" and item["family"] == "pitch_layer"
            ]
            candidates = ready or [
                item for item in report["files"]
                if item["family"] == "pitch_layer"
            ]
            preview = None
            if candidates:
                preview = prepare_preview(root, candidates[0]["source"], wet=DEFAULT_WET)
            return {"report": report, "candidates": candidates, "preview": preview}

        def complete(result):
            self.scan_report = result["report"]
            self.preview_data = result["preview"]
            self.sample_combo.blockSignals(True)
            self.sample_combo.clear()
            for item in result["candidates"]:
                self.sample_combo.addItem(
                    f'{item["source"]}  ·  {item["status"].upper()}',
                    item["source"],
                )
            self.sample_combo.blockSignals(False)

            summary = self.scan_report["summary"]
            self.summary_label.setText(
                f'{self.scan_report["files_found"]} WAV files found. '
                f'{summary["ready"]} eligible for the current EQ test; '
                f'{summary["partial"]} partial; {summary["manual_review"]} need OTO review; '
                f'{summary["bypassed"]} intentionally left unchanged.'
            )
            if self.preview_data:
                self._show_preview_result(self.preview_data)
                self.play_original_button.setEnabled(True)
                self.play_processed_button.setEnabled(True)
            else:
                self.preview_note.setText(
                    "No pitch-layer sample was found for an automatic preview. The whole bank can still be exported with unchanged files and its processing report."
                )
            self.export_button.setEnabled(True)
            self._set_busy(False, "Scan complete. Listen to the preview, then create the output copy.")
            self.progress.setValue(100)

        self._start_worker(task, complete)

    def _show_preview_result(self, preview: dict):
        if preview["same_audio"]:
            self.preview_note.setText(
                f'{preview["relative_path"]}: no change is applied to this sample. {preview["reason"]}'
            )
            self.play_processed_button.setEnabled(False)
        else:
            bands = ", ".join(
                f'{item["frequency_hz"]:.0f} Hz {item["gain_db"]:+.1f} dB'
                for item in preview["profile"]["bands"]
            )
            self.preview_note.setText(
                f'{preview["relative_path"]}: {bands}. '
                f'{preview["mask"]["mask_coverage_percent"]:.1f}% of the WAV is inside eligible OTO body regions. '
                "Only that region is changed; consonant/fixed regions are preserved."
            )
            self.play_processed_button.setEnabled(True)

    def change_preview_sample(self, index: int):
        if index < 0 or self.worker and self.worker.isRunning():
            return
        relative = self.sample_combo.currentData()
        if not relative:
            return
        root = Path(self.root_edit.text()).expanduser().resolve()
        try:
            preview = prepare_preview(root, relative, wet=DEFAULT_WET)
        except Exception as exc:
            self.preview_note.setText(f"Preview failed: {exc}")
            self.play_processed_button.setEnabled(False)
            return
        self.preview_data = preview
        self._show_preview_result(preview)

    def play_audio(self, which: str):
        if not self.preview_data:
            return
        audio = self.preview_data["original"] if which == "original" else self.preview_data["processed"]
        if which == "processed" and self.preview_data["same_audio"]:
            return
        file_name = "original.wav" if which == "original" else "processed.wav"
        target = Path(self.preview_directory.name) / file_name
        sf.write(str(target), audio, self.preview_data["sample_rate"], subtype="PCM_16")
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(target.resolve())))
        self.player.play()

    def export_bank(self):
        if not self.scan_report:
            QMessageBox.information(self, "Scan first", "Scan the voicebank and prepare a preview first.")
            return
        root = Path(self.root_edit.text()).expanduser().resolve()
        output = Path(self.output_edit.text()).expanduser().resolve()
        if root == output or root in output.parents or output in root.parents:
            QMessageBox.warning(self, "Unsafe output", "Choose an output folder completely separate from the source voicebank.")
            return
        summary = self.scan_report["summary"]
        expected = summary["ready"]
        answer = QMessageBox.question(
            self, "Create processed voicebank copy",
            f"Create a replacement-ready copy of the complete bank?\n\n"
            f"WAVs found: {self.scan_report['files_found']}\n"
            f"Eligible for current processing: {expected}\n"
            f"Will remain unchanged / require review: "
            f"{summary['partial'] + summary['manual_review'] + summary['bypassed']}\n\n"
            f"Output: {output}\n\n"
            "The original voicebank will not be modified. Existing files in the output folder with matching names will be overwritten.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.progress.setValue(0)
        self._set_busy(True, "Creating the replacement-ready voicebank copy…")

        def task(progress):
            return process_auto(
                root, output, wet=DEFAULT_WET, dry_run=False, progress=progress,
            )

        def complete(report):
            summary = report["summary"]
            self.scan_report = report
            self._set_busy(False, "Export complete. The source bank remains untouched.")
            self.progress.setValue(100)
            QMessageBox.information(
                self, "Voicebank copy created",
                f'Output bank created successfully.\n\n'
                f'WAVs changed: {summary["processed"]}\n'
                f'Copied unchanged / skipped: {summary["partial"] + summary["manual_review"] + summary["bypassed"]}\n'
                f'Output: {output}\n\n'
                "You can now back up your original bank and replace it with this output copy. Check processing_report.json before adopting it.",
            )
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output)))

        self._start_worker(task, complete)

    def open_output(self):
        path = Path(self.output_edit.text()).expanduser()
        if path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        else:
            QMessageBox.information(self, "Output not created", "Run the export first.")

    def closeEvent(self, event):
        self.player.stop()
        self.preview_directory.cleanup()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Eugene:RE Quick Voicebank Restore")
    window = QuickRestoreWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
