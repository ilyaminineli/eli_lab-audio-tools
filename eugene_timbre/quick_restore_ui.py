"""Simple one-pass Eugene:RE whole-file restoration interface.

The interface exposes bank selection, waveform/audio preview, and full-bank export.
Every WAV receives the complete-duration signal chain, without OTO-mask gating.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal, Qt, QUrl, QTimer, QRectF
from PySide6.QtGui import QDesktopServices, QColor, QPainter, QPainterPath, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QComboBox, QPushButton,
    QProgressBar, QVBoxLayout, QWidget, QSizePolicy,
)

from .full_sample_processing import process_full_sample_bank
from .quick_restore import DEFAULT_WET, prepare_preview


DEFAULT_INPUT = "C:/Users/photo/OneDrive/Документы/UtauV/Data/Singers/EugeneRE"
DEFAULT_OUTPUT = "C:/EugeneRE_Restoration"


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


class WaveformComparisonWidget(QWidget):
    """Original and processed waveforms side by side, with a difference track."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original = np.zeros(0, dtype=np.float32)
        self.processed = np.zeros(0, dtype=np.float32)
        self.sample_rate = 0
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @staticmethod
    def _mono(audio: np.ndarray) -> np.ndarray:
        data = np.asarray(audio, dtype=np.float32)
        return np.mean(data, axis=1) if data.ndim == 2 else data

    def set_waveforms(self, original, processed, sample_rate: int):
        self.original = self._mono(original).copy()
        self.processed = self._mono(processed).copy()
        self.sample_rate = int(sample_rate)
        self.update()

    @staticmethod
    def _draw_waveform(painter, signal, rect, color, scale):
        if signal.size == 0 or rect.width() < 2 or rect.height() < 4:
            return
        width = max(2, int(rect.width()))
        block = max(1, int(np.ceil(signal.size / width)))
        count = int(np.ceil(signal.size / block))
        clipped = signal[:count * block]
        padded_size = count * block
        if clipped.size < padded_size:
            clipped = np.pad(clipped, (0, padded_size - clipped.size))
        chunks = clipped.reshape(count, block)
        low = np.min(chunks, axis=1)
        high = np.max(chunks, axis=1)
        mid = rect.center().y()
        half = rect.height() * 0.44
        painter.setPen(QPen(QColor("#e4e0d7"), 1))
        painter.drawLine(int(rect.left()), int(mid), int(rect.right()), int(mid))
        painter.setPen(QPen(QColor(color), 1))
        for index, (minimum, maximum) in enumerate(zip(low, high)):
            x = rect.left() + index
            y1 = mid - float(maximum) / max(scale, 1e-8) * half
            y2 = mid - float(minimum) / max(scale, 1e-8) * half
            painter.drawLine(int(x), int(y1), int(x), int(y2))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fbfaf7"))
        if self.original.size == 0 or self.processed.size == 0:
            painter.setPen(QColor("#777a74"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a WAV sample to display the waveform comparison.")
            return

        width, height = self.width(), self.height()
        margin = 16
        label_h = 20
        gap = 14
        panel_w = (width - margin * 2 - gap) / 2
        panel_h = max(56, int((height - 90) * 0.48))
        top = 27
        rect_a = QRectF(margin, top, panel_w, panel_h)
        rect_b = QRectF(margin + panel_w + gap, top, panel_w, panel_h)
        painter.setPen(QColor("#4b514d"))
        painter.drawText(int(rect_a.left()), 17, "ORIGINAL")
        painter.drawText(int(rect_b.left()), 17, "PROCESSED")
        painter.setPen(QPen(QColor("#e7e4dd"), 1))
        painter.drawRect(rect_a)
        painter.drawRect(rect_b)

        shared_peak = max(
            float(np.max(np.abs(self.original))),
            float(np.max(np.abs(self.processed))),
            1e-4,
        )
        self._draw_waveform(painter, self.original, rect_a, "#59645e", shared_peak)
        self._draw_waveform(painter, self.processed, rect_b, "#e7795b", shared_peak)

        difference = self.processed - self.original
        diff_top = top + panel_h + 34
        diff_h = max(48, height - diff_top - 24)
        diff_rect = QRectF(margin, diff_top, width - margin * 2, diff_h)
        diff_peak = max(float(np.max(np.abs(difference))), 1e-5)
        painter.setPen(QColor("#4b514d"))
        painter.drawText(int(diff_rect.left()), int(diff_top - 8), f"DIFFERENCE  ·  peak {diff_peak:.5f}")
        painter.setPen(QPen(QColor("#e7e4dd"), 1))
        painter.drawRect(diff_rect)
        self._draw_waveform(painter, difference, diff_rect, "#7b68a6", diff_peak)
        duration = self.original.size / max(1, self.sample_rate)
        painter.setPen(QColor("#777a74"))
        painter.drawText(
            margin, height - 4,
            f"0.00 s    —    {duration:.2f} s  ·  same time scale; waveform amplitudes normalised together",
        )


class QuickRestoreWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eugene:RE — Full-Sample Voicebank Restore")
        self.resize(1180, 900)
        self.setMinimumSize(940, 760)

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
            "One automatic full-sample pass → waveform and audio comparison → replacement-ready bank copy."
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
            "The bank has not been scanned yet. Every WAV will be analysed and processed across its complete duration; the source bank remains untouched."
        )
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("summary")
        summary_layout.addWidget(self.summary_label)
        self.sample_combo = QComboBox()
        self.sample_combo.setMinimumHeight(38)
        self.sample_combo.currentIndexChanged.connect(self.change_preview_sample)
        summary_layout.addWidget(QLabel("Preview sample"))
        summary_layout.addWidget(self.sample_combo)
        self.waveform_view = WaveformComparisonWidget()
        summary_layout.addWidget(self.waveform_view, 1)
        self.preview_note = QLabel(
            "Original and processed waveform views will appear here, with a difference track below."
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
        self._set_busy(True, "Analysing every WAV and preparing a full-duration preview…")

        def task(progress):
            report = process_full_sample_bank(
                root, root.parent / (root.name + "_preflight"),
                wet=DEFAULT_WET, dry_run=True, progress=progress,
            )
            candidates = list(report["files"])
            candidates.sort(key=lambda item: (
                0 if item["family"] == "pitch_layer" and item["changed"] else
                1 if item["changed"] else
                2 if item["family"] == "pitch_layer" else 3,
                item["source"].casefold(),
            ))
            preview = (
                prepare_preview(root, candidates[0]["source"], wet=DEFAULT_WET)
                if candidates else None
            )
            return {"report": report, "candidates": candidates, "preview": preview}

        def complete(result):
            self.scan_report = result["report"]
            self.preview_data = result["preview"]
            self.sample_combo.blockSignals(True)
            self.sample_combo.clear()
            for item in result["candidates"]:
                state = "CHANGED" if item["changed"] else "NO TONAL CHANGE"
                self.sample_combo.addItem(
                    f'{item["source"]}  ·  {state}  ·  {item["family"]}',
                    item["source"],
                )
            self.sample_combo.blockSignals(False)

            summary = self.scan_report["summary"]
            self.summary_label.setText(
                f'{summary["wav_files"]} WAV files found and analysed across their full duration. '
                f'{summary["changed"]} currently show measurable changes; '
                f'{summary["unchanged_after_analysis"]} have no detected correction beyond the full-sample pass. '
                "Every WAV is included in export; OTO timing does not gate processing."
            )
            if self.preview_data:
                self._show_preview_result(self.preview_data)
                self.play_original_button.setEnabled(True)
                self.play_processed_button.setEnabled(True)
            else:
                self.preview_note.setText(
                    "No WAV file was found for preview. The export is unavailable."
                )
            self.export_button.setEnabled(bool(self.scan_report["files_found"]))
            self._set_busy(False, "Scan complete. Every WAV has a full-file processing route. Review the preview, then export.")
            self.progress.setValue(100)

        self._start_worker(task, complete)

    def _show_preview_result(self, preview: dict):
        self.waveform_view.set_waveforms(
            preview["original"], preview["processed"], preview["sample_rate"]
        )
        stages = preview["processing"]["stages"]
        active = [
            name.replace("_", " ")
            for name, info in stages.items()
            if info.get("applied")
        ]
        details = ", ".join(active) if active else "analysis found no corrective action"
        delta = np.asarray(preview["processed"], dtype=np.float64) - np.asarray(preview["original"], dtype=np.float64)
        peak_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        rms_delta = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
        status = "AUDIBLE/NUMERICAL CHANGE" if not preview["same_audio"] else "NO MEASURABLE SAMPLE CHANGE"
        self.preview_note.setText(
            f'{preview["relative_path"]}  ·  {status}\n'
            f'Full duration processed. Stages: {details}.\n'
            f'Peak difference: {peak_delta:.6f}  ·  Difference RMS: {20 * np.log10(max(rms_delta, 1e-12)):.1f} dBFS.'
        )
        self.play_original_button.setEnabled(True)
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
        expected = summary["wav_files"]
        answer = QMessageBox.question(
            self, "Create processed voicebank copy",
            f"Run the full-sample processing chain on every WAV and mirror the complete bank?\n\n"
            f"WAVs to process: {expected}\n"
            f"Expected measurable changes: {summary['changed']}\n"
            f"Files with no detected correction: {summary['unchanged_after_analysis']}\n\n"
            f"Output: {output}\n\n"
            "No OTO mask restricts the sample duration. Cleanup stages are applied when detected; tonal EQ is family-aware. "
            "The original bank will not be modified.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.progress.setValue(0)
        self._set_busy(True, "Creating the replacement-ready voicebank copy…")

        def task(progress):
            return process_full_sample_bank(
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
                f'WAVs processed at full duration: {summary["processed_full_duration"]}\n'
                f'WAVs with measurable changes: {summary["changed"]}\n'
                f'WAVs with no detected correction: {summary["unchanged_after_analysis"]}\n'
                f'Engine analysis sidecars removed for changed WAVs: {summary["sidecars_invalidated"]}\n'
                f'Output: {output}\n\n'
                "The output mirrors the complete bank. The original voicebank remains untouched. Check processing_report.json before adopting it.",
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
