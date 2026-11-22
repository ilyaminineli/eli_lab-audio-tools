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
from PySide6.QtGui import QDesktopServices, QColor, QPainter, QPen, QImage
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QComboBox, QPushButton,
    QProgressBar, QVBoxLayout, QWidget, QSizePolicy, QScrollArea, QFrame,
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


class SpectrogramComparisonWidget(QWidget):
    """Aligned log-frequency spectrograms with a signed spectral-difference view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spectrogram = None
        self._image_cache = {}
        self.setMinimumHeight(440)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

    def set_spectrogram(self, data: dict):
        self.spectrogram = data
        self._image_cache.clear()
        self.update()

    @staticmethod
    def _resample_matrix(matrix, frequencies, width: int, height: int):
        source = np.asarray(matrix, dtype=np.float32)
        freq = np.asarray(frequencies, dtype=np.float64)
        width = max(2, int(width))
        height = max(2, int(height))
        if source.ndim != 2 or source.shape[0] != freq.size:
            raise ValueError("Spectrogram matrix and frequency axis do not match.")

        fmin = max(float(freq[0]), 1.0)
        fmax = max(float(freq[-1]), fmin * 1.01)
        target_frequency = np.geomspace(fmin, fmax, height)
        log_frequency = np.log(np.maximum(freq, 1.0))
        target_log_frequency = np.log(target_frequency)
        frequency_resampled = np.empty((height, source.shape[1]), dtype=np.float32)
        for column in range(source.shape[1]):
            frequency_resampled[:, column] = np.interp(
                target_log_frequency, log_frequency, source[:, column]
            )
        frequency_resampled = frequency_resampled[::-1, :]

        target_time = np.linspace(0, source.shape[1] - 1, width)
        output = np.empty((height, width), dtype=np.float32)
        source_time = np.arange(source.shape[1], dtype=np.float64)
        for row in range(height):
            output[row, :] = np.interp(
                target_time, source_time, frequency_resampled[row, :]
            )
        return output

    @staticmethod
    def _spectrogram_rgb(values):
        level = np.clip((values + 75.0) / 75.0, 0.0, 1.0)
        # Dark navy -> muted teal -> warm ivory, fixed scale shared by A and B.
        stops = np.array([0.0, 0.42, 0.76, 1.0], dtype=np.float32)
        palette = np.array([
            [14, 20, 29],
            [43, 78, 91],
            [129, 126, 105],
            [247, 226, 184],
        ], dtype=np.float32)
        channels = [
            np.interp(level, stops, palette[:, channel])
            for channel in range(3)
        ]
        return np.stack(channels, axis=-1).astype(np.uint8)

    @staticmethod
    def _difference_rgb(values):
        delta = np.clip(values / 6.0, -1.0, 1.0)
        neutral = np.array([39, 43, 45], dtype=np.float32)
        positive = np.array([239, 135, 78], dtype=np.float32)
        negative = np.array([83, 153, 211], dtype=np.float32)
        rgb = np.empty((*delta.shape, 3), dtype=np.float32)
        pos = np.maximum(delta, 0.0)[..., None]
        neg = np.maximum(-delta, 0.0)[..., None]
        rgb[:] = neutral
        rgb = rgb * (1.0 - pos) + positive * pos
        rgb = rgb * (1.0 - neg) + negative * neg
        return np.clip(rgb, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_image(rgb: np.ndarray) -> QImage:
        pixels = np.ascontiguousarray(rgb, dtype=np.uint8)
        height, width, _ = pixels.shape
        return QImage(
            pixels.data, width, height, width * 3,
            QImage.Format.Format_RGB888,
        ).copy()

    def _make_image(self, key: str, matrix, frequencies, width: int, height: int):
        cache_key = (key, int(width), int(height))
        cached = self._image_cache.get(cache_key)
        if cached is not None:
            return cached
        values = self._resample_matrix(matrix, frequencies, width, height)
        rgb = self._difference_rgb(values) if key == "difference" else self._spectrogram_rgb(values)
        image = self._to_image(rgb)
        self._image_cache[cache_key] = image
        return image

    @staticmethod
    def _draw_frequency_axis(painter, rect):
        painter.setPen(QPen(QColor("#d5d2c9"), 1))
        for frequency, label in ((100.0, "100 Hz"), (1000.0, "1 kHz"), (10000.0, "10 kHz")):
            fraction = np.log10(frequency / 70.0) / np.log10(12000.0 / 70.0)
            y = rect.bottom() - fraction * rect.height()
            painter.drawLine(int(rect.left()), int(y), int(rect.left() + 5), int(y))
            painter.setPen(QColor("#73766f"))
            painter.drawText(int(rect.left() - 48), int(y + 4), label)
            painter.setPen(QPen(QColor("#d5d2c9"), 1))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#fbfaf7"))
        if not self.spectrogram:
            painter.setPen(QColor("#777a74"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "Select a WAV sample to display its spectrogram comparison.",
            )
            return

        data = self.spectrogram
        margin_left, margin_right, gap = 58, 16, 18
        content_width = max(120, self.width() - margin_left - margin_right)
        panel_width = max(50, int((content_width - gap) / 2))
        top = 30
        panel_height = max(100, int((self.height() - 150) * 0.43))
        rect_a = QRectF(margin_left, top, panel_width, panel_height)
        rect_b = QRectF(margin_left + panel_width + gap, top, panel_width, panel_height)

        painter.setPen(QColor("#343a37"))
        painter.drawText(int(rect_a.left()), 19, "ORIGINAL  ·  RELATIVE LEVEL")
        painter.drawText(int(rect_b.left()), 19, "PROCESSED  ·  SAME dB SCALE")
        painter.drawImage(
            rect_a,
            self._make_image(
                "original", data["original_db"], data["frequencies_hz"],
                int(rect_a.width()), int(rect_a.height()),
            ),
        )
        painter.drawImage(
            rect_b,
            self._make_image(
                "processed", data["processed_db"], data["frequencies_hz"],
                int(rect_b.width()), int(rect_b.height()),
            ),
        )
        painter.setPen(QPen(QColor("#d5d2c9"), 1))
        painter.drawRect(rect_a)
        painter.drawRect(rect_b)
        self._draw_frequency_axis(painter, rect_a)

        diff_top = top + panel_height + 54
        diff_height = max(72, self.height() - diff_top - 34)
        diff_rect = QRectF(margin_left, diff_top, content_width, diff_height)
        painter.setPen(QColor("#343a37"))
        painter.drawText(
            int(diff_rect.left()), int(diff_top - 10),
            "SPECTRAL DIFFERENCE  ·  PROCESSED − ORIGINAL  ·  BLUE = LESS / ORANGE = MORE",
        )
        painter.drawImage(
            diff_rect,
            self._make_image(
                "difference", data["difference_db"], data["frequencies_hz"],
                int(diff_rect.width()), int(diff_rect.height()),
            ),
        )
        painter.setPen(QPen(QColor("#d5d2c9"), 1))
        painter.drawRect(diff_rect)
        self._draw_frequency_axis(painter, diff_rect)
        painter.setPen(QColor("#73766f"))
        painter.drawText(
            margin_left, self.height() - 7,
            f'0.00 s  —  {data["duration_seconds"]:.2f} s     '
            f'Frequency axis is logarithmic · A/B use a shared level reference · Difference range ±6 dB',
        )


class QuickRestoreWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eugene:RE — Full-Sample Voicebank Restore")
        self.resize(1180, 860)
        self.setMinimumSize(820, 620)

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
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
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
        self.sample_combo.setMinimumContentsLength(30)
        self.sample_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.sample_combo.currentIndexChanged.connect(self.change_preview_sample)
        summary_layout.addWidget(QLabel("Preview sample"))
        summary_layout.addWidget(self.sample_combo)
        self.spectrogram_view = SpectrogramComparisonWidget()
        summary_layout.addWidget(self.spectrogram_view, 1)
        self.preview_note = QLabel(
            "Original and processed log-frequency sonograms will appear here, with a spectral-difference view below."
        )
        self.preview_note.setWordWrap(True)
        self.preview_note.setObjectName("muted")
        self.preview_note.setMinimumHeight(54)
        summary_layout.addWidget(self.preview_note)
        self.summary_group.setMinimumHeight(590)
        layout.addWidget(self.summary_group)

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

        outer_layout.addWidget(self.scroll_area)
        self.scroll_area.setWidget(scroll_content)

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
                f'{summary["unchanged_after_analysis"]} have no detected correction; '
                f'{summary["fallback_copied"]} could not be decoded and will be copied unchanged. '
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
        self.spectrogram_view.set_spectrogram(preview["spectrogram"])
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
                f'WAVs copied as fallback after read/processing error: {summary["fallback_copied"]}\n'
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
