"""UI builder for MainWindow.

Creates the three VistaWidget views, ControlsWidget, and lays them out.
Keeps MainWindow thin by moving verbose UI construction here.
"""
from __future__ import annotations
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSplitter
from PyQt6.QtCore import Qt

from ..vista_widget import VistaWidget
from ..controls_widget import ControlsWidget


def build_ui(mw: QWidget) -> None:
    """Constructs UI components and attaches them to the MainWindow instance.

    Expects mw to be a QMainWindow subclass with setCentralWidget available.
    """
    # Center widget and main layout
    central_widget = QWidget(mw)
    mw.setCentralWidget(central_widget)
    main_layout = QVBoxLayout(central_widget)

    # Three 3D views
    mw.view_left = VistaWidget(mw)
    mw.view_left.setObjectName("view_left")

    mw.view_center = VistaWidget(mw)
    mw.view_center.setObjectName("view_center")

    mw.view_right = VistaWidget(mw)
    mw.view_right.setObjectName("view_right")

    # Top splitter with three views
    top_splitter = QSplitter(Qt.Orientation.Horizontal)
    top_splitter.addWidget(mw.view_left)
    top_splitter.addWidget(mw.view_center)
    top_splitter.addWidget(mw.view_right)
    top_splitter.setSizes([300, 500, 300])

    # Bottom controls
    mw.controls_widget = ControlsWidget(mw)
    mw.controls_widget.setObjectName("controls_widget")

    # Main vertical splitter
    main_splitter = QSplitter(Qt.Orientation.Vertical)
    main_splitter.addWidget(top_splitter)
    main_splitter.addWidget(mw.controls_widget)
    main_splitter.setSizes([600, 400])

    main_layout.addWidget(main_splitter)
