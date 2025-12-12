# -*- coding: utf-8 -*-
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QPushButton,
    QMessageBox,
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor, QFont


from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsField,
    QgsSymbol,
    QgsCategorizedSymbolRenderer,
    QgsRendererCategory,
    QgsTextFormat,
    QgsPalLayerSettings,
    QgsVectorLayerSimpleLabeling
)

from pathlib import Path
import traceback
import numpy as np
import random

from . import watershed_backend


class TreeCrownWatershedDockWidget(QDockWidget):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("Tree Crown Watershed")

        self.crowns_layer = None  # warstwa z koronami (punkty z tree_id)

        root = QWidget(self)
        self.setWidget(root)

        main_layout = QVBoxLayout(root)

        # --- wiersz z nazwą warstwy ---
        row_layer = QHBoxLayout()
        row_layer.addWidget(QLabel("Aktywna warstwa:"))
        self.lblLayer = QLabel("(brak)")
        self.lblLayer.setStyleSheet("font-weight: bold;")
        row_layer.addWidget(self.lblLayer)
        row_layer.addStretch()
        main_layout.addLayout(row_layer)

        # --- parametry ---
        main_layout.addWidget(QLabel("Parametry segmentacji:"))

        row_h = QHBoxLayout()
        row_h.addWidget(QLabel("Min. wysokość czubka [m]:"))
        self.spinMinHeight = QDoubleSpinBox()
        self.spinMinHeight.setRange(0.0, 100.0)
        self.spinMinHeight.setSingleStep(0.1)
        self.spinMinHeight.setValue(watershed_backend.MIN_HEIGHT)
        row_h.addWidget(self.spinMinHeight)
        main_layout.addLayout(row_h)

        row_d = QHBoxLayout()
        row_d.addWidget(QLabel("Min. odległość LM [px]:"))
        self.spinMinDist = QSpinBox()
        self.spinMinDist.setRange(1, 500)
        self.spinMinDist.setValue(watershed_backend.MIN_DIST)
        row_d.addWidget(self.spinMinDist)
        main_layout.addLayout(row_d)

        row_s = QHBoxLayout()
        row_s.addWidget(QLabel("Sigma Gaussa:"))
        self.spinSigma = QDoubleSpinBox()
        self.spinSigma.setRange(0.0, 50.0)
        self.spinSigma.setSingleStep(0.1)
        self.spinSigma.setValue(watershed_backend.GAUSS_SIGMA)
        row_s.addWidget(self.spinSigma)
        main_layout.addLayout(row_s)

        # --- opcje ---
        self.chkShowLM = QCheckBox("Dodaj warstwę z czubkami drzew (LM)")
        self.chkShowLM.setChecked(True)
        main_layout.addWidget(self.chkShowLM)

        self.chkCreateCrowns = QCheckBox("Dodaj warstwę z koronami (punkty z tree_id)")
        self.chkCreateCrowns.setChecked(True)
        main_layout.addWidget(self.chkCreateCrowns)

        main_layout.addStretch()

        # --- przycisk ---
        self.btnSegment = QPushButton("Segmentuj drzewa (watershed)")
        self.btnSegment.setStyleSheet("font-weight: bold; padding: 6px;")
        main_layout.addWidget(self.btnSegment)

        # sygnały
        self.btnSegment.clicked.connect(self.run_segmentation)
        self.iface.layerTreeView().currentLayerChanged.connect(self.update_active_layer)

        self.update_active_layer()

    # ------------------ logika ------------------

    def update_active_layer(self, *args):
        layer = self.iface.activeLayer()
        if layer:
            try:
                src = layer.dataProvider().dataSourceUri().split("|")[0]
            except Exception:
                src = ""
            self.lblLayer.setText(src if src else "(brak ścieżki pliku)")
        else:
            self.lblLayer.setText("(brak)")

    def run_segmentation(self):
        try:
            layer = self.iface.activeLayer()
            if layer is None:
                QMessageBox.warning(self, "Błąd", "Zaznacz warstwę LAS/LAZ w QGIS.")
                return

            src_path = layer.dataProvider().dataSourceUri().split("|")[0]
            src = Path(src_path)
            if not src.exists():
                QMessageBox.warning(self, "Błąd", f"Plik nie istnieje:\n{src}")
                return

            watershed_backend.MIN_HEIGHT = self.spinMinHeight.value()
            watershed_backend.MIN_DIST = self.spinMinDist.value()
            watershed_backend.GAUSS_SIGMA = self.spinSigma.value()

            result = watershed_backend.process_wycinek(src)

            treetops = result.get("treetops_xy", np.empty((0, 2)))
            crown_xy = result.get("crown_points_xy", np.empty((0, 2)))
            crown_ids = result.get("crown_tree_ids", np.array([], dtype=np.uint32))

            if self.chkShowLM.isChecked() and len(treetops) > 0:
                self.add_lm_layer(treetops, layer.crs().authid())

            if (
                self.chkCreateCrowns.isChecked()
                and len(crown_xy) > 0
                and len(crown_ids) == len(crown_xy)
            ):
                self.crowns_layer = self.add_crowns_layer(
                    crown_xy, crown_ids, layer.crs().authid()
                )

            QMessageBox.information(
                self,
                "Zakończono",
                f"Segmentacja zakończona.\n"
                f"Punktów wejściowych: {result['n_points']}\n"
                f"Liczba drzew (segmentów): {result['n_trees']}\n",
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Błąd krytyczny",
                f"{e}\n\n{traceback.format_exc()}",
            )

    def add_lm_layer(self, coords, crs_authid: str):
        vlayer = QgsVectorLayer(f"Point?crs={crs_authid}", "Treetops", "memory")
        pr = vlayer.dataProvider()

        pr.addAttributes([QgsField("id", QVariant.Int)])
        vlayer.updateFields()

        feats = []
        for i, (x, y) in enumerate(coords, start=1):
            f = QgsFeature()
            f.setFields(vlayer.fields())
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(x), float(y))))
            f.setAttribute("id", int(i))
            feats.append(f)

        pr.addFeatures(feats)
        vlayer.updateExtents()
        QgsProject.instance().addMapLayer(vlayer)

        # czerwone punkty
        sym = QgsSymbol.defaultSymbol(vlayer.geometryType())
        sym.setColor(QColor(255, 0, 0))
        vlayer.renderer().setSymbol(sym)

        text_format = QgsTextFormat()
        text_format.setFont(QFont("Arial", 11))
        text_format.setSize(11)
        text_format.setColor(QColor(255, 255, 255))

        # Czarna obwódka wokół liter
        buffer = text_format.buffer()
        buffer.setEnabled(True)
        buffer.setSize(1.2)  # grubość obwódki
        buffer.setColor(QColor(0, 0, 0))
        text_format.setBuffer(buffer)

        pal = QgsPalLayerSettings()
        pal.fieldName = "id"
        pal.enabled = True
        pal.setFormat(text_format)

        labeling = QgsVectorLayerSimpleLabeling(pal)
        vlayer.setLabeling(labeling)
        vlayer.setLabelsEnabled(True)
        vlayer.triggerRepaint()

        return vlayer

    def add_crowns_layer(self, coords, tree_ids, crs_authid: str):
        if coords is None or len(coords) == 0:
            return None
        if tree_ids is None or len(tree_ids) == 0:
            return None

        vlayer = QgsVectorLayer(f"Point?crs={crs_authid}", "CrownsPoints", "memory")
        pr = vlayer.dataProvider()

        pr.addAttributes([QgsField("tree_id", QVariant.Int)])
        vlayer.updateFields()

        feats = []
        for (x, y), tid in zip(coords, tree_ids):
            f = QgsFeature()
            f.setFields(vlayer.fields())
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(x), float(y))))
            f.setAttribute("tree_id", int(tid))
            feats.append(f)

        pr.addFeatures(feats)
        vlayer.updateExtents()
        QgsProject.instance().addMapLayer(vlayer)

        unique_ids = sorted(set(int(t) for t in tree_ids if t > 0))
        categories = []

        for tid in unique_ids:
            sym = QgsSymbol.defaultSymbol(vlayer.geometryType())
            sym.setSize(1.5)
            sym.setColor(
                QColor(
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255),
                )
            )
            cat = QgsRendererCategory(tid, sym, f"Drzewo {tid}")
            categories.append(cat)

        renderer = QgsCategorizedSymbolRenderer("tree_id", categories)
        vlayer.setRenderer(renderer)
        vlayer.triggerRepaint()
        return vlayer
