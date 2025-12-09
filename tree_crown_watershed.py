# -*- coding: utf-8 -*-
from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import Qt

from .tree_crown_watershed_dockwidget import TreeCrownWatershedDockWidget


class TreeCrownWatershed:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dock = None

    def initGui(self):
        icon = QIcon(":/plugins/tree_crown_watershed/icon.png")
        self.action = QAction(icon, "Tree Crown Watershed", self.iface.mainWindow())
        self.action.triggered.connect(self.show_dock)

        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("Tree Crown Watershed", self.action)

    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("Tree Crown Watershed", self.action)
        if self.dock:
            self.iface.removeDockWidget(self.dock)

    def show_dock(self):
        if not self.dock:
            self.dock = TreeCrownWatershedDockWidget(self.iface)
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        self.dock.show()
