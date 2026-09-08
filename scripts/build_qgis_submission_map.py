"""Create the submission map with QGIS/PyQGIS as the cartographic engine."""

from __future__ import annotations

from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFillSymbol,
    QgsGraduatedSymbolRenderer,
    QgsLayoutExporter,
    QgsLayoutItemLabel,
    QgsLayoutItemLegend,
    QgsLayoutItemMap,
    QgsLayoutItemScaleBar,
    QgsLayoutPoint,
    QgsLayoutSize,
    QgsLegendStyle,
    QgsMarkerSymbol,
    QgsPrintLayout,
    QgsProject,
    QgsRectangle,
    QgsRendererRange,
    QgsUnitTypes,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QRectF
from qgis.PyQt.QtGui import QColor, QFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission" / "assets" / "maps"
QGIS_PREFIX = "/Applications/QGIS-LTR.app/Contents/MacOS"
BOUNDARY_PATH = ROOT / "assets" / "boundaries" / "shenzhen_districts.gpkg"
PRS_PATH = ROOT / "outputs" / "submission" / "restorative_quality_prs11" / "spatial_screening.gpkg"
SIXDIM_PATH = ROOT / "outputs" / "submission" / "urban_perception_sixdim" / "spatial_screening.gpkg"
PROJECT_CRS_AUTHID = "EPSG:32649"

SCORE_COLORS = ["#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"]


def _score_renderer(field: str) -> QgsGraduatedSymbolRenderer:
    ranges = []
    for index, (lower, upper) in enumerate([(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]):
        symbol = QgsMarkerSymbol.createSimple(
            {
                "name": "circle",
                "color": SCORE_COLORS[index],
                "outline_color": "#FFFFFF",
                "outline_width": "0.15",
                "size": "2.2",
            }
        )
        ranges.append(QgsRendererRange(lower, upper, symbol, f"{lower:.1f}-{upper:.1f}"))
    renderer = QgsGraduatedSymbolRenderer(field, ranges)
    renderer.setMode(QgsGraduatedSymbolRenderer.Custom)
    return renderer


def _load_layer(path: Path, name: str, field: str | None = None) -> QgsVectorLayer:
    uri = f"{path}|layername=screening" if path.name == "spatial_screening.gpkg" else str(path)
    layer = QgsVectorLayer(uri, name, "ogr")
    if not layer.isValid():
        raise RuntimeError(f"QGIS could not load {path}")
    if field:
        layer.setRenderer(_score_renderer(field))
    return layer


def _add_label(
    layout: QgsPrintLayout,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    size: int,
    bold: bool = False,
    color: str = "#1F2937",
) -> QgsLayoutItemLabel:
    item = QgsLayoutItemLabel(layout)
    item.setText(text)
    item.setFont(QFont("Arial", size, QFont.Bold if bold else QFont.Normal))
    item.setFontColor(QColor(color))
    item.adjustSizeToText()
    item.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
    item.attemptResize(QgsLayoutSize(width, height, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(item)
    return item


def _add_map(
    layout: QgsPrintLayout,
    layers: list[QgsVectorLayer],
    extent: QgsRectangle,
    x: float,
    crs: QgsCoordinateReferenceSystem,
) -> QgsLayoutItemMap:
    item = QgsLayoutItemMap(layout)
    item.setRect(QRectF(0, 0, 130, 122))
    item.setCrs(crs)
    item.setLayers(layers)
    item.setExtent(extent)
    item.setFrameEnabled(True)
    item.setFrameStrokeColor(QColor("#4B5563"))
    item.attemptMove(QgsLayoutPoint(x, 31, QgsUnitTypes.LayoutMillimeters))
    item.attemptResize(QgsLayoutSize(130, 122, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(item)
    return item


def main() -> int:
    QgsApplication.setPrefixPath(QGIS_PREFIX, True)
    app = QgsApplication([], False)
    app.initQgis()
    try:
        project = QgsProject.instance()
        project.clear()
        project_crs = QgsCoordinateReferenceSystem(PROJECT_CRS_AUTHID)
        if not project_crs.isValid():
            raise RuntimeError(f"QGIS could not initialize {PROJECT_CRS_AUTHID}")
        project.setCrs(project_crs)

        boundary = _load_layer(BOUNDARY_PATH, "Shenzhen boundary")
        boundary.renderer().setSymbol(
            QgsFillSymbol.createSimple({"color": "#F7F7F7", "outline_color": "#59636E", "outline_width": "0.45"})
        )
        prs = _load_layer(PRS_PATH, "Score classes", "vlm_restorative_average")
        sixdim = _load_layer(SIXDIM_PATH, "Beautiful", "vlm_beautiful")
        for layer in (boundary, prs, sixdim):
            project.addMapLayer(layer)

        transform = QgsCoordinateTransform(prs.crs(), project_crs, project)
        bounds = transform.transformBoundingBox(prs.extent())
        bounds.combineExtentWith(transform.transformBoundingBox(sixdim.extent()))
        margin_x = bounds.width() * 0.08
        margin_y = bounds.height() * 0.08
        extent = QgsRectangle(
            bounds.xMinimum() - margin_x,
            bounds.yMinimum() - margin_y,
            bounds.xMaximum() + margin_x,
            bounds.yMaximum() + margin_y,
        )

        layout = QgsPrintLayout(project)
        layout.initializeDefaults()
        layout.setName("PyRestore Shenzhen task comparison")
        page = layout.pageCollection().page(0)
        page.setPageSize(QgsLayoutSize(297, 210, QgsUnitTypes.LayoutMillimeters))

        _add_label(
            layout,
            "Shenzhen street-view interpretation scores",
            10,
            4,
            277,
            10,
            size=20,
            bold=True,
        )
        _add_label(
            layout,
            "Two task definitions, one matched 465-location geospatial contract",
            10,
            14,
            277,
            7,
            size=10,
            color="#4B5563",
        )
        _add_label(layout, "(a) PRS four-dimension mean", 10, 23, 130, 7, size=12, bold=True)
        _add_label(layout, "(b) Beautiful", 157, 23, 130, 7, size=12, bold=True)

        left_map = _add_map(layout, [prs, boundary], extent, 10, project_crs)
        right_map = _add_map(layout, [sixdim, boundary], extent, 157, project_crs)

        legend = QgsLayoutItemLegend(layout)
        legend.setTitle("Score (fixed 0-1 scale)")
        legend.setLinkedMap(left_map)
        legend.setAutoUpdateModel(False)
        legend_root = legend.model().rootGroup()
        for child in list(legend_root.children()):
            if child.name() != prs.name():
                legend_root.removeChildNode(child)
        legend.setStyleFont(QgsLegendStyle.Title, QFont("Arial", 9, QFont.Bold))
        legend.setStyleFont(QgsLegendStyle.SymbolLabel, QFont("Arial", 8))
        legend.setFrameEnabled(False)
        legend.attemptMove(QgsLayoutPoint(10, 157, QgsUnitTypes.LayoutMillimeters))
        legend.attemptResize(QgsLayoutSize(82, 46, QgsUnitTypes.LayoutMillimeters))
        layout.addLayoutItem(legend)

        scale = QgsLayoutItemScaleBar(layout)
        scale.setStyle("Single Box")
        scale.setLinkedMap(right_map)
        scale.setUnits(QgsUnitTypes.DistanceKilometers)
        scale.setNumberOfSegments(2)
        scale.setNumberOfSegmentsLeft(0)
        scale.setUnitsPerSegment(5)
        scale.setUnitLabel("km")
        scale.setFont(QFont("Arial", 8))
        scale.applyDefaultSize()
        scale.attemptMove(QgsLayoutPoint(160, 160, QgsUnitTypes.LayoutMillimeters))
        layout.addLayoutItem(scale)

        project.layoutManager().addLayout(layout)
        OUT.mkdir(parents=True, exist_ok=True)
        project.write(str(OUT / "shenzhen_task_comparison.qgz"))

        exporter = QgsLayoutExporter(layout)
        for output_name in (
            "shenzhen_task_comparison_qgis.pdf",
            "shenzhen_task_comparison_qgis.png",
            "shenzhen_task_comparison_qgis.svg",
        ):
            (OUT / output_name).unlink(missing_ok=True)
        pdf_settings = QgsLayoutExporter.PdfExportSettings()
        pdf_settings.dpi = 300
        if (
            exporter.exportToPdf(str(OUT / "shenzhen_task_comparison_qgis.pdf"), pdf_settings)
            != QgsLayoutExporter.Success
        ):
            raise RuntimeError("QGIS PDF export failed")
        image_settings = QgsLayoutExporter.ImageExportSettings()
        image_settings.dpi = 300
        if (
            exporter.exportToImage(str(OUT / "shenzhen_task_comparison_qgis.png"), image_settings)
            != QgsLayoutExporter.Success
        ):
            raise RuntimeError("QGIS PNG export failed")
        svg_settings = QgsLayoutExporter.SvgExportSettings()
        svg_settings.dpi = 300
        if (
            exporter.exportToSvg(str(OUT / "shenzhen_task_comparison_qgis.svg"), svg_settings)
            != QgsLayoutExporter.Success
        ):
            raise RuntimeError("QGIS SVG export failed")
        print(f"QGIS submission map written to {OUT}")
        return 0
    finally:
        app.exitQgis()


if __name__ == "__main__":
    raise SystemExit(main())
