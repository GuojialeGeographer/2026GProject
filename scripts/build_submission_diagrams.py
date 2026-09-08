"""Generate editable native-element Draw.io specification diagrams for PyRestore."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission" / "assets" / "diagrams"

COLORS = {
    "contract": ("#DCEBFA", "#2F6B9A"),
    "compute": ("#FCE8D5", "#B85C20"),
    "evidence": ("#E4F2E8", "#397A4A"),
    "control": ("#EEE9F7", "#6B52A3"),
    "neutral": ("#F3F4F6", "#666666"),
    "danger": ("#FBE3E3", "#B04444"),
}


class Diagram:
    def __init__(self, name: str, width: int = 1600, height: int = 900) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.mxfile = ET.Element("mxfile", host="app.diagrams.net", agent="PyRestore")
        diagram = ET.SubElement(self.mxfile, "diagram", id=name, name="Page-1")
        self.model = ET.SubElement(
            diagram,
            "mxGraphModel",
            dx="1600",
            dy="900",
            grid="1",
            gridSize="10",
            guides="1",
            tooltips="1",
            connect="1",
            arrows="1",
            fold="1",
            page="1",
            pageScale="1",
            pageWidth=str(width),
            pageHeight=str(height),
            math="0",
            shadow="0",
        )
        root = ET.SubElement(self.model, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")
        self.root = root
        self.counter = 2

    def vertex(
        self,
        text: str,
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        role: str = "neutral",
        font_size: int = 20,
        bold: bool = False,
        rounded: bool = True,
        dashed: bool = False,
    ) -> str:
        cell_id = str(self.counter)
        self.counter += 1
        fill, stroke = COLORS[role]
        style = (
            f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;fillColor={fill};"
            f"strokeColor={stroke};strokeWidth=2;fontColor=#1F2937;fontSize={font_size};"
            f"fontStyle={1 if bold else 0};align=center;verticalAlign=middle;spacing=8;"
            f"dashed={1 if dashed else 0};"
        )
        cell = ET.SubElement(
            self.root,
            "mxCell",
            id=cell_id,
            value=text,
            style=style,
            vertex="1",
            parent="1",
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
        )
        return cell_id

    def label(self, text: str, x: int, y: int, w: int, h: int, font_size: int = 24) -> str:
        cell_id = str(self.counter)
        self.counter += 1
        cell = ET.SubElement(
            self.root,
            "mxCell",
            id=cell_id,
            value=text,
            style=(
                f"text;html=1;strokeColor=none;fillColor=none;align=center;"
                f"verticalAlign=middle;whiteSpace=wrap;fontSize={font_size};fontStyle=1;"
            ),
            vertex="1",
            parent="1",
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
        )
        return cell_id

    def shaped_vertex(
        self,
        text: str,
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        role: str = "neutral",
        shape: str = "ellipse",
        font_size: int = 18,
        bold: bool = False,
    ) -> str:
        cell_id = str(self.counter)
        self.counter += 1
        fill, stroke = COLORS[role]
        style = (
            f"shape={shape};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"strokeWidth=2;fontColor=#1F2937;fontSize={font_size};"
            f"fontStyle={1 if bold else 0};align=center;verticalAlign=middle;spacing=8;"
        )
        cell = ET.SubElement(
            self.root,
            "mxCell",
            id=cell_id,
            value=text,
            style=style,
            vertex="1",
            parent="1",
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
        )
        return cell_id

    def edge(
        self,
        source: str,
        target: str,
        text: str = "",
        *,
        dashed: bool = False,
        color: str = "#4B5563",
        exit: tuple[float, float] | None = None,
        entry: tuple[float, float] | None = None,
    ) -> str:
        cell_id = str(self.counter)
        self.counter += 1
        style = (
            f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
            f"html=1;strokeWidth=2;strokeColor={color};endArrow=block;endFill=1;"
            f"dashed={1 if dashed else 0};fontSize=16;labelBackgroundColor=#FFFFFF;"
        )
        if exit is not None:
            style += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
        if entry is not None:
            style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        cell = ET.SubElement(
            self.root,
            "mxCell",
            id=cell_id,
            value=text,
            style=style,
            edge="1",
            parent="1",
            source=source,
            target=target,
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        return cell_id

    def write(self) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        ET.indent(self.mxfile, space="  ")
        path = OUT / f"{self.name}.drawio"
        ET.ElementTree(self.mxfile).write(path, encoding="utf-8", xml_declaration=True)
        return path


def framework_overview() -> Path:
    d = Diagram("framework_overview")
    d.label("PyRestore: task-configurable street-view indicator pipeline", 260, 25, 1080, 55, 30)
    task = d.vertex("TaskDefinition\nYAML + prompt + output schema", 430, 95, 420, 90, role="control", bold=True)
    manifest = d.vertex(
        "Single-scene manifest\nSite + Capture + coordinates", 40, 260, 310, 100, role="contract", bold=True
    )
    pair = d.vertex(
        "Paired SceneSet\nbefore + after + comparability audit", 40, 480, 310, 110, role="contract", bold=True
    )
    contract = d.vertex(
        "Contract layer\nidentity + signature + field validation", 430, 350, 420, 110, role="contract", bold=True
    )
    cv = d.vertex("CV analyzer\nvisible scene composition", 930, 230, 300, 100, role="compute", bold=True)
    vlm = d.vertex("VLM analyzer\nsemantic interpretation", 930, 430, 300, 100, role="compute", bold=True)
    cache = d.vertex("SQLite cache\nSceneSet + request fingerprint", 930, 650, 300, 90, role="control")
    harmonise = d.vertex(
        "Failure-aware harmonisation\nleft joins + paired deltas", 1280, 330, 290, 110, role="evidence", bold=True
    )
    validate = d.vertex(
        "Claim-specific validation\nstatus + CI + calibration + FDR", 1280, 120, 290, 115, role="evidence", bold=True
    )
    products = d.vertex(
        "Managed product family\nCSV + GeoPackage + quality + provenance",
        1280,
        540,
        290,
        115,
        role="evidence",
        bold=True,
    )
    notebooks = d.vertex("Jupyter notebooks + Python API", 410, 690, 460, 75, role="neutral", bold=True)
    d.edge(task, contract)
    d.edge(manifest, contract)
    d.edge(pair, contract)
    d.edge(contract, cv)
    d.edge(contract, vlm)
    d.edge(vlm, cache)
    d.edge(cv, harmonise)
    d.edge(vlm, harmonise)
    d.edge(harmonise, validate)
    d.edge(harmonise, products)
    d.edge(validate, products)
    d.edge(notebooks, contract, dashed=True)
    return d.write()


def data_model() -> Path:
    d = Diagram("data_model")
    d.label("PyRestore domain and product data model", 360, 25, 880, 55, 30)
    site = d.vertex(
        "SITE\nsite_id (PK)\nspatial identity", 80, 140, 260, 120, role="contract", bold=True, rounded=False
    )
    capture = d.vertex(
        "CAPTURE\ncapture_id (PK)\nimage_path + SHA-256\nlon + lat + time + heading",
        420,
        120,
        330,
        160,
        role="contract",
        bold=True,
        rounded=False,
    )
    scene = d.vertex(
        "SCENE_SET\nsingle capture or audited pair\npair_id + comparability state",
        830,
        120,
        350,
        160,
        role="contract",
        bold=True,
        rounded=False,
    )
    d.vertex(
        "TASK_DEFINITION\ntask_id + version\ninput signature\noutput + validation schema",
        1260,
        120,
        300,
        160,
        role="control",
        bold=True,
        rounded=False,
    )
    run = d.vertex(
        "ANALYZER_RUN\nmodel + request hash\nexecution mode + status + error",
        180,
        430,
        330,
        150,
        role="compute",
        bold=True,
        rounded=False,
    )
    result = d.vertex(
        "ANALYZER_RESULT\nCV physical fields\nVLM semantic fields\ncache_hit",
        630,
        430,
        330,
        150,
        role="compute",
        bold=True,
        rounded=False,
    )
    validation = d.vertex(
        "VALIDATION_RESULT\nreference type + overlap\nr / rho / CI or P/R/F1\nstatus",
        1080,
        430,
        330,
        150,
        role="evidence",
        bold=True,
        rounded=False,
    )
    product = d.vertex(
        "GEOSPATIAL_PRODUCT\nindicators.gpkg + CSV\nquality + provenance\nrun manifest",
        630,
        700,
        340,
        145,
        role="evidence",
        bold=True,
        rounded=False,
    )
    d.edge(site, capture)
    d.edge(capture, scene)
    d.edge(scene, run)
    d.edge(run, result)
    d.edge(result, validation)
    d.edge(result, product, exit=(0.5, 1), entry=(0.5, 0))
    d.edge(validation, product, exit=(0, 1), entry=(1, 0.3))
    return d.write()


def request_state() -> Path:
    d = Diagram("request_state")
    d.label("VLM request, cache and failure-preservation states", 320, 25, 960, 55, 30)
    pending = d.vertex("Pending SceneSet", 30, 135, 220, 80, role="neutral", bold=True)
    fingerprint = d.vertex("Hash images + task/request settings", 310, 120, 330, 110, role="control", bold=True)
    cache = d.vertex("Cache lookup", 710, 135, 220, 80, role="control", bold=True)
    validate_cached = d.vertex("Cached object\nschema validation", 1030, 120, 260, 105, role="contract")
    complete_cached = d.vertex("Complete cached record", 1350, 135, 220, 80, role="evidence", bold=True)
    authorized = d.vertex("Remote transmission\nauthorization check", 690, 350, 260, 100, role="control", bold=True)
    request = d.vertex("Remote image request", 1030, 355, 250, 85, role="compute")
    parse = d.vertex("Parse + validate JSON", 1330, 355, 240, 85, role="contract")
    retry = d.vertex("Transient failure\nretry if attempts remain", 1010, 555, 270, 95, role="danger")
    store = d.vertex("Store validated response", 1320, 555, 260, 95, role="control")
    error = d.vertex(
        "Record vlm_status=error\nauthorization, schema or retries exhausted",
        650,
        720,
        350,
        100,
        role="danger",
        bold=True,
    )
    complete = d.vertex(
        "Complete requested record\nok or explicit error", 1190, 720, 300, 100, role="evidence", bold=True
    )
    d.edge(pending, fingerprint)
    d.edge(fingerprint, cache)
    d.edge(cache, validate_cached)
    d.edge(validate_cached, complete_cached)
    d.edge(cache, authorized)
    d.edge(authorized, request)
    d.edge(authorized, error)
    d.edge(request, parse)
    d.edge(parse, store)
    d.edge(store, complete)
    d.edge(request, retry, dashed=True, exit=(0.3, 1), entry=(0.3, 0))
    d.edge(retry, request, dashed=True, exit=(0.7, 0), entry=(0.7, 1))
    d.edge(retry, error)
    d.edge(error, complete)
    d.label("HIT", 950, 105, 60, 30, 14)
    d.label("MISS", 720, 270, 80, 30, 14)
    d.label("ALLOW", 965, 325, 80, 30, 14)
    d.label("DENY", 700, 580, 80, 30, 14)
    return d.write()


def run_sequence() -> Path:
    d = Diagram("run_sequence")
    d.label("Public Python interaction and run_task sequence", 350, 25, 900, 55, 30)
    lanes = [
        ("Analyst", 70, "neutral"),
        ("Notebook", 330, "neutral"),
        ("run_task", 590, "control"),
        ("Task + SceneSet", 850, "contract"),
        ("CV / VLM", 1110, "compute"),
        ("Validation + GIS", 1370, "evidence"),
    ]
    headers = {}
    for name, x, role in lanes:
        headers[name] = d.vertex(name, x, 105, 190, 65, role=role, bold=True)
        d.vertex("", x + 93, 170, 4, 640, role="neutral", rounded=False, dashed=True)

    y = 220
    notebook = d.vertex("Select manifest, task and config", 285, y, 280, 58, role="neutral")
    d.edge(headers["Analyst"], notebook, exit=(0.5, 1), entry=(0, 0.5))
    y += 90
    call = d.vertex("run_task(...)\nPython API", 555, y, 260, 70, role="control", bold=True)
    d.edge(notebook, call)
    y += 100
    load = d.vertex("Load + validate contracts", 820, y, 250, 65, role="contract")
    d.edge(call, load)
    y += 95
    analyse = d.vertex("Extract / read analyzer tables", 1075, y, 260, 65, role="compute")
    d.edge(load, analyse)
    y += 95
    harmonise = d.vertex("Harmonise + preserve states", 555, y, 260, 65, role="control")
    d.edge(analyse, harmonise)
    y += 95
    products = d.vertex("Validate + write managed products", 1325, y, 260, 70, role="evidence")
    d.edge(harmonise, products)
    y += 105
    result = d.vertex(
        "Analyst inspects tables + quality + provenance\nin notebook and QGIS",
        190,
        y,
        360,
        85,
        role="evidence",
        bold=True,
    )
    d.edge(products, result)
    return d.write()


def bpmn_process() -> Path:
    d = Diagram("bpmn_process")
    d.label("PyRestore BPMN-style execution process", 350, 25, 900, 55, 30)
    start = d.shaped_vertex("Start", 50, 350, 85, 85, role="neutral", shape="ellipse", bold=True)
    inputs = d.vertex("Load manifest or pairs\nand TaskDefinition", 190, 325, 250, 130, role="contract", bold=True)
    gateway = d.shaped_vertex("Contract\nvalid?", 500, 335, 120, 120, role="control", shape="rhombus", bold=True)
    reject = d.vertex("Record contract error\nand stop publication", 480, 600, 250, 100, role="danger", bold=True)
    cv = d.vertex("CV activity\nextract or read physical fields", 700, 210, 300, 110, role="compute", bold=True)
    vlm = d.vertex("VLM activity\nscore or read semantic fields", 700, 470, 300, 110, role="compute", bold=True)
    join = d.shaped_vertex("Join", 1060, 335, 110, 110, role="control", shape="rhombus", bold=True)
    harmonise = d.vertex("Harmonise\nretain success and error rows", 1230, 320, 300, 140, role="evidence", bold=True)
    validate = d.vertex("Validate and spatially screen\nonly eligible values", 1040, 610, 320, 110, role="evidence")
    publish = d.vertex("Stage and publish\nmanaged product family", 620, 720, 300, 110, role="control", bold=True)
    end = d.shaped_vertex("End", 390, 730, 85, 85, role="evidence", shape="ellipse", bold=True)
    d.edge(start, inputs)
    d.edge(inputs, gateway)
    d.edge(gateway, cv, "yes")
    d.edge(gateway, vlm)
    d.edge(gateway, reject, "no")
    d.edge(cv, join)
    d.edge(vlm, join)
    d.edge(join, harmonise)
    d.edge(harmonise, validate)
    d.edge(validate, publish)
    d.edge(publish, end)
    return d.write()


def module_uml() -> Path:
    d = Diagram("module_uml")
    d.label("PyRestore UML module and dependency model", 330, 25, 940, 55, 30)
    notebook = d.vertex(
        "Notebook client\nPython API calls", 60, 330, 250, 105, role="neutral", bold=True, rounded=False
    )
    pipeline = d.vertex(
        "pipeline.py\nrun_task orchestration\nstaged publication",
        390,
        300,
        300,
        165,
        role="control",
        bold=True,
        rounded=False,
    )
    manifest = d.vertex(
        "manifest.py / pairs.py\nSite + Capture + SceneSet",
        780,
        130,
        310,
        125,
        role="contract",
        bold=True,
        rounded=False,
    )
    config = d.vertex("config.py\nruntime and privacy", 1160, 130, 280, 125, role="contract", bold=True, rounded=False)
    cv = d.vertex("cv.py\nphysical estimates", 780, 350, 270, 110, role="compute", bold=True, rounded=False)
    vlm = d.vertex(
        "vlm.py + cache.py\nsemantic execution + identity",
        1120,
        340,
        350,
        130,
        role="compute",
        bold=True,
        rounded=False,
    )
    harmonise = d.vertex(
        "harmonise.py\nfailure-aware joins", 750, 590, 280, 115, role="evidence", bold=True, rounded=False
    )
    validate = d.vertex(
        "validate.py / map.py\nmetrics + spatial screening",
        1120,
        580,
        350,
        135,
        role="evidence",
        bold=True,
        rounded=False,
    )
    products = d.vertex(
        "CSV + GeoPackage\nquality + validation + provenance",
        430,
        690,
        340,
        125,
        role="evidence",
        bold=True,
        rounded=False,
    )
    d.edge(notebook, pipeline, dashed=True)
    d.edge(pipeline, manifest, dashed=True)
    d.edge(pipeline, config, dashed=True)
    d.edge(pipeline, cv, dashed=True, exit=(1, 0.5), entry=(0, 0.5))
    d.edge(pipeline, vlm, dashed=True, exit=(1, 0.65), entry=(0, 0.25))
    d.edge(pipeline, harmonise, dashed=True, exit=(0.3, 1), entry=(0, 0.2))
    d.edge(pipeline, validate, dashed=True, exit=(0.7, 1), entry=(0.15, 0))
    d.edge(harmonise, products, exit=(0, 1), entry=(1, 0.2))
    d.edge(validate, products, exit=(0.5, 1), entry=(1, 0.8))
    return d.write()


def main() -> int:
    paths = [
        framework_overview(),
        data_model(),
        request_state(),
        run_sequence(),
        bpmn_process(),
        module_uml(),
    ]
    print("generated", ", ".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
