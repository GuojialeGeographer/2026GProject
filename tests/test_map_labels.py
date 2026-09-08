import geopandas as gpd
from shapely.geometry import Point

from pyrestore.map import make_map


def test_high_candidate_map_has_correct_label(tmp_path):
    data = gpd.GeoDataFrame(
        {"score": [0.9], "candidate_high_cluster": [True]},
        geometry=[Point(114, 22.6)], crs=4326,
    )
    target = tmp_path / "map.html"
    make_map(data, "score", str(target), candidate_col="candidate_high_cluster")
    html = target.read_text()
    assert "Candidate high-score cluster" in html
    assert "significant High-High" in html
    assert "Candidate low-score" not in html
