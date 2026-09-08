"""VLM layer tests: task-driven parsing, caching, offline mode, failure records."""
from __future__ import annotations

import pandas as pd
import pytest

from pyrestore.vlm import (
    apply_derived_fields,
    load_task,
    parse_response,
    request_hash,
    score_vlm_batch,
    task_prompt_hash,
)
from tests.conftest import FakeVLMClient, make_image, prs11_reply, renewal_reply


@pytest.fixture
def prs11_task(prs11_task_file):
    return load_task(prs11_task_file)


@pytest.fixture
def renewal_task(renewal_task_file):
    return load_task(renewal_task_file)


def test_parse_response_handles_fences_and_prose(prs11_task):
    content = f"Sure! Here is my assessment:\n```json\n{prs11_reply()}\n```"
    response = parse_response(content, prs11_task)
    assert response["being_away"] == 0.4 and response["scope"] == 0.6


def test_parse_response_missing_field_raises(prs11_task):
    with pytest.raises(ValueError, match="missing output field"):
        parse_response('{"being_away": 0.4, "coherence": 0.5, "scope": 0.6}', prs11_task)


def test_parse_response_out_of_range_score_raises(prs11_task):
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_response(prs11_reply(being_away=1.7), prs11_task)


def test_parse_response_invalid_category_raises(renewal_task):
    content = renewal_reply(detected="maybe")
    with pytest.raises(ValueError, match="not in options"):
        parse_response(content, renewal_task)


def test_renewal_nested_categories_and_confidence_are_enforced(renewal_task):
    bad_type = renewal_reply().replace("greening_landscape", "invented_type")
    with pytest.raises(ValueError, match="not in options"):
        parse_response(bad_type, renewal_task)
    bad_confidence = renewal_reply().replace('"confidence": 0.8', '"confidence": 9.5')
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_response(bad_confidence, renewal_task)


def test_parse_response_list_field_coerces_to_strings(renewal_task):
    content = (
        '{"renewal_detected": "yes", "renewal_types": "greening_landscape", '
        '"nuisance": [], "confidence": 0.8, "evidence": "x"}'
    )
    with pytest.raises(ValueError, match="must be a list"):
        parse_response(content, renewal_task)


def test_derived_fields_mean(prs11_task):
    response = parse_response(prs11_reply(being_away=0.2, coherence=0.4, scope=0.6, fascination=0.8), prs11_task)
    derived = apply_derived_fields(response, prs11_task)
    assert derived["restorative_average"] == pytest.approx(0.5)


def test_prompt_hash_changes_with_output_contract(prs11_task, tmp_path):
    from tests.conftest import write_task

    other = load_task(
        write_task(
            tmp_path,
            "test_prs11",
            fields={
                "being_away": "score",
                "coherence": "score",
                "scope": "score",
                "fascination": "score",
                "extra": "score",
            },
            template_text=prs11_reply(),
        )
    )
    assert task_prompt_hash(prs11_task) != task_prompt_hash(other)


def test_hashes_cover_temperature_and_request_preprocessing(prs11_task, offline_cfg):
    changed_task = {**prs11_task, "prompt": {**prs11_task["prompt"], "temperature": 0.9}}
    assert task_prompt_hash(prs11_task) != task_prompt_hash(changed_task)
    changed_cfg = {
        **offline_cfg,
        "vlm": {**offline_cfg["vlm"], "max_image_px": 512},
    }
    assert request_hash(prs11_task, offline_cfg) != request_hash(prs11_task, changed_cfg)


def test_batch_with_fake_client_and_cache_reuse(tmp_path, manifest, prs11_task, offline_cfg):
    client = FakeVLMClient(prs11_reply())
    cfg = {**offline_cfg, "vlm": {**offline_cfg["vlm"], "offline": False}}
    first = score_vlm_batch(manifest, prs11_task, cfg, client=client)
    assert (first["vlm_status"] == "ok").all()
    assert (first["cache_hit"] == False).all()  # noqa: E712
    assert client.calls == 6
    assert "vlm_restorative_average" in first.columns

    second = score_vlm_batch(manifest, prs11_task, cfg, client=client)
    assert (second["cache_hit"] == True).all()  # noqa: E712
    assert client.calls == 6  # cache served every request
    pd.testing.assert_frame_equal(first.drop(columns=["cache_hit"]), second.drop(columns=["cache_hit"]))


def test_batch_preserves_failed_records(tmp_path, manifest, prs11_task, offline_cfg):
    from pyrestore.vlm import image_data_url

    cfg = {**offline_cfg, "vlm": {**offline_cfg["vlm"], "offline": False}}
    # Fail (persistently, across retries) only the record whose payload carries c3's image.
    broken = manifest.loc[manifest["capture_id"] == "c3"].iloc[0]
    marker = image_data_url(broken["image_path"], 1024)
    client = FakeVLMClient(prs11_reply(), fail_when=lambda sent: marker in sent)
    table = score_vlm_batch(manifest, prs11_task, cfg, client=client)
    assert len(table) == 6
    failed = table[table["capture_id"] == "c3"]
    ok = table[table["capture_id"] != "c3"]
    assert (failed["vlm_status"] == "error").all()
    assert "synthetic permanent failure" in failed["vlm_error"].iloc[0]
    assert (ok["vlm_status"] == "ok").all()


def test_offline_mode_records_failures(tmp_path, manifest, prs11_task, offline_cfg):
    # The batch never raises: offline cache misses become per-record failures.
    table = score_vlm_batch(manifest, prs11_task, offline_cfg)
    assert (table["vlm_status"] == "error").all()
    assert table["vlm_error"].iloc[0].startswith("offline mode")


def test_live_default_client_requires_remote_image_authorisation(
    manifest, prs11_task, offline_cfg
):
    cfg = {**offline_cfg, "vlm": {**offline_cfg["vlm"], "offline": False, "cache_db": ""}}
    table = score_vlm_batch(manifest.head(1), prs11_task, cfg)
    assert table.loc[0, "vlm_status"] == "error"
    assert "Remote image transmission is disabled" in table.loc[0, "vlm_error"]


def test_pair_cache_keys_distinguish_followup_images(tmp_path, renewal_task_file, offline_cfg):
    """Regression: hashing only the baseline let a different follow-up reuse a cached judgement."""
    task = load_task(renewal_task_file)
    cfg = {**offline_cfg, "vlm": {**offline_cfg["vlm"], "offline": False}}
    baseline = make_image(tmp_path / "base.png", color=(10, 10, 10))
    followup_1 = make_image(tmp_path / "f1.png", color=(20, 10, 10))
    followup_2 = make_image(tmp_path / "f2.png", color=(30, 10, 10))
    pairs = pd.DataFrame(
        [
            {"pair_id": "p1", "before_image_path": str(baseline), "after_image_path": str(followup_1)},
            {"pair_id": "p2", "before_image_path": str(baseline), "after_image_path": str(followup_2)},
        ]
    )
    client = FakeVLMClient(renewal_reply())
    first = score_vlm_batch(pairs, task, cfg, client=client)
    assert (first["vlm_status"] == "ok").all()
    assert client.calls == 2, "different follow-up images must not share a cache entry"

    second = score_vlm_batch(pairs, task, cfg, client=client)
    assert (second["cache_hit"] == True).all()  # noqa: E712
    assert client.calls == 2  # both served from cache on the second run


def test_image_data_url_roundtrip_and_downscale(tmp_path):
    """Regression guard for the image encoding path (imports, JPEG re-encode, downscaling)."""
    import base64
    import io

    from PIL import Image

    from pyrestore.vlm import image_data_url

    tiny = make_image(tmp_path / "tiny.png", size=(32, 24))
    url = image_data_url(str(tiny))
    assert url.startswith("data:image/jpeg;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert decoded.size == (32, 24)

    wide = make_image(tmp_path / "wide.png", size=(2048, 8))
    small_url = image_data_url(str(wide), max_px=1024)
    small = Image.open(io.BytesIO(base64.b64decode(small_url.split(",", 1)[1])))
    assert max(small.size) <= 1024
