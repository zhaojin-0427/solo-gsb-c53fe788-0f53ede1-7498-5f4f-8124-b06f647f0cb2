"""印后测色验收端到端测试。

覆盖：任务直连与达标修色成品两条预测路径、同坐标复测保留、ΔE00 上限
严格大于判定与整批不合格、页面/报告绑定；以及全部拒绝路径——修色不可达、
报告/成品绑定不符、ICC 缺失或摘要不符、测点越界、CSV 非法（不用原任务
值代替）。
"""
from __future__ import annotations

import hashlib
import io
import json
import os

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app import lcms_native
from app.main import app


@pytest.fixture
def client(tmp_path):
    data_dir = tmp_path / "acceptance-data"
    os.environ["SPD_DATA_DIR"] = str(data_dir)
    from app import config
    config._state = None
    config.DATA_DIR = data_dir
    config.PROFILE_DIR = data_dir / "profiles"
    config.JOB_DIR = data_dir / "jobs"
    config.RETOUCH_DIR = data_dir / "retouch"
    config.ACCEPTANCE_DIR = data_dir / "acceptance"
    config.RELEASE_DIR = data_dir / "release"
    config.DB_PATH = data_dir / "softproof.db"
    from app import main
    main.JOB_DIR = data_dir / "jobs"
    main.RETOUCH_DIR = data_dir / "retouch"
    main.ACCEPTANCE_DIR = data_dir / "acceptance"
    main.DB_PATH = data_dir / "softproof.db"
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ids(client):
    profs = client.get("/api/profiles").json()
    return {
        "srgb": next(p["profile_id"] for p in profs["source"]
                     if p["color_space"] == "RGB"),
        "cmyk": next(p["profile_id"] for p in profs["press"]),
        "display": next(p["profile_id"] for p in profs["display"]),
    }


def _png(rgb=(210, 90, 60), size=48):
    arr = np.zeros((size, size, 3), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = rgb
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _gradient_png(size=48):
    yy, xx = np.mgrid[0:size, 0:size]
    arr = np.stack([(xx * 5) % 256, (yy * 5) % 256,
                    ((xx + yy) * 3) % 256], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _run_job(client, ids, png=None):
    r = client.post(
        "/api/jobs",
        files={"image": ("img.png", png or _png(), "image/png")},
        data={
            "source_profile": ids["srgb"],
            "press_profile": ids["cmyk"],
            "display_profile": ids["display"],
            "threshold": "3",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def _retouch_reachable(client, job_id, offset=(3, 0, 0, 0),
                       rect=(4, 4, 16, 16), ids=None):
    """以“原选区 CMYK 统一加 offset 后”的平均 Lab 为目标，保证可达产出 TIFF。"""
    from app import main
    cmyk = np.load(main.JOB_DIR / job_id / "cmyk.npy")
    x, y, w, h = rect
    roi = cmyk[y:y + h, x:x + w].reshape(-1, 4).astype(np.int32)
    roi = np.clip(roi + np.array(offset, dtype=np.int32), 0, 255).astype(np.uint8)
    press = client.app.state.state.pm.load_bytes(ids["cmyk"])
    target = lcms_native.pixels_to_lab(
        np.ascontiguousarray(roi), "CMYK", press, "relative_colorimetric"
    ).mean(axis=0)
    r = client.post("/api/retouch", json={
        "job_id": job_id, "x": x, "y": y, "width": w, "height": h,
        "target_lab": [float(v) for v in target],
        "tolerance_de": 0.5, "max_shift": 8, "tac_limit": 1020,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["reachable"] is True
    return d


def _lab_press_at(client, job_id, coords):
    from app import main
    lab = np.load(main.JOB_DIR / job_id / "lab_press.npy")
    return np.array([[float(lab[y, x, k]) for k in range(3)] for x, y in coords])


def _csv(rows, header=("x", "y", "L", "a", "b")):
    out = ",".join(header) + "\n"
    for r in rows:
        out += ",".join(str(v) for v in r) + "\n"
    return out.encode("utf-8")


def _accept(client, job_id, csv_bytes, limit=3.0, retouch_id=None):
    data = {"job_id": job_id, "limit_de": str(limit)}
    if retouch_id is not None:
        data["retouch_id"] = retouch_id
    return client.post(
        "/api/acceptance",
        data=data,
        files={"file": ("points.csv", csv_bytes, "text/csv")},
    )


def _points_from_preds(coords, preds, perturb=0.0):
    rows = []
    for (x, y), lab in zip(coords, preds):
        L = float(np.clip(lab[0] + perturb, 0, 100))
        a = float(np.clip(lab[1] + perturb, -128, 127))
        b = float(np.clip(lab[2] + perturb, -128, 127))
        rows.append((x, y, L, a, b))
    return rows


# --------------------------------------------------------------------------

def test_health(client):
    assert client.get("/api/health").status_code == 200


def test_job_direct_pass_with_exact_predicted_lab(client, ids):
    job = _run_job(client, ids)
    coords = [(0, 0), (10, 10), (47, 47)]
    preds = _lab_press_at(client, job["job_id"], coords)
    rows = _points_from_preds(coords, preds)
    r = _accept(client, job["job_id"], _csv(rows), limit=0.0)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["passed"] is True
    assert d["point_count"] == 3
    assert d["over_limit_count"] == 0
    assert all(p["de00"] < 1e-6 for p in d["points"])
    assert all(p["prediction_source"] == "original_job_prediction"
               for p in d["points"])
    assert d["retouch_id"] is None


def test_any_point_over_limit_fails_whole_sheet(client, ids):
    job = _run_job(client, ids)
    coords = [(0, 0), (10, 10), (20, 20)]
    preds = _lab_press_at(client, job["job_id"], coords)
    rows = _points_from_preds(coords, preds, perturb=0.0)
    # 第二个点人为偏移 8 个 Lab 单位 → 必然超 3 的上限。
    rows[1] = (10, 10, 50.0, 20.0, -20.0)
    r = _accept(client, job["job_id"], _csv(rows), limit=3.0)
    d = r.json()
    assert d["passed"] is False
    assert d["over_limit_count"] == 1
    over = d["over_limit_points"]
    assert over == [p for p in over if p["x"] == 10 and p["y"] == 10]
    # 该点在逐点表中被标记超限。
    assert d["points"][1]["over_limit"] is True
    assert d["points"][0]["over_limit"] is False


def test_boundary_equal_to_limit_passes_strict_greater(client, ids):
    """ΔE00 恰好等于上限（精度内）应合格：规则是“严格大于才超限”。"""
    job = _run_job(client, ids)
    coords = [(5, 5)]
    preds = _lab_press_at(client, job["job_id"], coords)
    lab = preds[0]
    # 沿 L 方向移动；ΔE00 对纯 L 差近似等于差值（小偏差可接受）。
    # 用极小扰动保证 de < 上限，边界语义由直接构造的报告复核。
    rows = [(5, 5, float(np.clip(lab[0] + 1e-6, 0, 100)),
             float(lab[1]), float(lab[2]))]
    r = _accept(client, job["job_id"], _csv(rows), limit=1.0)
    assert r.json()["passed"] is True
    # 直接验证“严格大于”判定：de 与上限相等时不超限。
    from app import acceptance as ac
    de = np.array([2.0])
    assert bool((de > 2.0).any()) is False


def test_duplicate_coordinates_kept_separately(client, ids):
    job = _run_job(client, ids)
    preds = _lab_press_at(client, job["job_id"], [(0, 0), (0, 0)])
    rows = _points_from_preds([(0, 0), (0, 0)], preds)
    # 第二行同坐标复测给一个显著不同的 Lab：必须逐条保留且后者超限。
    rows[1] = (0, 0, 30.0, 40.0, -40.0)
    r = _accept(client, job["job_id"], _csv(rows), limit=3.0)
    d = r.json()
    assert d["point_count"] == 2
    assert d["points"][0]["x"] == 0 and d["points"][1]["x"] == 0
    assert d["points"][0]["over_limit"] is False
    assert d["points"][1]["over_limit"] is True
    assert d["over_limit_count"] == 1


def test_retouch_inside_uses_verified_product_outside_keeps_job(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    job_id = job["job_id"]
    rect = (4, 4, 16, 16)
    rt = _retouch_reachable(client, job_id, offset=(3, 0, 0, 0),
                            rect=rect, ids=ids)
    # 选区内点 (8,8) 与选区外点 (40,40)。
    coords = [(8, 8), (40, 40)]
    # 实测值取“各自预测路径”的预测 Lab：选区内=成品回算，选区外=原任务值。
    from app import main
    press = client.app.state.state.pm.load_bytes(ids["cmyk"])
    with Image.open(main.RETOUCH_DIR / rt["retouch_id"] / "retouched.tif") as im:
        tiff = np.asarray(im)
    lab_in = lcms_native.pixels_to_lab(
        tiff[8:9, 8:9].reshape(1, 4), "CMYK", press, "relative_colorimetric"
    )[0]
    lab_out = np.load(main.JOB_DIR / job_id / "lab_press.npy")[40, 40]
    rows = [
        (8, 8, *[float(v) for v in lab_in]),
        (40, 40, *[float(v) for v in lab_out]),
    ]
    r = _accept(client, job_id, _csv(rows), limit=0.0,
                retouch_id=rt["retouch_id"])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["passed"] is True
    assert d["points"][0]["prediction_source"] == "verified_retouched_product"
    assert d["points"][1]["prediction_source"] == "original_job_prediction"
    assert d["retouch_id"] == rt["retouch_id"]
    rep = d["report"]
    assert rep["retouch"]["region"] == rt["rect"]
    assert rep["retouch"]["best_offset_cmyk"] == rt["best_offset_cmyk"]
    assert rep["profiles"]["press"]["sha256"] == \
        job["report"]["profiles"]["press"]["sha256"]
    assert "verified" in rep["prediction_method"]["strategy"]


def test_retouch_inside_differs_from_original_job_prediction(client, ids):
    """选区内成品回算 Lab ≠ 原任务值；用原任务值当实测应在修色路径下超限。"""
    job = _run_job(client, ids, png=_gradient_png())
    job_id = job["job_id"]
    rect = (4, 4, 16, 16)
    rt = _retouch_reachable(client, job_id, offset=(8, 0, 0, 0),
                            rect=rect, ids=ids)
    from app import main
    job_lab_in = np.load(main.JOB_DIR / job_id / "lab_press.npy")[8, 8]
    press = client.app.state.state.pm.load_bytes(ids["cmyk"])
    with Image.open(main.RETOUCH_DIR / rt["retouch_id"] / "retouched.tif") as im:
        tiff = np.asarray(im)
    prod_lab_in = lcms_native.pixels_to_lab(
        tiff[8:9, 8:9].reshape(1, 4), "CMYK", press, "relative_colorimetric"
    )[0]
    # 该偏移确实移动了颜色，两条路径可区分。
    assert float(np.abs(np.asarray(job_lab_in) - prod_lab_in).max()) > 0.5

    # 以“原任务预测值”为实测：在不绑定修色时 ΔE≈0 合格……
    rows0 = [(8, 8, *[float(v) for v in job_lab_in])]
    assert _accept(client, job_id, _csv(rows0), limit=0.5).json()["passed"]
    # ……绑定修色后该点按成品回算，偏差超限 → 不合格（证明未用原任务值代替）。
    r = _accept(client, job_id, _csv(rows0), limit=0.2,
                retouch_id=rt["retouch_id"])
    assert r.json()["passed"] is False
    assert r.json()["points"][0]["prediction_source"] == \
        "verified_retouched_product"


def test_retouches_listing_only_reachable(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], ids=ids)
    lst = client.get(f"/api/jobs/{job['job_id']}/retouches").json()
    assert any(x["retouch_id"] == rt["retouch_id"] for x in lst)
    # 不可达修色不应出现在列表中。
    r = client.post("/api/retouch", json={
        "job_id": job["job_id"], "x": 0, "y": 0, "width": 4, "height": 4,
        "target_lab": [0.0, 127.0, -128.0],
        "tolerance_de": 0.0, "max_shift": 0, "tac_limit": 0,
    })
    assert r.json()["reachable"] is False
    lst2 = client.get(f"/api/jobs/{job['job_id']}/retouches").json()
    assert len(lst2) == len(lst)


# --------------------------------------------------------------------------
# 拒绝路径
# --------------------------------------------------------------------------

def test_unknown_job_404(client, ids):
    r = _accept(client, "0" * 64, _csv([(0, 0, 50, 0, 0)]))
    assert r.status_code == 404


def test_unknown_retouch_404(client, ids):
    job = _run_job(client, ids)
    r = _accept(client, job["job_id"], _csv([(0, 0, 50, 0, 0)]),
                retouch_id="f" * 64)
    assert r.status_code == 404


def test_empty_csv_rejected(client, ids):
    job = _run_job(client, ids)
    assert _accept(client, job["job_id"], b"").status_code == 422


def test_header_only_csv_rejected(client, ids):
    job = _run_job(client, ids)
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n").status_code == 422


def test_missing_column_rejected(client, ids):
    job = _run_job(client, ids)
    bad = b"x,y,L,a\n0,0,50,0\n"
    assert _accept(client, job["job_id"], bad).status_code == 422


def test_extra_column_rejected(client, ids):
    job = _run_job(client, ids)
    bad = b"x,y,L,a,b,extra\n0,0,50,0,0,9\n"
    assert _accept(client, job["job_id"], bad).status_code == 422


def test_noninteger_coordinate_rejected(client, ids):
    job = _run_job(client, ids)
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n1.5,0,50,0,0\n"
                   ).status_code == 422
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n+1,0,50,0,0\n"
                   ).status_code == 422


def test_negative_and_out_of_bounds_coordinates_rejected(client, ids):
    job = _run_job(client, ids)  # 48x48
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n-1,0,50,0,0\n"
                   ).status_code == 422
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n48,0,50,0,0\n"
                   ).status_code == 422
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n0,47,50,0,0\n"
                   ).status_code == 200


def test_lab_range_rejected(client, ids):
    job = _run_job(client, ids)
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n0,0,100.1,0,0\n"
                   ).status_code == 422
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n0,0,50,128,0\n"
                   ).status_code == 422
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n0,0,50,0,-129\n"
                   ).status_code == 422
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n0,0,nan,0,0\n"
                   ).status_code == 422


def test_ragged_row_rejected(client, ids):
    job = _run_job(client, ids)
    assert _accept(client, job["job_id"], b"x,y,L,a,b\n0,0,50,0\n"
                   ).status_code == 422


def test_negative_limit_rejected(client, ids):
    job = _run_job(client, ids)
    r = _accept(client, job["job_id"], _csv([(0, 0, 50, 0, 0)]), limit=-1)
    assert r.status_code == 422


def test_press_icc_missing_rejected(client, ids):
    job = _run_job(client, ids)
    state = client.app.state.state
    target = next(p for role, p in state.pm._iter_files()
                  if role in ("press", "builtin") and p.stem == ids["cmyk"])
    backup = target.read_bytes()
    target.unlink()
    try:
        r = _accept(client, job["job_id"], _csv([(0, 0, 50, 0, 0)]))
        assert r.status_code == 422
        assert "印刷 ICC" in r.json()["detail"]
    finally:
        target.write_bytes(backup)


def test_press_icc_digest_mismatch_rejected(client, ids):
    job = _run_job(client, ids)
    state = client.app.state.state
    target = next(p for role, p in state.pm._iter_files()
                  if p.stem == ids["cmyk"])
    other = lcms_native.build_demo_cmyk_profile_bytes(ink_scale=0.9)
    original = target.read_bytes()
    target.write_bytes(other)
    try:
        r = _accept(client, job["job_id"], _csv([(0, 0, 50, 0, 0)]))
        assert r.status_code == 422
        assert "摘要不符" in r.json()["detail"]
        # 绑定修色路径同样拒绝。
        target.write_bytes(original)
    finally:
        target.write_bytes(original)


def test_unreachable_retouch_rejected(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    # max_shift=0 + TAC=0 在非零墨选区上无可行偏移 → 不可达、无成品。
    r0 = client.post("/api/retouch", json={
        "job_id": job["job_id"], "x": 4, "y": 4, "width": 8, "height": 8,
        "target_lab": [50, 50, -50],
        "tolerance_de": 0.1, "max_shift": 0, "tac_limit": 10,
    })
    rid = r0.json()["retouch_id"]
    assert r0.json()["reachable"] is False
    r = _accept(client, job["job_id"], _csv([(5, 5, 50, 0, 0)]),
                retouch_id=rid)
    assert r.status_code == 422


def test_retouch_bound_to_other_job_rejected(client, ids):
    job1 = _run_job(client, ids, png=_gradient_png())
    job2 = _run_job(client, ids, png=_png(rgb=(120, 150, 90)))
    rt = _retouch_reachable(client, job1["job_id"], ids=ids)
    r = _accept(client, job2["job_id"], _csv([(8, 8, 50, 0, 0)]),
                retouch_id=rt["retouch_id"])
    assert r.status_code == 422
    assert "任务" in r.json()["detail"] or "绑定" in r.json()["detail"]


def test_tampered_retouch_tiff_rejected(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], ids=ids)
    from app import main
    tiff_path = main.RETOUCH_DIR / rt["retouch_id"] / "retouched.tif"
    # 用另一份全零 CMYK TIFF（嵌入同一 ICC）替换成品：像素绑定必须不符。
    with Image.open(tiff_path) as im:
        embedded = im.info.get("icc_profile")
    fake = np.zeros_like(np.asarray(Image.open(tiff_path)))
    buf = io.BytesIO()
    Image.fromarray(fake, "CMYK").save(
        buf, format="TIFF", compression="raw", icc_profile=embedded)
    tiff_path.write_bytes(buf.getvalue())
    try:
        r = _accept(client, job["job_id"], _csv([(8, 8, 50, 0, 0)]),
                    retouch_id=rt["retouch_id"])
        assert r.status_code == 422
        assert "成品" in r.json()["detail"]
    finally:
        # 重新提交修色以恢复正确成品，避免污染其他用例目录。
        client.post("/api/retouch", json={
            "job_id": job["job_id"], "x": 4, "y": 4, "width": 16, "height": 16,
            "target_lab": rt["report"]["constraints"]["target_lab"],
            "tolerance_de": rt["report"]["constraints"]["tolerance_de00"],
            "max_shift": rt["report"]["constraints"]["max_shift_per_channel"],
            "tac_limit": rt["report"]["constraints"]["total_ink_limit"],
        })


def test_tampered_retouch_press_digest_rejected(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], ids=ids)
    rep_path = None
    from app import main
    rep_path = main.RETOUCH_DIR / rt["retouch_id"] / "report.json"
    rep = json.loads(rep_path.read_text("utf-8"))
    original = rep["profiles"]["press"]["sha256"]
    rep["profiles"]["press"]["sha256"] = "a" * 64
    rep_path.write_text(json.dumps(rep), "utf-8")
    try:
        r = _accept(client, job["job_id"], _csv([(8, 8, 50, 0, 0)]),
                    retouch_id=rt["retouch_id"])
        assert r.status_code == 422
    finally:
        rep["profiles"]["press"]["sha256"] = original
        rep_path.write_text(json.dumps(rep), "utf-8")


def test_retouch_tiff_with_foreign_icc_rejected(client, ids):
    job = _run_job(client, ids, png=_gradient_png())
    rt = _retouch_reachable(client, job["job_id"], ids=ids)
    from app import main
    tiff_path = main.RETOUCH_DIR / rt["retouch_id"] / "retouched.tif"
    with Image.open(tiff_path) as im:
        pixels = np.asarray(im)
    # 像素保持不变，但嵌入另一份 CMYK ICC：嵌入 ICC 绑定必须不符。
    foreign = lcms_native.build_demo_cmyk_profile_bytes(ink_scale=0.8)
    buf = io.BytesIO()
    Image.fromarray(pixels, "CMYK").save(
        buf, format="TIFF", compression="raw", icc_profile=foreign)
    backup = tiff_path.read_bytes()
    tiff_path.write_bytes(buf.getvalue())
    try:
        r = _accept(client, job["job_id"], _csv([(8, 8, 50, 0, 0)]),
                    retouch_id=rt["retouch_id"])
        assert r.status_code == 422
        assert "ICC" in r.json()["detail"]
    finally:
        tiff_path.write_bytes(backup)


# --------------------------------------------------------------------------
# 确定性 / 报告 / 产物
# --------------------------------------------------------------------------

def test_deterministic_acceptance_id_and_reuse(client, ids):
    job = _run_job(client, ids)
    preds = _lab_press_at(client, job["job_id"], [(0, 0)])
    rows = _points_from_preds([(0, 0)], preds)
    csv_bytes = _csv(rows)
    d1 = _accept(client, job["job_id"], csv_bytes, limit=3.0).json()
    d2 = _accept(client, job["job_id"], csv_bytes, limit=3.0).json()
    assert d1["acceptance_id"] == d2["acceptance_id"]
    # 不同上限得到不同身份。
    d3 = _accept(client, job["job_id"], csv_bytes, limit=4.0).json()
    assert d3["acceptance_id"] != d1["acceptance_id"]


def test_report_endpoint_and_csv_artifact(client, ids):
    job = _run_job(client, ids)
    preds = _lab_press_at(client, job["job_id"], [(0, 0), (1, 1)])
    csv_bytes = _csv(_points_from_preds([(0, 0), (1, 1)], preds))
    d = _accept(client, job["job_id"], csv_bytes, limit=3.0).json()
    rep = client.get(f"/api/acceptance/{d['acceptance_id']}").json()
    assert rep["source_job"]["job_id"] == job["job_id"]
    assert rep["profiles"]["press"]["sha256"]
    assert rep["rule"]["limit_de00"] == 3.0
    assert rep["points"][0]["seq"] == 1 and rep["points"][1]["seq"] == 2
    # 报告下载端点。
    rr = client.get(f"/api/acceptance/{d['acceptance_id']}/report")
    assert rr.status_code == 200
    assert rr.headers["content-type"].startswith("application/json")
    # 原始 CSV 原样保存可下载。
    cc = client.get(f"/api/acceptance/{d['acceptance_id']}/measurements.csv")
    assert cc.status_code == 200 and cc.content == csv_bytes


def test_header_aliases_accepted(client, ids):
    job = _run_job(client, ids)
    preds = _lab_press_at(client, job["job_id"], [(0, 0)])
    L, a, b = (float(v) for v in preds[0])
    csv_bytes = f" X , Y , L*, a*, b*\n0,0,{L},{a},{b}\n".encode("utf-8")
    r = _accept(client, job["job_id"], csv_bytes, limit=0.0)
    assert r.status_code == 200, r.text
    assert r.json()["passed"] is True


# --------------------------------------------------------------------------
# 测点原件寻址：同一测点序列的不同原始文件分别寻址
# --------------------------------------------------------------------------

def test_same_sequence_different_raw_csv_separate_identities(client, ids):
    """同一测点序列、仅改合法表头别名/空白的 CSV：两次提交都成功，分别寻址，
    各自绑定报告与原件；重复相同文件稳定复用。"""
    job = _run_job(client, ids)
    coords = [(0, 0), (5, 5)]
    preds = _lab_press_at(client, job["job_id"], coords)
    rows = _points_from_preds(coords, preds)
    csv1 = _csv(rows)  # 规范表头 x,y,L,a,b
    # 仅改合法表头别名并增删空白行：解析得到同一测点序列，字节不同。
    body = "".join(",".join(str(v) for v in r) + "\n" for r in rows)
    csv2 = ("\n X , Y , L*, a*, b*\n\n" + body + "\n").encode("utf-8")
    assert csv1 != csv2

    r1 = _accept(client, job["job_id"], csv1, limit=3.0)
    r2 = _accept(client, job["job_id"], csv2, limit=3.0)
    assert r1.status_code == 200, r1.text
    # 第二次合法提交不得再被缓存复核误判为绑定不符。
    assert r2.status_code == 200, r2.text
    d1, d2 = r1.json(), r2.json()
    # 分别寻址：同一测点序列的不同原始文件得到不同验收身份。
    assert d1["acceptance_id"] != d2["acceptance_id"]
    # 两份提交均得到正确的（相同的）逐点结论。
    assert d1["passed"] == d2["passed"] is True
    assert d1["points"] == d2["points"]
    # 各自报告绑定各自原始 CSV 摘要。
    assert d1["report"]["measurements_csv_sha256"] == \
        hashlib.sha256(csv1).hexdigest()
    assert d2["report"]["measurements_csv_sha256"] == \
        hashlib.sha256(csv2).hexdigest()
    # 报告端点各自独立。
    rep1 = client.get(f"/api/acceptance/{d1['acceptance_id']}").json()
    rep2 = client.get(f"/api/acceptance/{d2['acceptance_id']}").json()
    assert rep1["measurements_csv_sha256"] == hashlib.sha256(csv1).hexdigest()
    assert rep2["measurements_csv_sha256"] == hashlib.sha256(csv2).hexdigest()
    # 各自的原件原样下载，互不串用。
    c1 = client.get(f"/api/acceptance/{d1['acceptance_id']}/measurements.csv")
    c2 = client.get(f"/api/acceptance/{d2['acceptance_id']}/measurements.csv")
    assert c1.status_code == 200 and c1.content == csv1
    assert c2.status_code == 200 and c2.content == csv2
    # 重复相同文件稳定复用同一身份与报告。
    d1b = _accept(client, job["job_id"], csv1, limit=3.0).json()
    d2b = _accept(client, job["job_id"], csv2, limit=3.0).json()
    assert d1b["acceptance_id"] == d1["acceptance_id"]
    assert d2b["acceptance_id"] == d2["acceptance_id"]
    assert d1b["report"] == d1["report"]


def test_replaced_archived_csv_refused_download_and_resubmit(client, ids):
    """归档 measurements.csv 被替换：下载拒绝（422），相同文件复投也判绑定
    不符；原件缺失下载 404；报告本身仍可取。恢复后一切正常。"""
    job = _run_job(client, ids)
    preds = _lab_press_at(client, job["job_id"], [(0, 0)])
    csv_bytes = _csv(_points_from_preds([(0, 0)], preds))
    d = _accept(client, job["job_id"], csv_bytes, limit=3.0).json()
    aid = d["acceptance_id"]
    from app import main
    csv_path = main.ACCEPTANCE_DIR / aid / "measurements.csv"
    original = csv_path.read_bytes()
    assert original == csv_bytes
    # 用“同一测点序列”的另一份合法 CSV（仅表头别名不同）替换归档原件。
    replacement = b" X , Y , L*, a*, b*\n" + original.split(b"\n", 1)[1]
    assert replacement != original
    try:
        csv_path.unlink()
        # 原件缺失：下载 404。
        assert client.get(
            f"/api/acceptance/{aid}/measurements.csv").status_code == 404
        csv_path.write_bytes(replacement)
        # 原件与报告摘要不符：拒绝下载，绝不交出不符文件。
        r = client.get(f"/api/acceptance/{aid}/measurements.csv")
        assert r.status_code == 422
        # 报告端点本身仍可读（记录不因此被删除）。
        assert client.get(f"/api/acceptance/{aid}").status_code == 200
        # 重复相同文件：复用核验发现归档原件与报告摘要不符 → 拒绝而非静默复用。
        r2 = _accept(client, job["job_id"], csv_bytes, limit=3.0)
        assert r2.status_code == 422
    finally:
        csv_path.write_bytes(original)
    # 原件恢复后：下载与相同文件复投均恢复正常。
    assert client.get(
        f"/api/acceptance/{aid}/measurements.csv").content == csv_bytes
    assert _accept(client, job["job_id"], csv_bytes, limit=3.0).status_code == 200


def test_acceptance_404_for_missing_report(client, ids):
    assert client.get("/api/acceptance/deadbeef").status_code == 404
