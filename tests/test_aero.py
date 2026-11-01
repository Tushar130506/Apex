import numpy as np, pandas as pd, pytest
from apex.aero.cfd_results import validate_cfd
from fixtures import toy_cfd
from apex.aero.doe import latin_hypercube, run_sheet
from apex.aero.optimize import propose
from apex.aero.vehicle_bridge import forces, lap_time, lap_time_delta, Vehicle
from apex.models.aero_surrogate import AeroSurrogate
from apex.schemas.contracts import ContractError


@pytest.fixture(scope="module")
def doe():
    return toy_cfd(latin_hypercube(30, seed=3))


def test_non_ansys_data_refused_by_default(doe):
    with pytest.raises(ContractError):
        validate_cfd(doe)
    validate_cfd(doe, allowed_provenance=("test_fixture",))


def test_unconverged_rows_rejected(doe):
    bad = doe.copy(); bad.loc[0, "residual_max"] = 0.1
    with pytest.raises(ContractError):
        validate_cfd(bad, allowed_provenance=("test_fixture",))


def test_run_sheet_columns():
    rs = run_sheet(latin_hypercube(5))
    assert rs.provenance.eq("ansys_fluent_pending").all() and len(rs) == 5


def test_surrogate_holdout_and_extrapolation_flag(doe):
    tr, te = doe.iloc[:24], doe.iloc[24:]
    s = AeroSurrogate().fit(tr)
    p = s.predict(te)
    assert np.abs(p.mean[:, 0] - te.cd).mean() < 0.02 and np.abs(p.mean[:, 1] - te.cl).mean() < 0.05
    far = pd.DataFrame({"aoa_deg": [30.0], "ride_height_mm": [200.0], "yaw_deg": [15.0]})
    assert s.predict(far).extrapolating.all()


def test_forces_scale_with_v_squared():
    d1, l1 = forces(30, 0.5, 1.4, Vehicle()); d2, l2 = forces(60, 0.5, 1.4, Vehicle())
    assert d2 / d1 == pytest.approx(4.0) and l2 / l1 == pytest.approx(4.0)


def test_bridge_directionality():
    base = lap_time(0.5, 1.2)
    assert lap_time(0.5, 2.0) < base               # more downforce at same drag -> faster on a twisty lap
    assert lap_time(0.8, 1.2) > base               # more drag at same downforce -> slower
    assert lap_time_delta((0.5, 1.2), (0.5, 1.2)) == pytest.approx(0.0)
    assert 40 < base < 140


def test_propose_returns_flagged_candidates(doe):
    s = AeroSurrogate().fit(doe)
    out = propose(s, (0.52, 1.35), n_search=300, top_k=3)
    assert len(out) <= 3 and (out.status == "SURROGATE_ESTIMATE_NEEDS_CFD_RERUN").all() and not out.extrapolating.any()
