from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def fresh():
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_runs_without_exception():
    assert not fresh().exception


def test_module1_inputs():
    at = fresh()
    at.slider(key="sim_hour").set_value(3).run()
    assert not at.exception
    at.selectbox(key="sim_depot").select("Medanta Gurugram").run()
    assert not at.exception


def test_module3_inputs_and_methods():
    at = fresh()
    at.slider(key="hub_k").set_value(5).run()
    assert not at.exception
    at.radio(key="hub_method").set_value("k-means (heuristic)").run()
    assert not at.exception
    at.slider(key="hub_thr").set_value(4).run()
    assert not at.exception


def test_module4_extremes():
    at = fresh()
    at.slider(key="bat_wind").set_value(40).run()
    at.slider(key="bat_angle").set_value(90).run()
    assert not at.exception
    at.slider(key="bat_payload").set_value(3.5).run()
    at.slider(key="bat_soc").set_value(0.3).run()
    assert not at.exception


def test_module5_edge_cases():
    at = fresh()
    at.slider(key="dsp_drones").set_value(1).run()
    at.slider(key="dsp_orders").set_value(120).run()
    assert not at.exception
    at.multiselect(key="dsp_hubs").set_value([]).run()
    assert not at.exception
    at.multiselect(key="dsp_hubs").set_value(["AIIMS New Delhi"]).run()
    at.slider(key="dsp_wind").set_value(40).run()
    assert not at.exception
