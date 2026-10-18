from forecasting import detect_forecast_intent, extract_horizon


def test_forecast_intent():
    assert detect_forecast_intent("Forecast sales for the next 3 months")
    assert not detect_forecast_intent("Show last month's sales")
    assert not detect_forecast_intent("Project Phoenix status update")
    assert detect_forecast_intent("please project revenue")
    assert detect_forecast_intent("project Q4 revenue")


def test_extract_horizon():
    assert extract_horizon("Forecast the next 6 months") == (6, "months")
