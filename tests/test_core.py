import pandas as pd
from zpa.core import compute_simple_returns

def test_compute_simple_returns_happy_path():
    df = pd.DataFrame({"Adj Close": [100, 101, 99, 102]})
    r = compute_simple_returns(df)
    assert [round(x, 4) for x in r.tolist()] == [0.01, -0.0198, 0.0303]

def test_compute_simple_returns_missing_col():
    df = pd.DataFrame({"Close": [1, 2, 3]})
    try:
        compute_simple_returns(df)
        assert False, "should have raised KeyError"
    except KeyError:
        assert True
