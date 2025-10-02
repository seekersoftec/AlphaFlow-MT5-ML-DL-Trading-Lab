# backtests/vectorbt_backtest.py

import numpy as np
import pandas as pd
import vectorbt as vbt

def run_vectorbt_backtest(
    model,
    X,
    selected_features,
    data,
    scaler,
    init_cash=10000,
    freq='4H',
    threshold=0.0
):
    """
    Simple, short-enabled backtest using target exposure (-1, 0, +1).
    """
    # features -> scale -> predict
    X_sel = X[selected_features]
    preds = model.predict(scaler.transform(X_sel))

    # map preds -> {-1, 0, 1}
    if threshold > 0.0:
        exposure = np.where(preds > threshold, 1.0,
                   np.where(preds < -threshold, -1.0, 0.0))
    else:
        exposure = np.sign(preds).astype(float)

    # align to prices
    close = data.loc[X_sel.index, "close"]
    target = pd.Series(exposure, index=close.index)

    # build portfolio: -1 short, 0 flat, +1 long
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=target,
        size_type='targetpercent',
        init_cash=init_cash,
        freq=freq
    )
    return pf
