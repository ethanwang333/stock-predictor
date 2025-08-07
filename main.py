import yfinance as yf
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period = "max")
#print(sp500.head())
# sp500.plot.line(y = "Close", use_index=True)
# plt.show()
del sp500["Dividends"] # not considered in the analysis
del sp500["Stock Splits"]

sp500["Tommorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tommorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()
# print(sp500.head())

# model = RandomForestClassifier(n_estimators=100, min_samples_split = 100, random_state=42) #random state = 1
# train = sp500.iloc[:-100]
# test = sp500.iloc[-100:]

# predictors = ["Open", "High", "Low", "Close", "Volume"]
# model.fit(train[predictors], train["Target"])
# pred = model.predict(test[predictors])
# pred = pd.Series(pred, index=test.index)
# print(pred)

# combined = pd.concat([test["Target"], pred], axis=1)
# combined.plot()
# plt.show()

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    pred = model.predict_proba(test[predictors])[:, 1]
    pred[pred >= 0.6] = 1 #.6 is the threshold for a positive prediction
    pred[pred < 0.6] = 0
    pred = pd.Series(pred, index=test.index, name = "Predicted")
    combined = pd.concat([test["Target"], pred], axis=1)
    return combined
    
def backtest(data, model, predictors, start = 2500, step = 250):
    all_preds = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predicted = predict(train, test, predictors, model)
        all_preds.append(predicted)
    return pd.concat(all_preds)

def compute_rsi(close_series, window=14, use_ewm=True):
    delta = close_series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    if use_ewm:
        avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    else:
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
    rs  = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_momentum(close_series, window=10):
    return close_series - close_series.shift(window)

def compute_sma(series, window):
    return series.rolling(window=window).mean()

def compute_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def compute_obv(close, volume):
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def compute_bollinger_bands(close, window=20, num_std=2):
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return upper_band, lower_band

def compute_macd(close, short=12, long=26, signal=9):
    ema_short = compute_ema(close, short)
    ema_long = compute_ema(close, long)
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def add_time_features(df):
    df = df.copy()
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["DayOfMonth"] = df.index.day
    df["Year"] = df.index.year

    # Count consecutive up and down days
    df["Return"] = df["Close"].pct_change()
    df["Up"] = df["Return"] > 0

    df["UpStreak"] = 0
    df["DownStreak"] = 0
    up_streak = down_streak = 0

    for i in range(1, len(df)):
        if df["Up"].iloc[i]:
            up_streak += 1
            down_streak = 0
        elif not df["Up"].iloc[i] and not pd.isna(df["Return"].iloc[i]):
            down_streak += 1
            up_streak = 0
        else:
            up_streak = down_streak = 0
                                       
        df.at[df.index[i], "UpStreak"] = up_streak
        df.at[df.index[i], "DownStreak"] = down_streak

    df.drop(["Return", "Up"], axis=1, inplace=True)
    return df
def add_features(sp500):
    sp500 = sp500.copy()
    sp500 = add_time_features(sp500)
    
    sp500["Momentum"] = compute_momentum(sp500["Close"], window=10)
    sp500["RSI_14"] = compute_rsi(sp500["Close"], window=14)
    sp500["SMA_50"] = compute_sma(sp500["Close"], window=50)
    sp500["SMA_200"] = compute_sma(sp500["Close"], window=200)
    sp500["EMA_12"] = compute_ema(sp500["Close"], window=12)
    sp500["EMA_26"] = compute_ema(sp500["Close"], window=26)
    sp500["EMA_Diff"] = sp500["EMA_12"] - sp500["EMA_26"]
    sp500["OBV"] = compute_obv(sp500["Close"], sp500["Volume"])
    sp500["Upper_Band"], sp500["Lower_Band"] = compute_bollinger_bands(sp500["Close"], window=20, num_std=2)
    sp500["MACD_Line"], sp500["Signal_Line"], sp500["MACD_Histogram"] = compute_macd(sp500["Close"], short=12, long=26, signal=9)

    horizons = [2, 5, 60, 250]
    new_predictors = ["RSI_14", "Momentum", "SMA_50", "SMA_200", "EMA_Diff", "OBV", "Upper_Band", "Lower_Band", "MACD_Line", "Signal_Line", "MACD_Histogram",
                      "DayOfWeek", "Month", "DayOfMonth", "Year", "UpStreak", "DownStreak"]
    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
        
        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
        
        new_predictors += [ratio_column, trend_column]
    
    sp500 = sp500.dropna()
    return sp500, new_predictors


def trainTest(sp500):
    sp500, new_predictors = add_features(sp500)
    
    trainDate = "2020-01-01" # little more than 80% of the data
    train = sp500.loc[:trainDate].copy()
    test = sp500.loc[trainDate:].copy()
    
    X_train, X_test, y_train, y_test = train[new_predictors], test[new_predictors], train["Target"], test["Target"]
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    precision = precision_score(y_test, preds)
    print("Precision Score:", precision)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='precision')
    importances = perm_importance.importances_mean
    stds = perm_importance.importances_std
    features = new_predictors
    indices = importances.argsort()[::-1]

    print("\nPermutation Feature Importance:")
    for i in range(len(features)):
        print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f} +/- {stds[indices[i]]:.4f}")

    plt.figure(figsize=(12,6))
    plt.title("Permutation Feature Importance (Precision)")
    plt.bar(range(len(features)), importances[indices], yerr=stds[indices], align="center")
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()
    return model, sp500, new_predictors

model, sp500, predictors = trainTest(sp500)
predictions = backtest(sp500, model, predictors)
print("Backtest Precision Score:", precision_score(predictions["Target"], predictions["Predicted"]))
