import yfinance as yf
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd
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
print(sp500.head())

model = RandomForestClassifier(n_estimators=100, min_samples_split = 100, random_state=42) #random state = 1
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Open", "High", "Low", "Close", "Volume"]
model.fit(train[predictors], train["Target"])
pred = model.predict(test[predictors])
pred = pd.Series(pred, index=test.index)
print(pred)

# combined = pd.concat([test["Target"], pred], axis=1)
# combined.plot()
# plt.show()

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    pred = model.predict(test[predictors])
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