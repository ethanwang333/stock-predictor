import yfinance as yf
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
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