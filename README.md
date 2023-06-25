# Cross Exchange Crypto Arbitrage

`import crossexchangearb as cxa`

The main purpose of this project is to demonstrate how much money could be made from cross-exchange cryptocurrency trades under the following (very large) assumptions:

- Capital controls do not restrict trades.
- 0 transaction fees.
- Trades occur instantaneously.
- No limit to the amount of capital that we have.

Go to `./library/crossexchangearb.py` for the Python library that I created. Funtionality includes refactoring 4 years of minute data from 4 different exchanges into a multindex dataframe and computing cross exchange arbitrage for all assets in a multindex dataframe. In this project, I arbitraged spots vs. spots and futures vs. futures because I had no access to the date that the futures expired, so arbitraging spots vs. futures didn't make sense. Note that it is obivously not feasible to make most of these trades due to capital controls

Go to the `./notebooks/` directory to see how to use the library functions.

Go to `./results/` to see total profits for coins across spots markets and futures markets.