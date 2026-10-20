"""From headline sentiment to a tested daily long/short equity signal.

prices.py     download + cache daily adjusted OHLC (yfinance), wide price panels
signal.py     score aggregation and look-ahead-safe alignment to trade dates
portfolio.py  forward returns, dollar-neutral long/short book, costs, IC, permutation test
"""
