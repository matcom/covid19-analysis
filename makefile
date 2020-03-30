.PHONY: timeseries
timeseries:
	curl https://pomber.github.io/covid19/timeseries.json > data/timeseries.json