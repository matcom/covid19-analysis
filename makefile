.PHONY: timeseries
timeseries:
	curl https://pomber.github.io/covid19/timeseries.json > data/timeseries.json
	curl https://raw.githubusercontent.com/covid19cubadata/covid19cubadata.github.io/master/data/covid19-cuba.json > data/covid19-cuba.json
	curl https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv > data/testing.csv
	