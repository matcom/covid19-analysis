.PHONY: timeseries
timeseries:
	curl https://pomber.github.io/covid19/timeseries.json > data/timeseries.json
	curl https://raw.githubusercontent.com/covid19cubadata/covid19cubadata.github.io/master/data/covid19-cuba.json > data/covid19-cuba.json
	curl https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv > data/testing.csv
	
.PHONY: build
build:
	docker-compose -f build/docker-compose.yml build

.PHONY: code
code:
	docker-compose -f build/docker-compose.yml up -d code
	google-chrome --app=http://localhost:8443

.PHONY: app
app:
	docker-compose -f build/docker-compose.yml up app

.PHONY: extensions-list
extensions-list:
	code-server --list-extensions > build/extensions.txt

.PHONY: extensions-install
extensions-install:
	bash build/install_extensions.sh
