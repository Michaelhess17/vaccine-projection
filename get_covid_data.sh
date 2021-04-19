wget -O data/covid_cases.csv https://raw.githubusercontent.com/jeffcore/covid-19-usa-by-state/master/COVID-19-Cases-USA-By-State.csv
wget -O data/covid_deaths.csv https://raw.githubusercontent.com/jeffcore/covid-19-usa-by-state/master/COVID-19-Deaths-USA-By-State.csv 
python3 clean_covid_data.py
rm data/covid_cases.csv
rm data/covid_deaths.csv

