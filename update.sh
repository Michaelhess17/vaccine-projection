#!/bin/bash
cd covid-vaccine-tracker-data
wget -O data/historical_usa_doses_administered.csv https://raw.githubusercontent.com/BloombergGraphics/covid-vaccine-tracker-data/master/data/historical-usa-doses-administered.csv 
cd ..
wget -O data/us_state_vaccinations.csv https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/us_state_vaccinations.csv
python3 -m calc_coefs.py
now=$(date)
echo $now
