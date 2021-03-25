#!/bin/bash
cd covid-vaccine-tracker-data
git pull origin master
cd ..
wget -O data/us_state_vaccinations.csv https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/us_state_vaccinations.csv
now=$(date)
echo $now
