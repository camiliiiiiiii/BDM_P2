#! /bin/bash
BASEDIR=./

# running mongo connection script
echo "Creating mongo Connection"
python Mongo_connection.py

# running formatted script
echo "Cleaning and taking the data from  Persistent Zone to Formatted Zone"
python Formatted.py

# running KPI exploit and predictive model script
echo "Computing KPIs and Training Predictive Model"
python KPI_model.py

# running stream script
echo "Computing Stream results"
python Stream.py