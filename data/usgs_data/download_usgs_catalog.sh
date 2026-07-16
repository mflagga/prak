#!/bin/bash
# Pobranie globalnego katalogu trzęsień ziemi USGS M>=4.0, 2005-01-01 -- 2025-02-01.
# Zapytania miesięczne, bo USGS FDSN API ogranicza liczbę zdarzeń zwracanych
# w jednym zapytaniu (~20 000), a cały zakres to ~290 000 zdarzeń.
# Uruchomić z katalogu głównego repo: bash usgs_data/download_usgs_catalog.sh
set -euo pipefail

mkdir -p usgs_data
cd usgs_data

start_year=2005
end_year=2025
out="usgs_m4_2005_2025.csv"
> "$out"
first=1

for year in $(seq $start_year $end_year); do
  for month in $(seq -w 1 12); do
    if [ "$year" -eq "$end_year" ] && [ "$month" -gt "01" ]; then
      break
    fi
    y2=$year
    m2=$((10#$month + 1))
    if [ $m2 -gt 12 ]; then
      m2=1
      y2=$((year+1))
    fi
    m2=$(printf "%02d" $m2)
    start="${year}-${month}-01"
    end="${y2}-${m2}-01"
    tmp=$(mktemp)
    curl -s "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=${start}&endtime=${end}&minmagnitude=4.0" -o "$tmp"
    if [ $first -eq 1 ]; then
      cat "$tmp" >> "$out"
      first=0
    else
      tail -n +2 "$tmp" >> "$out"
    fi
    rm -f "$tmp"
  done
done
