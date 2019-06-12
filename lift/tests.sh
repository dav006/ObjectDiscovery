#!/bin/bash


UMBRALES=(10830 3209 1353 693 401 252 169 119 86)
# UMBRALES=(1733 433 192 108 69 48 35 27 21 17)

for i in "${UMBRALES[@]}"
do
	python createModel.py $i 
	python rankingImages.py 
	python evaluate.py
done
