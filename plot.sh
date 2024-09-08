#!/bin/bash

gnuplot -p -e "set terminal dumb size 120, 30; set autoscale; plot '-' using 1:3 with lines notitle"
