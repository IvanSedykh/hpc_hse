#!/bin/bash

python3 ./make_data.py
make clean
make run
python3 ./validate.py
