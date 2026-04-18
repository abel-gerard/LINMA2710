#!/bin/sh

scp 'abgerard@manneback:/home/ucl/inma/abgerard/LINMA2710/project/performance/*.csv' performance/
uv run plot_perf.py