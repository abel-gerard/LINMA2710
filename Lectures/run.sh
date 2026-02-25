#!/bin/sh

julia -e 'import Pkg; Pkg.instantiate(); import Pluto; Pluto.run()'
