#!/usr/bin/bash

maraboupy_dir=$1
if [ ! -f "$maraboupy_dir/Marabou.py" ]; then
    echo "'$maraboupy_dir' does not point to maraboupy directory"
    exit
fi
maraboupy_dir=$(realpath "$maraboupy_dir")

if [[ $# -eq 2 ]] ; then
    maraboupy=$2
else
    maraboupy="maraboupy"
fi

python_libs_path=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

ln -s "$maraboupy_dir" "$python_libs_path/$maraboupy"

# test
python -c "from $maraboupy import MarabouCore"
ret=$?
if [ $ret -ne 0 ]; then
    echo "Failed to import $maraboupy. Maybe you forgot to compile?"
else
    echo "Successfully installed maraboupy to $python_libs_path"
fi