#!/usr/bin/bash

maraboupy=$1

python_libs_path=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

if ! rm "$python_libs_path/$maraboupy" 2> /dev/null; then
    echo "$maraboupy is not installed"
    exit 0;
fi

# test
python -c "from $maraboupy import MarabouCore" 2> /dev/null
ret=$?
if [ $ret -ne 0 ]; then
    echo "Successfully uninstalled maraboupy to $python_libs_path"
else
    echo "Failed to uninstall $maraboupy from $python_libs_path"
fi