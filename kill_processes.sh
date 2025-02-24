#!/bin/bash

# Define the pattern to match your parallel loop processes
# pattern="python -u parallel_recovery.py"
pattern="python expt_optimisation.py"

# Kill the processes matching the pattern
pkill -f "$pattern"

# Verify that no processes are running
if pgrep -f "$pattern" > /dev/null; then
    echo "Some processes are still running."
else
    echo "All processes have been terminated."
fi
