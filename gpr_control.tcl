# Define total simulation steps
set totalSteps 35000000

# Define GPR-related step ranges
set gprStartStep [expr int($totalSteps * 0.0002)]
set gprFinalPhaseStep [expr int($totalSteps * 0.7)]

# Procedure to update bias (including error handling and logging)
proc updateBias {} {
    global step

    # Call the Python script and capture its output
    set output [exec /bin/bash -c "source /home/u28/krishnaghanta/anaconda3/bin/activate && python apply_gpr_gpu.py"]

    puts ">>> Successfully updated bias in apply_gpr_gpu.py at step $step"
    puts "Python script output: $output"

    # Log the last updated bias value from update_bias.dat
#    set lastLine [exec tail -n 1 update_bias.dat]
#    set fields [split $lastLine]
#    set lastValue [lindex $fields end]
#    puts "Updated bias value at step $step: $lastValue"
}

# Procedure to calculate GPR at specific steps
proc calcGPR {step} {
    global gprStartStep gprFinalPhaseStep

    # Check if we are in the initial phase (before gprFinalPhaseStep)
    if { $step >= $gprStartStep && $step < $gprFinalPhaseStep && $step % 50000 == 0 } {
        puts ">>> Executing GPR at step $step (Initial Phase)"
        updateBias
    }

    # Check if we are in the final phase (after gprFinalPhaseStep)
    if { $step >= $gprFinalPhaseStep && $step % 100000 == 0 } {
        puts ">>> Executing GPR at step $step (Final Phase)"
        updateBias
    }
}

# Procedure to calculate forces and apply GPR bias
proc calcforces {} {
    global step
    set step [getstep]

    # Call the GPR update check
    calcGPR $step

    # Read the last bias value from update_bias.dat
    set infile [open "update_bias.dat" r]
    set lines [split [read $infile] "\n"]
    close $infile
    set lastline [lindex $lines end-2] ;# last non-empty line
    set fields [split $lastline]
    set bias_value [lindex $fields end]

  # Print the bias only every 1000 steps (for example)
    if {$step % 50000 == 0} {
        puts ">>> Applying updated bias: $bias_value"
    }

    # Apply the bias using addenergy
    addenergy $bias_value
}
