# Define total simulation steps
set totalSteps 50000000

# Define GPR-related step ranges
set gprStartStep [expr int($totalSteps * 0.0002)]
set gprFinalPhaseStep [expr int($totalSteps * 0.7)]

proc updateBias {} {
    exec /bin/bash -c "source /home/u28/krishnaghanta/anaconda3/bin/activate && python apply_gpr_gpu.py"
    puts ">>> GPR updated bias at step [getstep]"
}

proc calcGPR {step} {
    global gprStartStep gprFinalPhaseStep

    if { $step >= $gprStartStep && $step < $gprFinalPhaseStep && $step % 50000 == 0 } {
        puts ">>> Executing GPR at step $step (Initial Phase)"
        updateBias
    }

    if { $step >= $gprFinalPhaseStep && $step % 100000 == 0 } {
        puts ">>> Executing GPR at step $step (Final Phase)"
        updateBias
    }
}

proc calcforces {} {
    global step
    set step [getstep]

    # Call GPR function
    calcGPR $step
}
