logistic(x) = 1 / (1 + exp(-x))
antilogistic(x) = 1 / (1 + log(-x))

function exp_to_log_tform(tape::Tape)
    new_tape = similar(tape)
    changed = false
    for op in tape
        if op isa Call && op.fn == exp
            changed = true
            push!(new_tape, copy_with(op, fn=log))
        else
            push!(new_tape, op)
        end
    end
    return new_tape, changed
end

@testset "transform" begin
    @test transform(t -> (t, false), logistic, 4.0) == logistic
    f1 = transform(t -> (t, true), logistic, 4.0)
    # not the same, but recompiled function
    @test f1 != logistic
    # but should still do the same thing
    @test f1(20.0) == logistic(20.0)
    # actual transformation
    f2 = transform(exp_to_log_tform, logistic, -4.0)
    @test f2(-1.0) == antilogistic(-1.0)
end
