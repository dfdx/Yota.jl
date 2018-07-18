

grad!(dy::TAny, ::Val{1}, op::Call{typeof(*), Tuple{TReal, TReal}}) = dy * op.args[2]
grad!(dy::TAny, ::Val{2}, op::Call{typeof(*), Tuple{TReal, TReal}}) = dy * op.args[1]


function grad!(dy::TAny, ::Val{1}, op::Call{typeof(+), Tuple{TReal, TReal}})
    x, y = op.args
    return record!(dy.tape, Assign, dy)
end

function grad!(dy::TAny, ::Val{2}, op::Call{typeof(+), Tuple{TReal, TReal}})
    x, y = op.args
    return record!(dy.tape, Assign, dy)
end


function grad!(dy::TAny, ::Val{1}, op::Call{typeof(-), Tuple{TReal, TReal}})
    x, y = op.args
    return record!(dy.tape, Assign, dy)
end

function grad!(dy::TAny, ::Val{2}, op::Call{typeof(-), Tuple{TReal, TReal}})
    x, y = op.args
    return record!(dy.tape, Call, -, (dy,))
end

grad!(dy::TAny, ::Val{1}, op::Call{typeof(sin), Tuple{TReal}}) = cos(op.args[1]) * dy
grad!(dy::TAny, ::Val{1}, op::Call{typeof(cos), Tuple{TReal}}) = -sin(op.args[1]) * dy
grad!(dy::TAny, ::Val{1}, op::Call{typeof(log), Tuple{TReal}}) = dy / op.args[1]
grad!(dy::TAny, ::Val{1}, op::Call{typeof(exp), Tuple{TReal}}) = exp(op.args[1]) * dy
grad!(dy::TAny, ::Val{1}, op::Call{typeof(abs), Tuple{TReal}}) = sign(op.args[1]) * dy
grad!(dy::TAny, ::Val{1}, op::Call{typeof(abs2), Tuple{TReal}}) = 2 * op.args[1] * dy
grad!(dy::TAny, ::Val{1}, op::Call{typeof(sign), Tuple{TReal}}) = 0
grad!(dy::TAny, ::Val{1}, op::Call{typeof(tanh), Tuple{TReal}}) =
    (x = op.args[1]; (1.0 - tanh(x)  * tanh(x))  * dy)
