
function grad!(dy::TAny, ::Val{1}, op::Call{typeof(*), Tuple{TArray, TArray}})
    x, y = op.args
    yt = record!(dy.tape, Call, transpose, y)
    return record!(dy.tape, Call, *, (dy, yt))
end

function grad!(dy::TAny, ::Val{2}, op::Call{typeof(*), Tuple{TArray, TArray}})
    x, y = op.args
    xt = record!(dy.tape, Call, transpose, x)
    return record!(dy.tape, Call, *, (xt, dy))
end


function grad!(dy::TAny, ::Val{1}, op::Call{typeof(sum), Tuple{TArray{T,N}}}) where {T,N}
    return record!(dy.tape, Call, sum_grad, (op.args[1], dy); kwargs=op.kwargs)
end
function grad!(dy::TAny, ::Val{1}, op::Call{typeof(mean), Tuple{TArray{T,N}}}) where {T,N}
    return record!(dy.tape, Call, sum_grad, (op.args[1], dy); kwargs=op.kwargs)
end


grad!(dy::TAny, ::Val{1}, op::Call{typeof(transpose), Tuple{TArray{T,N}}}) where {T,N} =
    transpose(dy)
# TODO: .==
# TODO: minimum, maximum



# TODO: make single function to handle all broadcasting of arrays of the same size
function grad!(dy::TAny, ::Val{1}, op::Bcast{typeof(+), Tuple{TArray{T,N}, TArray{T,N}}}) where {T,N}
    return record!(dy.tape, Assign, dy)
end

function grad!(dy::TAny, ::Val{2}, op::Bcast{typeof(+), Tuple{TArray{T,N}, TArray{T,N}}}) where {T,N}
    return record!(dy.tape, Assign, dy)
end


