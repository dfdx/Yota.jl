function train_epoch!(loss_fn, model, data, state)
    losses = []
    for xs in data
        loss, g = grad(loss_fn, model, xs...)
        state, model = Optimisers.update!(state, model, g[2])
        push!(losses, loss)
    end
    return losses
end


function train_all!(loss_fn, model, data, opt; epochs=10)
    state = Optimisers.setup(opt, model)
    for epoch=1:epochs
        losses = train_epoch!(loss_fn, model, data, state)
        @info("epoch $epoch loss = $(mean(losses))")
    end
end