function train_epoch!(loss_fn, model, data, opt)
    state = Optimisers.setup(opt, model)
    losses = []
    for xs in data
        loss, g = grad(loss_fn, model, xs...)
        state, model = Optimisers.update!(state, model, g[2])
        push!(losses, loss)
    end
    return losses
end


function train_all!(loss_fn, model, data, opt; epochs=10)
    for epoch=1:epochs
        losses = train_epoch!(loss_fn, model, data, opt)
        @info("epoch $epoch loss = $(mean(losses))")
    end
end