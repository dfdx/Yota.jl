# based on https://fluxml.ai/tutorials/2021/02/07/convnet.html
# using Yota
include("../../src/core.jl")
using Flux, MLDatasets, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy, flatten
using Optimisers
using MLDatasets: MNIST
using Base.Iterators: partition
using Printf, BSON
using Parameters: @with_kw
using CUDA
CUDA.allowscalar(false)

include("utils.jl")

@with_kw mutable struct Args
    lr::Float64 = 3e-3
    epochs::Int = 20
    batch_size = 128
    savepath::String = "./"
 end


 function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X)[1:end-1]..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[:,:,idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
 end


 function get_processed_data(args)
    # Load labels and images
    train_imgs, train_labels = MNIST(split=:train)[:]
    mb_idxs = partition(1:length(train_labels), args.batch_size)
    train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]

    # Prepare test set as one giant minibatch:
    test_imgs, test_labels = MNIST(split=:test)[:]
    test_set = make_minibatch(test_imgs, test_labels, 1:length(test_labels))

    return train_set, test_set

 end


 function build_model(args; imgsize = (28,28,1), nclasses = 10)
    cnn_output_size = Int.(floor.([imgsize[1]/8,imgsize[2]/8,32]))
    return Chain(
        # First convolution, operating upon a 28x28 image
        Conv((3, 3), imgsize[3]=>16, pad=(1,1), relu),
        MaxPool((2,2)),

        # Second convolution, operating upon a 14x14 image
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        MaxPool((2,2)),

        # Third convolution, operating upon a 7x7 image
        Conv((3, 3), 32=>32, pad=(1,1), relu),
        MaxPool((2,2)),

        # Reshape 3d array into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
        flatten,
        Dense(prod(cnn_output_size), 10))
 end



augment(x) = x .+ gpu(0.1f0*randn(eltype(x), size(x)))
anynan(x) = any(y -> any(isnan, y), x)
accuracy(x, y, model) = mean(onecold(cpu(model(x))) .== onecold(cpu(y)))
collect_params(model) = Optimisers.destructure(model)[1]


function train(; kws...)
    args = Args(; kws...)

    @info("Loading data set")
    train_set, test_set = get_processed_data(args)

    # Define our model.  We will use a simple convolutional architecture with
    # three iterations of Conv -> ReLU -> MaxPool, followed by a final Dense layer.
    @info("Building model...")
    model = build_model(args)

    # Load model and datasets onto GPU, if enabled
    train_set = gpu.(train_set)
    test_set = gpu.(test_set)
    model = gpu(model)

    # Make sure our model is nicely precompiled before starting our training loop
    model(train_set[1][1])

    # `loss_fn()` calculates the crossentropy loss between our prediction `y_hat`
    # (calculated from `model(x)`) and the ground truth `y`.  We augment the data
    # a bit, adding gaussian random noise to our image to make it more robust.
    function loss_fn(model, x, y)
        x̂ = augment(x)
        ŷ = model(x̂)
        return logitcrossentropy(ŷ, y)
    end

    # Train our model with the given training set using the ADAM optimizer and
    # printing out performance against the test set as we go.
    opt = Optimisers.Adam(args.lr)

    @info("Beginning training loop...")
    best_acc = 0.0
    last_improvement = 0
    state = Optimisers.setup(opt, model)
    for epoch_idx in 1:args.epochs
        # Train for a single epoch
        train_epoch!(loss_fn, model, train_set, state)

        # Terminate on NaN
        if anynan(collect_params(model))
            @error "NaN params"
            break
        end

        # Calculate accuracy:
        acc = accuracy(test_set..., model)

        @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
        # If our accuracy is good enough, quit out.
        if acc >= 0.999
            @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc >= best_acc
            @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
            BSON.@save joinpath(args.savepath, "mnist_conv.bson") params=cpu.(collect_params(model)) epoch_idx acc
            best_acc = acc
            last_improvement = epoch_idx
        end

        # If we haven't seen improvement in 5 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
            # opt.eta /= 10.0
            opt = Optimisers.Adam(opt.eta / 10)
            @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end

        if epoch_idx - last_improvement >= 10
            @warn(" -> We're calling this converged.")
            break
        end
    end
 end


function test(; kws...)
    args = Args(; kws...)

    # Loading the test data
    _, test_set = get_processed_data(args)

    # Re-constructing the model with random initial weights
    model = build_model(args)
    _, re = destructure(model)

    # Loading the saved parameters
    BSON.@load joinpath(args.savepath, "mnist_conv.bson") params

    # Loading parameters onto the model
    model = re(params)

    test_set = gpu.(test_set)
    model = gpu(model)
    @show accuracy(test_set..., model)
end


