# based on https://fluxml.ai/tutorials/2021/01/26/mlp.html
using Yota
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDA
using MLDatasets
import Optimisers


include("utils.jl")


if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end


@with_kw mutable struct Args
    η::Float64 = 3e-4       # learning rate
    batchsize::Int = 1024   # batch size
    epochs::Int = 10        # number of epochs
    device::Function = gpu  # set as gpu, if gpu available
end


function getdata(args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset
    xtrain, ytrain = MLDatasets.MNIST(Float32, split=:train)[:]
    xtest, ytest = MLDatasets.MNIST(Float32, split=:test)[:]

    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Batching
    train_data = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_data = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return train_data, test_data
end


function build_model(; imgsize=(28,28,1), nclasses=10)
    return Chain(
 	    Dense(prod(imgsize), 32, relu),
            Dense(32, nclasses))
end


function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end


function train(; kws...)
    # Initializing Model parameters
    args = Args(; kws...)

    # Load Data
    train_data,test_data = getdata(args)

    # Construct model
    m = build_model()
    m = args.device(m)
    train_data = args.device.(train_data)
    test_data = args.device.(test_data)

    loss_fn = (m, x, y) -> logitcrossentropy(m(x), y)
    train_all!(loss_fn, m, train_data, Optimisers.Adam(); epochs=args.epochs)

    @show accuracy(train_data, m)
    @show accuracy(test_data, m)
end


train(epochs=100)