using Printf
using Statistics

using OneHotArrays
using Flux
using Flux.Losses
using CairoMakie

using StatsBase: Weights, sample

names = readlines(joinpath(@__DIR__, "../names.txt"))

all_chars = vcat('.', 'a':'z')
nc = length(all_chars)
bigrams = zeros(Int, nc, nc)

char2int = Dict(c => i for (i, c) in enumerate(all_chars))

# Building the dataset

block_size = 3

function generate_dataset(names)
    xs = []
    ys = []

    for name in names
        context = ones(Int, block_size)

        _name = name * '.'
        for c in _name
            i = char2int[c]

            if isempty(xs)
                xs = context
            else
                xs = hcat(xs, context)
            end

            push!(ys, i)
            context = vcat(context[2:end], i)
        end
    end

    xs = permutedims(xs, (2, 1))

    return xs, ys
end

@info "Generating dataset..."
xs, ys = generate_dataset(names)

ys = Int.(ys)

embedding_dims = 2
C = randn(nc, embedding_dims)

W₁ = randn(block_size*embedding_dims, 100)
b₁ = reshape(randn(100), (1, 100))

W₂ = randn(100, nc)
b₂ = reshape(randn(nc), (1, nc))

function loss(xs, ys, C, W₁, b₁, W₂, b₂)
    ndata = size(xs, 1)
    nembs = block_size * embedding_dims
    embedding = reshape(permutedims(C[xs, :], (1, 3, 2)), (ndata, nembs))

    h = tanh.(embedding * W₁ .+ b₁)
    logits = h * W₂ .+ b₂

    return logitcrossentropy(transpose(logits), onehotbatch(ys, 1:nc))
end

@info "Training..."

# Proportions: 80% training set, 10% dev set, 10% test set
p_train = 0.8
p_dev = 0.1
p_test = 0.1

minibatch_size = 256

# Number of training examples = 80% of the data, rounded to nearest multiple of `minibatch_size`.
# Training set is indices 1:n_train
# Dev set is indices n_train:n_dev
# Test set is indices n_test:end
n_train = minibatch_size * round(Int, length(ys) / minibatch_size * p_train)
n_dev = minibatch_size * round(Int, length(ys) / minibatch_size * (p_train + p_dev))

xs_train = xs[1:n_train, :]
ys_train = ys[1:n_train]

xs_dev = xs[n_train:n_dev, :]
ys_dev = ys[n_train:n_dev]

xs_test = xs[n_dev:end, :]
ys_test = ys[n_dev:end]

ηs = [0.1, 0.01] # Learning rate / step size with a decay

epochs = 10000 # Epochs per learning rate

loss_history = []

for η in ηs, epoch in 1:epochs

    is_minibatch = rand(1:n_train, minibatch_size)
    xs_minibatch = xs_train[is_minibatch, :]
    ys_minibatch = ys_train[is_minibatch]

    loss_value, grads = Flux.withgradient(loss, xs_minibatch, ys_minibatch, C, W₁, b₁, W₂, b₂)

    # Gradient descent
    ∇C, ∇W₁, ∇b₁, ∇W₂, ∇b₂ = grads[3:7]

    C .-= η .* ∇C
    W₁ .-= η .* ∇W₁
    W₂ .-= η .* ∇W₂
    b₁ .-= η .* ∇b₁
    b₂ .-= η .* ∇b₂

    push!(loss_history, loss_value)
    @printf("epoch %s: loss = %f, |∇C| = %.8e\n", epoch, loss_value, mean(abs, ∇C))
end

loss_history = float.(loss_history)

loss_train = loss(xs_train, ys_train, C, W₁, b₁, W₂, b₂)
loss_dev = loss(xs_dev, ys_dev, C, W₁, b₁, W₂, b₂)
loss_test = loss(xs_test, ys_test, C, W₁, b₁, W₂, b₂)

println("Training loss: $loss_train")
println("     Dev loss: $loss_dev")
println("    Test loss: $loss_test")

let
    total_epochs = length(loss_history)
    fig = Figure(resolution = (900, 600), padding=100)
    ax = Axis(fig[1, 1],
        xlabel = "Epochs",
        ylabel = "Cross-entropy loss",
        xgridvisible = false,
        ygridvisible = false,
        xscale = log10
    )

    lines!(1:total_epochs, loss_history, linewidth=2)
    xlims!(ax, (1, total_epochs))
    save("mlp_loss.png", fig, px_per_unit=2)
end

let
    fig = Figure(resolution = (750, 750))
    ax = Axis(fig[1, 1],
        xlabel  = "Embedding dim 1",
        ylabel = "Embedding dim 2",
        xgridvisible = false,
        ygridvisible = false
    )

    for (i, char) in enumerate(all_chars)
        x, y = C[i, :]
        scatter!(ax, x, y, color=:skyblue, markersize=40)
        text!(ax, string(char), position=(x, y), align=(:center, :center))
    end

    save("embedding_space.png", fig, px_per_unit=2)
end

function sample_name_mlp(C, W₁, b₁, W₂, b₂)
    context = ones(Int, block_size)
    name = ""

    while true
        embedding = reshape(permutedims(C[context, :], (2, 1)), (1, 6))
        h = tanh.(embedding * W₁ .+ b₁)
        logits = h * W₂ .+ b₂

        probs = softmax(logits, dims=2)[:]

        next_char_index = sample(1:nc, Weights(probs))
        next_char = all_chars[next_char_index]
        name *= next_char
        context = vcat(context[2:end], next_char_index)

        if name[end] == '.'
            return name[1:end-1]
        end
    end
end

println("Sampling names from the multi-layer perceptron:")
for _ in 1:25
    println(sample_name_mlp(C, W₁, b₁, W₂, b₂))
end
