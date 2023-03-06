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

# Set up the model

dim_embedding = 10
n_hidden = 200

C = randn(nc, dim_embedding)

# Kaiming initialization for tanh activation with `block_size * dim_embedding` inputs
ω = (5/3) / sqrt(block_size*dim_embedding)
W₁ =    ω  .* randn(block_size*dim_embedding, n_hidden)
b₁ = 0.01 .* reshape(randn(n_hidden), (1, n_hidden))

W₂ = 0.01 .* randn(n_hidden, nc)
b₂ = 0    .* reshape(randn(nc), (1, nc))

# Define loss function

function loss(xs, ys, C, W₁, b₁, W₂, b₂)
    ndata = size(xs, 1)
    nembs = block_size * dim_embedding
    embedding = reshape(permutedims(C[xs, :], (1, 3, 2)), (ndata, nembs))

    h = tanh.(embedding * W₁ .+ b₁)
    logits = h * W₂ .+ b₂

    return logitcrossentropy(transpose(logits), onehotbatch(ys, 1:nc))
end

# Partition data into training, dev, and test sets

p_train = 0.8  # 80% training set
p_dev = 0.1    # 10% dev set
p_test = 0.1   # 10% test set

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

# Train!

@info "Training..."

iters = 20000

# Learning rate / step size with a decay
η(i) = i < 10000 ? 0.1 : 0.01

loss_history = []

for i in 1:iters

    is_minibatch = rand(1:n_train, minibatch_size)
    xs_minibatch = xs_train[is_minibatch, :]
    ys_minibatch = ys_train[is_minibatch]

    loss_value, grads = Flux.withgradient(loss, xs_minibatch, ys_minibatch, C, W₁, b₁, W₂, b₂)

    # Gradient descent
    ∇C, ∇W₁, ∇b₁, ∇W₂, ∇b₂ = grads[3:7]

    ηₑ = η(i)
    C .-= ηₑ .* ∇C
    W₁ .-= ηₑ .* ∇W₁
    W₂ .-= ηₑ .* ∇W₂
    b₁ .-= ηₑ .* ∇b₁
    b₂ .-= ηₑ .* ∇b₂

    push!(loss_history, loss_value)

    if (i == 1) || (i % 1000 == 0)
        lossₑ = loss(xs_train, ys_train, C, W₁, b₁, W₂, b₂)
        @printf("Iteration %5d (epoch %6.3f): loss = %f\n", i, i / (n_train / minibatch_size), lossₑ)
    end
end

loss_history = float.(loss_history)

loss_train = loss(xs_train, ys_train, C, W₁, b₁, W₂, b₂)
loss_dev = loss(xs_dev, ys_dev, C, W₁, b₁, W₂, b₂)
loss_test = loss(xs_test, ys_test, C, W₁, b₁, W₂, b₂)

println("Training loss: $loss_train")
println("     Dev loss: $loss_dev")
println("    Test loss: $loss_test")

# Plot loss history

let
    fig = Figure(resolution = (900, 600), padding=100)
    ax = Axis(fig[1, 1],
        xlabel = "Epochs",
        ylabel = "Cross-entropy loss",
        xgridvisible = false,
        ygridvisible = false
    )

    epochs = (1:iters) ./ (n_train / minibatch_size)
    lines!(epochs, loss_history, linewidth=2)
    xlims!(ax, (1, maximum(epochs)))
    save("mlp_loss.png", fig, px_per_unit=2)
end

# Plot embedding space

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

# Sample names from trained MLP

function sample_name_mlp(C, W₁, b₁, W₂, b₂)
    context = ones(Int, block_size)
    name = ""

    while true
        embedding = reshape(permutedims(C[context, :], (2, 1)), (1, block_size * dim_embedding))
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
