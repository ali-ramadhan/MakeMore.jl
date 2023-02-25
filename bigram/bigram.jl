using Printf
using StatsBase
using CairoMakie

names = readlines("../names.txt")

# Look at name length distribution

let
    name_lengths = [length(name) for name in names]

    @printf("Length | Frequency\n")
    for l in minimum(name_lengths):maximum(name_lengths)
        c = count(name_lengths .== l)
        @printf("    %2d | %d (%.2f%%)\n", l, c, 100 * c / length(names))
    end
end

# Bigrams

all_chars = vcat('.', 'a':'z')
nc = length(all_chars)
bigrams = zeros(Int, nc, nc)

char2int = Dict(c => i for (i, c) in enumerate(all_chars))

for w in names
    chars = vcat('.', collect(w), '.')
    for (c1, c2) in zip(chars, chars[2:end])
        i1, i2 = char2int[c1], char2int[c2]
        bigrams[i1, i2] += 1
    end
end

let
    fig = Figure(resolution = (1500, 1500))
    ticklabels = string.(all_chars)
    ax = Axis(fig[1, 1])

    hmap = heatmap!(ax, bigrams, colormap=:matter)

    for i in eachindex(all_chars), j in eachindex(all_chars)
        b = string(all_chars[i], all_chars[j])

        color = bigrams[i, j] < 4000 ? :black : :white
        text!(ax, b,
            position = (i, j),
            align = (:center, :top),
            color = color
        )

        text!(ax, string(bigrams[i, j]),
            position = (i, j),
            align = (:center, :bottom),
            color = color
        )
    end

    Colorbar(fig[1, 2], hmap; label="frequency", width=15, ticksize=15)

    hidedecorations!(ax)

    save("bigram_frequency.png", fig, px_per_unit=2)
end

# Sampling from the bigram model

γ = 1 # Model smoothing
B = bigrams .+ 1
P = B ./ sum(B, dims=2)

function sample_name(bigrams)
    name = "."
    while true
        prev_char = name[end]
        freq_next_letter = bigrams[char2int[prev_char], :]
        next_char = sample(all_chars, FrequencyWeights(freq_next_letter))
        name *= next_char
        if name[end] == '.'
            return name[2:end-1]
        end
    end
end

function negative_log_likelihood(name)
    name = '.' * name * '.'
    nll = 0
    nb = 0 # count number of bigrams
    for i in 1:length(name)-1
        c1, c2 = name[i:i+1]
        i1, i2 = char2int[c1], char2int[c2]
        nll += log(P[i1, i2])
        nb += 1
    end
    return -nll / nb
end

#####
##### Training a "neural network" y = f(x) = Wx to learn the exact same bigram model.
##### Here, x is a one-hot vector for the current character and y is a one-hot vector for the next character.
##### W is a square matrix of weights (which when trained should correspond to the bigrams probability matrix).
#####

using OneHotArrays
using Flux

function onehot_name(name)
    name = '.' * name * '.'
    c1 = [c for c in name[1:end-1]]
    c2 = [c for c in name[2:end]]
    return onehotbatch(c1, all_chars), onehotbatch(c2, all_chars)
end

function training_dataset(names)
    xs, ys = onehot_name(names[1])

    for name in names[2:end]
        x, y = onehot_name(name)
        xs = hcat(xs, x)
        ys = hcat(ys, y)
    end

    return xs, ys
end

# Negative log-likelihood loss function
function loss(xs, ys, W)
    logits = W * xs
    probs = softmax(logits)
    nll = -mean(log.(probs[ys]))
    return nll
end

xs, ys = training_dataset(names)
W = randn(nc, nc)

η = 10 # Learning rate / step size

for epoch in 1:100
    loss_value, grads = Flux.withgradient(loss, xs, ys, W)

    # Gradient descent
    ∇W = grads[3]
    W .-= η .* ∇W

    println("epoch $epoch: loss = $loss_value")
end

function sample_name_nn(W)
    name = "."
    while true
        prev_char = name[end]
        x = onehot(prev_char, all_chars)
        probs = softmax(W * x)
        next_char = sample(all_chars, Weights(probs))
        name *= next_char
        if name[end] == '.'
            return name[2:end-1]
        end
    end
end
