module DeepFilters

using Plots, Flux, Zygote, LinearAlgebra, Statistics, Random, Printf, IterTools, Distributions, ChangePrecision, SliceMap, Juno
using Flux: params

export AbstractDeepFilter, DeepFilter, DVBF, OptTrace
export train, loss, sim

abstract type AbstractDeepFilter end

include("DeepFilter.jl")
include("DVBF.jl")

Base.@kwdef struct OptTrace
    loss1::Vector{Float64} = Float64[]
    loss2::Vector{Float64} = Float64[]
end
Base.push!(ot::OptTrace, l1, l2) = (push!(ot.loss1, l1); push!(ot.loss2, l2);)
Base.length(ot::OptTrace) = length(ot.loss1)

function train(df, y, u, epochs, opt;
    cb             = i->nothing,
    schedule       = I->copy(opt.eta),
    batchsize      = 1,
    annealing_time = 4000,
    ot             = OptTrace())

    ps = params(df)
    dataset = collect(zip(y, u))
    @assert length(dataset) % batchsize == 0 "batchsize must divide length(y)"
    @progress for epoch = 1:epochs
        for i = 1:batchsize:length(dataset)-batchsize+1
            I = length(ot)+1
            opt.eta = schedule(I)
            # Flux.reset!(model)
            (l1,l2), back = Zygote.forward(ps) do
                mean(loss(I, d..., df,annealing_time) for d in dataset[i:i+batchsize-1])
            end
            grads = back((1f0,1f0))
            push!(ot, l1, l2)
            Flux.Optimise.update!(opt, ps, grads)
            cb(I*batchsize)
        end
    end
    ot
end

function k(df,z,e)
    # kn([z;mean(y)*ones(1,np)])
    df.kn([z;e])
end
f(df,z,u,zc) = df.fn([z;u;zc]) + 0.9f0*z
hf(df,z,y) = df.h([z;y])

function hstack(xs, n)
    buf = Zygote.Buffer(xs, length(xs), n)
    for i = 1:n
        buf[:, i] = xs
    end
    return copy(buf)
end

Zygote.@adjoint function Base.reduce(::typeof(hcat), V::AbstractVector{<:AbstractVector})
    reduce(hcat, V), dV -> (nothing, collect(eachcol(dV)))
end

function stats(e)
    μ = mean(e, dims=2)
    μ, sum(abs2, e .- μ, dims=2) ./ size(e,2) .+ 1f-3
end

@changeprecision Float32 function kl(e::AbstractMatrix, μ2, σ2, c = 1)
    μ1, σ1² = stats(e)
    lσ1 = log.(sqrt.(σ1²))
    lσ2 = log.(σ2)#log.(sqrt.(var(dy)))
    l = 0f0
    for i = eachindex(μ1)
        l += c*2lσ2[i] - 2lσ1[i] +
        c*(σ1²[i] + abs2(μ1[i] - μ2[i]))/(σ2[i]^2 + 1f-5)
    end
    0.5f0l
end

@changeprecision Float32 function klng(μ, σ, c = 1)
    σ² = σ^2
    lσ = log(σ)
    0.5*(-2lσ + c*(σ² + abs2(μ)))
end

@changeprecision Float32 function kl2(e, c = 1)
    μ, σ² = stats(e)
    0.5c*(sum(abs2,μ) + sum(σ²)) - sum(log, σ²)
end

Base.:(+)(a::Tuple{Float32,Float32}, b::Tuple{Float32,Float32}) = (a[1]+b[1], a[2]+b[2])
Base.:(/)(a::Tuple{Float32,Float32}, b::Real) = (a[1]/b, a[2]/b)

function varloss(e,σ)
    error("getindex in sum is broken in Zygote")
    sum(i->0.5f0*abs2(e[i])/σ[i]^2 + log(σ[i]), eachindex(e))
end

function varloss(e,σ::Number)
    sum(e->0.5f0*(abs2(e)/σ^2), e)
end


function partlik(e, σ)
    w = mapcols(e->-0.5*sum(abs2.(e)./σ^2), e)
    offset = maximum(w)
    log(sum(w->exp(w - offset), w)) + offset - log(size(e,2))
end


function splitμσ(μσ)
    nz = length(μσ) ÷ 2
    μ = μσ[1:nz]
    σ = exp.(μσ[(nz+1):end])
    μ, σ
end

function samplenet(μσ, noise=true)
    np = size(μσ,2)
    μ, σ = splitμσ(μσ)
    nz = length(μ)
    w = μ .+ σ .* randn(Float32, nz, np).*noise
    μ, σ, w
end

function dropout(e)
    mapcols(e->rand((0,0,0,1))*e, e)
end

Zygote.@adjoint function dropout(e)
    mask = rand((0,0,0,1))
    e.*mask, x->(x.*mask,)
end

Zygote.@nograd function randu(nz,np)
    rand(Float32, nz, np) .- 0.5
    # randn(Float32, nz, np)
end

Zygote.@nograd function randmm(nz,np)
    randn(Float32, nz, np) .+ rand((-4,4), 1, np)
end


end # module
