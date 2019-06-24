using Distributions, Flux, Zygote, Test


mutable struct IAF{T1,T2}
    encoder::T1
    ann::T2
end
Flux.@treelike IAF



# a = randn(10)
# splitin(a,2)

function Base.rand(iaf::IAF, x, ϵ = randn(iaf.ann[1].nin ÷ 2))
    µ,lσ,h = iaf.encoder(x) |> splitin(3)
    trans = (µ,exp.(lσ))
    z = exp.(lσ) .* ϵ .+ µ
    l = -sum(lσ .+ 0.5ϵ)
    for t = 1:length(iaf.ann)
        μ, s = iaf.ann[t]([z; h]) |> splitin(2)
        σ = NNlib.σ.(s)
        trans = ((μ, σ),trans)
        z = @. σ * z + (1 - σ)μ
        l -= sum(log.(σ))
    end
    z,l,trans
end

(iaf::IAF)(x) = rand(iaf, x)

# rand(iaf, y[:,1])

function transformations(iaf::IAF, x, ϵ)
    µ,lσ,h = iaf.encoder(x) |> splitin(3)
    trans = (µ,exp.(lσ))
    z = µ
    for t = 1:T
        μ, s = iaf.ann[t]([z; h]) |> splitin(2)
        σ = NNlib.σ.(s)
        trans = ((μ, σ),trans)
        z = @. σ * z + (1 - σ)μ
    end
    trans
end

# function Base.inv(iaf, x, y, trans = transformations(iaf, x))
#     for i in 1:T
#         (μ, σ),trans = trans
#         y = @. (y - (1 - σ)μ) / σ
#     end
#     (μ, σ) = trans
#     y = @. (y - μ) / σ
#     y
# end

function Base.inv(iaf::IAF, x, y)
    error("Incorrect")
    order = iaf.ann[1].m[0]
    order = filter(x-> x <= length(y), order)
    # e = Zygote.Buffer(y)
    e = similar(y)
    e[1:length(y)] = zeros(length(y))
    e[order[1]] = y[order[1]]
    for d = 2:length(y)
        z,_,trans = rand(iaf, x, copy(e))
        for i in 1:T
            (μ, σ),trans = trans
            z = @. (z - (1 - σ)μ) / σ
        end
        (μ, σ) = trans
        @show z = @. (z - μ) / σ
        e[order[d]] = z[order[d]]
    end
    copy(e)
end


# y1 = randn(1000)
# y2d = Normal.(y1.^2, 1)
# y2 = rand.(y2d)
# y = [y1 y2]'
# yv = eachcol(y)
# scatter(y1,y2)
#
# nh = 20
# ni = 2
# T = 2
# # iaf = IAF(Chain(Dense(ni,2,tanh), Dense(2,3ni)),
# #          ntuple(_->Chain(Dense(2ni,nh,tanh), Dense(nh,nh,tanh), Dense(nh,2ni)),T))
#
# iaf = IAF(Chain(Dense(ni,2,tanh), Dense(2,3ni)),
#             ntuple(i->MADE(2ni, [nh,nh], 2ni, true, 1),T))

# trans = transformations(iaf, zeros(2))
# a,trans = trans
# ϵ = randn(2)
# s,_, trans = rand(iaf, zeros(2), ϵ)
# ϵ2 = inv(iaf, zeros(2), s)
# display((round.(ϵ, digits=3), round.(ϵ2, digits=3)))
# @test ϵ ≈ ϵ2

# function klng(μ, σ, c = 1)
#     σ² = σ^2
#     lσ = log(σ)
#     0.5*(-2lσ + c*(σ² + abs2(μ)))
# end
#
#
# function varloss(e,σ::Number)
#     sum(e->0.5f0*(abs2(e)/σ^2), e)
# end
#
# function stats(e::Vector{<:AbstractArray})
#     # println("hej")
#     μ = sum(e) ./ length(e)
#     σ2 = sum(e->abs2.(e .- μ), e)
#     μ, σ2 ./ length(e) .+ 1f-5
# end
#
# function cost()
#     Z = zeros(2)
#     ϵs = Zygote.Buffer(Z, 2, 1000)
#     l = 0.
#     for i = 1:1000
#         ϵ = inv(iaf, Z, y[:,i])
#         ϵs[:,i] = ϵ
#         # l += rand(iaf, Z, ϵ)[2]
#     end
#     μ, σ = stats(copy(ϵs))
#     sum(klng.(μ, sqrt.(σ))) #+ l
# end
#
# cost()
# pars = Flux.params((iaf.encoder, iaf.ann...))
# pars = Flux.params(iaf)
# Zygote.refresh()
#
#
# grads = Zygote.gradient(cost, pars)
#
# losses = Float64[]
# function train(epochs)
#     @progress for j = 1:epochs
#         # for i = 1:size(y,2)
#         c,back = Zygote.forward(cost, pars)
#         push!(losses, c)
#         grads = back(1)
#         Flux.Optimise.update!(opt, pars, grads)
#         # end
#         # update_masks.(iaf.ann)
#         j % 2 == 0 && display(plot(plotdist(iaf, 100), plot(losses, yscale=:identity)))
#     end
# end
#
# function plotdist(iaf, N)
#     ys = [rand(iaf, zeros(2))[1] for _ in 1:N]
#     ys = reduce(hcat, ys)'
#     scatter(y1,y2, alpha=0.1)
#     scatter!(eachcol(ys)..., legend=false, alpha=0.2)
# end
#
# function plotinv(iaf, N)
#     ys = [inv(iaf, zeros(2), y) for y in yv]
#     ys = reduce(hcat, ys)'
#     scatter(y1,y2, alpha=0.1)
#     scatter!(eachcol(ys)..., legend=false, alpha=0.2)
# end
# Zygote.refresh()
# # opt = Nesterov(0.001)
# opt = ADAGrad(0.1)
# train(50)
#
# is = reduce(hcat, inv.(Ref(iaf), Ref(zeros(2)), yv))'
# plot(plotdist(iaf, 1000),plot(losses, yscale=:log10),scatter(is[:,1], is[:,2]))
