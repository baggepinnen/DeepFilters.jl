# Added norm(e) to k in DVO

cd(@__DIR__)
using DeepFilters, Plots, Flux, Zygote, LinearAlgebra, Statistics, Random, Distributions, DSP
default(lab="", grid=false)
# include("SkipRNN.jl")
Random.seed!(0)

include("pendcart.jl")
##
Z,Y,U = generate_data_pendcart(4,100)


cb = function (i=0)
    Flux.reset!(df)
    i % 1500 == 0 || return
    lm = [ot.loss1 ot.loss2]
    # lm = length(loss1) > Ta ? lm[Ta:end,:] : lm
    # lm = filt(ones(80), [80], lm, fill(lm[1,1], 79))
    fig = plot(lm, layout=@layout([[a;b] c]), sp=1:2, yscale=[:log10 :identity])# yscale=minimum(lm) <= 0 ? :identity : :log10)

    z,y,u = generate_data_pendcart(5, [pi-0.1, 0])
    y = collect(eachcol(y))
    t = range(0,step=h, length=length(y))
    yh,_, = sim(df,y,u, true, false)
    ##
    plot!(t, reduce(hcat,y)', sp=3)
    plot!(t, reduce(hcat, getindex.(yh, 1, :))', l=(:green,), sp=3, markerstrokecolor=:auto)
    # scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.2,:green), sp=3, markerstrokecolor=:auto)
    yh,zh = sim(df,y,u, false, false)
    plot!(t, reduce(hcat, getindex.(yh, 1, :))', l=(:black,), sp=3, markerstrokecolor=:auto)
    vline!([1], l=(:magenta, :dash), sp=3)
    # scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.2,:black), sp=3, markerstrokecolor=:auto)
    display(fig)
end

##

# opt = Momentum(0.00001f0, 0.8)
opt = ADAGrad(0.01f0)
ot = OptTrace()
# sched = I -> (I ÷ 500) % 2 == 0 ? 0.01 : 0.01
df = DeepFilters.DVO(2,1,3,25)

# df = DeepFilter(1,1,10,50)
# df.g[end].b[end÷2+1:end] .= log(0.05)
pars = Flux.params(df)
# loss(1,Y[1],U[1],df,1000)
Zygote.refresh()
# (l1,l2), back = Zygote.forward(df->loss(1,Y[1],U[1],df,1000), df)
# grads = back((1f0,1f0))
# @btime Zygote.gradient(()->+(loss(1,$(Y[1]),$U[1],df, 1000)...), $pars)
# grads = Zygote.gradient(()->+(loss(1,Y[1],U[1],df, 1000)...), pars)
# 10.360 ms (58637 allocations: 4.43 MiB) # params
# 14.011 ms (77238 allocations: 4.18 MiB) # df
train(df, Y, U, 1000, opt, cb=cb, batchsize=2, ot = ot)
##
Random.seed!(123)
plots = map(1:9) do i
    # i = 10
    z,y,u = generate_data_pendcart(5)
    y = collect(eachcol(y))
    # yt = cos.(z[1,:])
    yh,zh = sim(df,y,u, false, false)
    YH = reduce(hcat,yh)'
    plot(reduce(hcat,y)', layout=1, l=(2,))
    # scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.5,:black), sp=2, markerstrokecolor=:auto)
    plot!(YH, l=(2,), xaxis=false, ylims=(-1.1,1.1))
    plot!(reduce(hcat,u)', l=(2,0.2,:green))
end
plot(plots...) |> display

## Plot tubes

Zs = map(1:length(Z)) do i
    yh,zh = sim(df,Y[i],U[i], true, false)
    zh = reduce(hcat, zh)'
    zmat = reduce(hcat,Z[i])'[1:end-1,:]
    zmat[:,1] .= mod2pi.(zmat[:,1])
    zh,zmat
end

zh   = reduce(vcat, getindex.(Zs,1))
zmat = reduce(vcat, getindex.(Zs,2))
s    = svd(zh .- mean(zh, dims=1))
# zh   = s.U.*s.S'

fig = plot(layout=2)
scatter3d!(eachcol(zh)..., m=(2,), zcolor=zmat[:,1], sp=1, markerstrokealpha=0, layout=2)
scatter3d!(eachcol(zh)..., m=(2,), zcolor=zmat[:,2], sp=2, markerstrokealpha=0)
display(fig)

## Plot correlations
plots = map(Iterators.product(eachcol(zh), eachcol(zmat))) do (zh,z)
    scatter(z,zh, m=(0.5,2))
end
plot(plots...) |> display


## Plot particles
noise = true
z,y,u = generate_data_pendcart(5, [pi-0.1, 0])
y = collect(eachcol(y))
zmat = z'
yt = cos.(zmat[:,1])
t = range(0,step=h, length=length(y))
# yh,zh,zp = sim((YU[i][1], YU[i][2]), false, true)
# zmat = reduce(hcat,trajs_state[i])'
yh,zh,yh2 = sim(df,y,u, false, noise)
_, s = DeepFilters.simvar(df,y,u, false; samples=100)
# zpj = reduce(hcat, zp)'
YH = reduce(hcat, yh)'
YH2 = reduce(hcat, yh2)'
plot(t,yt, lab="\$y\$", l=(3,), layout=(3,1), size=(400, 500))
plot!(t,YH, lab="\$\\hat{y}_0\$", ribbon= 2sqrt.(s), sp=1)
vline!([1], l=(:magenta, :dash))

yh,zh,yh2 = sim(df,y,u, true, noise)
YH = reduce(hcat, yh)'
YH2 = reduce(hcat, yh2)'
plot!(t,YH, lab="\$\\hat{y}_t\$", sp=1)


plot!(t,u', lab="u", sp=2, seriestype=:steps)

z,y,u = generate_data_pendcart(5, [pi+0.1, 0])
y = collect(eachcol(y))
yh,zh,yh2 = sim(df,y,u, false, noise)
_, s = DeepFilters.simvar(df,y,u, false; samples=100)

YH = reduce(hcat, yh)'
YH2 = reduce(hcat, yh2)'
plot!(t,yt, lab="\$y\$", l=(3,), sp=3)
plot!(t,YH, lab="\$\\hat{y}\$", ribbon= 2sqrt.(s), sp=3)
vline!([1], l=(:magenta, :dash), sp=3)

yh,zh,yh2 = sim(df,y,u, true, noise)
YH = reduce(hcat, yh)'
YH2 = reduce(hcat, yh2)'
plot!(t,YH, lab="\$\\hat{y}_t\$", sp=3)

plot!(t,u', lab="u", sp=2, seriestype=:steps)
##

# QUESTION: how can one construct a reparametrization trick to sample from a categorical variable?



# g = Dense(1,1)
# x = [1]
# function testdrop(g,x)
#     y = g(x)
#     l1 = sum(y)
#     # l2 = sum(Zygote.dropgrad(g)(x)) + y
#     l2 = sum(Zygote.dropgrad(g)(x)) + sum(Zygote.dropgrad(y))
#     l1,l2
# end
# Zygote.refresh()
# (l1,l2),back = Zygote.forward(g->testdrop(g,x), g)
# grads = back((1,1))
