struct LSTMFilter{Tf,Tz} <: AbstractDeepFilter
    fn::Tf
    z0::Tz
end
# Flux.@treelike LSTMFilter
Flux.params(df::LSTMFilter) = params((df.fn[1],df.fn[3],df.fn[4],df.fn[2].cell.Wh,df.fn[2].cell.Wi,df.fn[2].cell.b,df.fn[2].init..., df.z0))
# Flux.params(df::LSTMFilter) = params((df.fn,df.g,df.kn,df.z0,df.w0))

function LSTMFilter(ny::Int,nu::Int,nz::Int,nh::Int)
    fn  = Chain(Dense(nu,nz,tanh), LSTM(nz,nz), Dense(nz,nh,tanh), Dense(nh,ny))

    z0 = Chain(Dense(10ny,nh,tanh), Dense(nh,nz))

    df = LSTMFilter(fn,z0)
end

const ⊗ = Zygote.dropgrad

function sim(df::LSTMFilter, y, u, feedback=true, noise=true)
    fn,z0 = df.fn,df.z0
    Flux.reset!(fn)
    # y,u = yu
    z   = z0(reduce(vcat, y[1:10]))
    fn[2].init = fn[2].state = (z, zeros(length(z)))
    # z   = z0(samplenet(w0([y[1];y[2];y[3];y[4]]),noise)[3])
    yh  = []
    yh2 = []
    sy  = []#
    zh  = []
    for t in 1:length(y)
        push!(zh, fn[2].cell.h)
        ŷ   = fn([u[t]])
        push!(yh, ŷ)
    end
    Flux.reset!(fn)
    yh, zh, yh2#, sy
end


function loss(i,y,u,df::LSTMFilter,Ta)
    fn,z0 = df.fn,df.z0
    T = length(y)
    z = z0(reduce(hcat, y[1:10])')
    fn[2].init = fn[2].state = (z, zeros(length(z)))
    # l1 = l2 = 0f0
    # for t in 1:T
    #     ŷ = fn(u[t])
    #     l1 += sum(abs2, y[t].-ŷ)
    # end
    l1 = sum(x->norm(x)^2, df.fn.(u) .- y)
    l2 = 0
    Float32(l1/T), Float32(l2/T)
end
