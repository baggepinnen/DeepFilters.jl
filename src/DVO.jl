struct DVO{Tf, Tg, Tw, Tz, Tk, Th} <: AbstractDeepFilter
    fn::Tf
    g::Tg
    w0::Tw
    z0::Tz
    kn::Tk
    h::Th
end
# Flux.params(df::DVO) = params((df.fn,df.g,df.kn,df.z0))
Flux.params(df::DVO) = params((df.fn,df.g,df.kn,df.z0,df.w0))

function DVO(ny::Int,nu::Int,nz::Int,nh::Int)
    fn  = Chain(Dense(2nz+nu,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nz))
    g  = Chain(Dense(nz,nh,tanh), Dense(nh,nh,σ), Dense(nh,ny))

    # w0 = Chain(Dense(nz,nh,tanh), Dense(nh,nz))
    # z0 = Chain(Dense(10ny,nh,tanh), Dense(nh,nz))

    z0 = Chain(Dense(nz,nh,tanh), Dense(nh,nz))
    w0 = Chain(Dense(4ny,nh,tanh), Dense(nh,2nz))

    kn  = Chain(Dense(nz+2ny,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nz))
    h = Chain(Dense(nz+ny, nh, tanh), Dense(nh, nz))
    df = DVO(fn,g,w0,z0,kn,h)
end

function k(df::DVO,z,e,y)
    # kn([z;mean(y)*ones(1,np)])
    df.kn([z;e;y])
end

const ⊗ = Zygote.dropgrad

function sim(df::DVO, y, u, feedback=true, noise=true)
    fn,g,kn,z0,w0 = df.fn,df.g,df.kn,df.z0,df.w0
    # y,u = yu
    # z   = z0(reduce(vcat, y[1:10]))
    z   = z0(samplenet(w0([y[1];y[2];y[3];y[4]]),noise)[3])
    yh  = []
    yh2 = []
    sy  = []#
    zh  = []
    for t in 1:length(y)-1
        # ŷ,σy   = splitμσ(g(z))
        ŷ   = g(z)
        push!(yh, ŷ)
        # push!(sy, σy)
        e   = y[t] .- ŷ
        zc  = k(df, z, e, y[t+1])
        push!(zh, z)
        ze  = hf(df,z,y[t])
        # ŷ = splitμσ(g(ze))[1]
        ŷ = g(ze)
        push!(yh2, ŷ)
        z   = f(df, z,u[t], feedback.*zc + noise*randn(size(zc)))
    end
    ŷ   = g(z)
    push!(yh, ŷ)
    yh, zh, yh2#, sy
end


function loss(i,y,u,df::DVO,Ta)
    fn,g,kn,z0,w0,h = df.fn,df.g,df.kn,df.z0,df.w0,df.h
    T = length(y)
    c = min(1, 0.01 + i/Ta)
    # z = z0(reduce(hcat, y[1:10])')
    μ, σ, w = samplenet(w0([y[1];y[2];y[3];y[4]]))
    l2 = sum(klng.(μ, σ))
    z = z0(w)
    l1 = 0f0
    for t in 1:T-1
        # ŷ,σy = splitμσ(g(z))
        ŷ    = g(z)
        e    = y[t] .- ŷ
        l1  += varloss(e, 0.05)
        zc   = k(df,z,e,y[t+1])
        l2  += sum(abs2, zc)
        z = f(df,z,u[t], zc + randn(Float32, size(zc)))
    end
    ŷ    = g(z)
    e    = y[end] .- ŷ
    l1  += varloss(e, 0.05)
    Float32(l1/T), Float32(l2/T)
end
