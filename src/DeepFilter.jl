struct DeepFilter{Tf, Tg, Tw, Tz, Tk, Th} <: AbstractDeepFilter
    f::Tf
    g::Tg
    w0::Tw
    z0::Tz
    k::Tk
    h::Th
end
Flux.params(df::DeepFilter) = params((df.f,df.g,df.k,df.z0))
# Flux.params(df::DeepFilter) = params((df.f,df.g,df.k,df.z0,df.w0))

function DeepFilter(ny::Int,nu::Int,nz::Int,nh::Int)
    f  = Chain(Dense(2nz+nu,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nz))
    g  = Chain(Dense(nz,nh,tanh), Dense(nh,nh,σ), Dense(nh,ny))

    w0 = Chain(Dense(nz,nh,tanh), Dense(nh,nz))
    z0 = Chain(Dense(4ny,nh,tanh), Dense(nh,nz))

    # z0 = Chain(Dense(nz,nh,tanh), Dense(nh,nz))
    # w0 = Chain(Dense(4ny,nh,tanh), Dense(nh,2nz))

    k  = Chain(Dense(nz+ny,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nz))
    h = Chain(Dense(nz+ny, nh, tanh), Dense(nh, nz))
    df = DeepFilter(f,g,w0,z0,k,h)
end

const ⊗ = Zygote.dropgrad

function sim(df::DeepFilter, y, u, feedback=true, noise=true)
    g,z0,w0 = df.g,df.z0,df.w0
    # y,u = yu
    z   = z0([y[1];y[2];y[3];y[4]])
    # z   = z0(samplenet(w0([y[1];y[2];y[3];y[4]]),noise)[3])
    yh  = []
    yh2 = []
    sy  = []#
    zh  = []
    for t in 1:length(y)
        # ŷ,σy   = splitμσ(g(z))
        ŷ   = g(z)
        push!(yh, ŷ)
        # push!(sy, σy)
        e   = y[t] .- ŷ
        ξ   = k(df, z,feedback.*e)
        push!(zh, z)
        ze  = hf(df,z,y[t])
        # ŷ = splitμσ(g(ze))[1]
        ŷ = g(ze)
        push!(yh2, ŷ)
        z   = f(df, z,u[t], ξ)
    end
    yh, zh, yh2#, sy
end


function loss(i,y,u,df::DeepFilter,Ta)
    g,z0,w0,h = df.g,df.z0,df.w0,df.h
    T = length(y)
    c = min(1, 0.01 + i/Ta)
    z = z0([y[1];y[2];y[3];y[4]])
    # z = z0(samplenet(w0([y[1];y[2];y[3];y[4]]))[3])
    l1 = l2 = 0f0
    for t in 1:T
        # ŷ,σy = splitμσ(g(z))
        ŷ    = g(z)
        e    = y[t] .- ŷ
        l1  += varloss(e, 0.05)
        # l1  += sum(varloss.(e, σy))
        ξ    = k(df,z,e)
        l2  += sum(abs2,ξ)
        # ze  = hf(df,⊗(z),y[t])
        # e2  = y[t] .- ⊗(g)(ze)
        # l2 += varloss(e2, σy)
        # l2 += sum(x->norm(x)^2,ξ)/size(ξ,2)
        z = f(df,z,u[t], ξ)
    end
    Float32(l1/T), Float32(l2/T)
end
