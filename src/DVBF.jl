struct DVBF{Tf, Tg, Tw, Tz, Tk} <: AbstractDeepFilter
    f::Tf
    g::Tg
    w0::Tw
    z0::Tz
    k::Tk
end
Flux.params(df::DVBF) = params((df.f,df.g,df.k,df.z0,df.w0))

function DVBF(ny::Int,nu::Int,nz::Int,nh::Int)
    dy2 = MvNormal(2, 0.05f0)
    f  = Chain(Dense(2nz+nu,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nz))
    g  = Chain(Dense(nz,nh,tanh), Dense(nh,ny))
    z0 = Chain(Dense(nz,nh,tanh), Dense(nh,nz))
    w0 = Chain(Dense(4ny,nh,tanh), Dense(nh,2nz))
    k  = Chain(Dense(nz+ny,nh,tanh), Dense(nh,nh,tanh), Dense(nh,2nz))
    df = DVBF(f,g,w0,z0,k)
end

function sim(df::DVBF, y, u, feedback=true, noise=true)
    g,z0,w0 = df.g,df.z0,df.w0
    z   = z0(samplenet(w0([y[1];y[2];y[3];y[4]]),noise)[3])
    yh = []
    zh  = []
    for t in 1:length(y)-1
        ŷ   = g(z)
        push!(yh, ŷ)
        μ, σ, zc = samplenet(k(df,z,y[t+1]))
        push!(zh, z)
        z   = f(df, z, u[t], feedback*(noise ? zc : μ))
    end
    ŷ   = g(z)
    push!(yh, ŷ)
    yh, zh, 0
end


function loss(i,y,u,df::DVBF,Ta)
    T = length(y)
    g,z0,w0 = df.g,df.z0,df.w0
    c        = min(1, 0.01 + i/Ta)
    μ, σ, zc = samplenet(w0([y[1];y[2];y[3];y[4]]))
    z        = z0(zc)
    l1       = 0f0
    l2       = sum(klng.(μ, σ, c))
    for t in 1:length(y)-1
        ŷ   = g(z)
        e   = y[t] .- ŷ
        l1 += varloss(e, 0.05f0)
        μ, σ, zc  = samplenet(k(df,z,y[t+1]))
        l2 += sum(klng.(μ, σ, c))
        z = f(df,z,u[t], zc)
    end
    Float32(c*l1/T), Float32(l2/T)
end
