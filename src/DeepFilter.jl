struct DeepFilter{Tf, Tg, Tw, Tz, Tk, Th} <: AbstractDeepFilter
    fn::Tf
    g::Tg
    w0::Tw
    z0::Tz
    kn::Tk
    h::Th
end
Flux.params(df::DeepFilter) = params((df.fn,df.g,df.kn,df.z0,df.w0,df.h))

function DeepFilter(ny::Int,nu::Int,nz::Int,nh::Int)

    dy2 = MvNormal(2, 0.05f0)

    np = 1
    fn  = Chain(Dense(2nz+nu,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nz))
    g  = Chain(Dense(nz,nh,tanh), Dense(nh,ny))
    # attention = Chain(Dense(nz+ny, nz), LSTM(nz, np), softmax)
    # attend(z,e) = sum(attention([z;e])' .* z, dims=2)
    z0 = Chain(Dense(nz,nh,tanh), Dense(nh,nz))
    w0 = Chain(Dense(4ny,nh,tanh), Dense(nh,2nz))
    # w0[2].b[nz+1:end] .= 1
    kn  = Chain(Dense(nz+ny,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nz))
    h = Chain(Dense(nz+ny, nh, tanh), Dense(nh, nz))
    df = DeepFilter(fn,g,w0,z0,kn,h)
end

const ⊗ = Zygote.dropgrad

function sim(df::DeepFilter, y, u, feedback=true, noise=true)
    fn,g,kn,z0,w0 = df.fn,df.g,df.kn,df.z0,df.w0
    # y,u = yu
    z   = z0(samplenet(w0([y[1];y[2];y[3];y[4]]),noise)[3])
    yh = []
    yh2 = []
    zh  = []
    zp  = []
    for t in 1:length(y)
        ŷ   = g(z)
        push!(yh, ŷ)
        e   = y[t] .- ŷ
        zc  = k(df, z,feedback.*e)
        push!(zh, mean(z, dims=2)[:])
        push!(zp, z)
        ze  = hf(df,z,y[t])
        ŷ = g(ze)
        push!(yh2, ŷ)
        z   = f(df, z,u[t], zc)
    end
    yh, zh, zp, yh2
end


function loss(i,y,u,df::DeepFilter,Ta)
    fn,g,kn,z0,w0,h = df.fn,df.g,df.kn,df.z0,df.w0,df.h
    T = length(y)
    c = min(1, 0.01 + i/Ta)
    z = z0(samplenet(w0([y[1];y[2];y[3];y[4]]))[3])
    l1 = l2 = 0f0
    for t in 1:T
        ŷ   = g(z)
        e   = y[t] .- ŷ
        l1 += varloss(e, 0.05f0)
        zc  = k(df,z,e)
        l2 += sum(abs2,zc)
        ze  = hf(df,⊗(z),y[t])
        e2  = y[t] .- ⊗(g)(ze)
        l2 += varloss(e2, 0.05f0)
        # l2 += sum(x->norm(x)^2,zc)/size(zc,2)
        z = f(df,z,u[t], zc)
    end
    Float32(c*l1/T), Float32(l2/T)
end
