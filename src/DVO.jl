struct DVO{Tf, Tg, Tw, Tz, Tk, Tq} <: AbstractDeepFilter
    f::Tf
    g::Tg
    w0::Tw
    z0::Tz
    k::Tk
    q::Tq
end
# Flux.params(df::DVO) = params((df.f,df.g,df.k,df.z0))
Flux.params(df::DVO) = params((df.f,df.g,df.k,df.z0,df.w0,df.q))

function DVO(ny::Int,nu::Int,nz::Int,nh::Int)
    f  = Chain(Dense(3nz+nu,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nz))
    g  = Chain(Dense(nz,nh,tanh), Dense(nh,nh,σ), Dense(nh,ny))

    # w0 = Chain(Dense(nz,nh,tanh), Dense(nh,nz))
    # z0 = Chain(Dense(10ny,nh,tanh), Dense(nh,nz))

    z0 = Chain(Dense(nz,nh,tanh), Dense(nh,nz))
    w0 = Chain(Dense(4ny,nh,tanh), Dense(nh,2nz))

    k  = Chain(Dense(nz+ny,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nz))
    # TODO: initialize q to saturate the sigmoid.
    q = IAF(Chain(Dense(nz+ny,nh,tanh), Dense(nh,3nz)),
                ntuple(i->MADE(2nz, [nh,nh], 2nz, false, 1), 3))
    # q = Chain(Dense(nz+ny,nh,tanh), Dense(nh,nh,tanh), Dense(nh,2nz))
    df = DVO(f,g,w0,z0,k,q)
end


f(df::DVO,z,u,ξ,w) = df.f([z;u;ξ;w]) + 0.9f0*z

function k(df::DVO,z,e)
    # kn([z;mean(y)*ones(1,np)])
    df.k([z;e])
end

function q(df::DVO,z,y)
    # kn([z;mean(y)*ones(1,np)])
    df.q([z;y])
end

const ⊗ = Zygote.dropgrad

function sim(df::DVO, y, u, feedback=true, noise=true)
    g,z0,w0 = df.g,df.z0,df.w0
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
        ξ  = k(df, z, feedback.*e)
        push!(zh, z)
        # w    = samplenet(q(df,z,y[t+1]),noise)[3]
        w,ll,_ = q(df,z,y[t+1])
        # @show size.((z,u[t], ξ ,w, μ))
        z   = f(df, z,u[t], ξ, feedback ? w : noise ? randn(length(z)) : 0w)
    end
    ŷ   = g(z)
    push!(yh, ŷ)
    yh, zh#, sy
end


function loss(i,y,u,df::DVO,Ta)
    g,z0,w0 = df.g,df.z0,df.w0
    T = length(y)
    c = min(1, 0.01 + i/Ta)
    # z = z0(reduce(hcat, y[1:10])')
    μ, σ, w = samplenet(w0([y[1];y[2];y[3];y[4]]))
    l2 = sum(klng.(μ, σ))
    z = z0(w)
    l1 = 0f0
    for t in 1:T-1
        # ŷ,σy = splitμσ(g(z))
        ŷ     = g(z)
        e     = y[t] .- ŷ
        l1   += varloss(e, 0.05)
        ξ     = k(df,z,e)
        l2   += 10sum(abs2, ξ)
        # μ,σ,w = samplenet(q(df,z,y[t+1]))
        w,ll,_ = q(df,z,y[t+1])
        l2   += sum(ll)

        z     = f(df,z,u[t], ξ ,w)
    end
    ŷ    = g(z)
    e    = y[end] .- ŷ
    l1  += varloss(e, 0.05)
    Float32(l1/T), Float32(l2/T)
end
