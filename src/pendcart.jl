using OrdinaryDiffEq, ChangePrecision

const h = 0.1

struct SawtoothGenerator{T <: Real} <: Function
    f::T #Frequency
    p::T #Phase
end
sawfun(t) = t - floor(t)
SawtoothGenerator(f)    = SawtoothGenerator(f, 0)
# SawtoothGenerator(f, p) = SawtoothGenerator{typeof(f)}(f, p)

function (ref::SawtoothGenerator)(t::Real)
    sawfun(ref.f*t+ref.p)
end

saw = SawtoothGenerator(1)
@changeprecision Float32 function controller(x, t)
    l = 1.0; d = 0.5
    t < 1 && (return 0.)
    u = 0#t > 1 ? 4(saw(t)-0.5) : 0.
    # u += - 2θ - 2x[2]
    u += l*(d+0.1)*x[2]/cos(x[1])
    clamp(u, -5, 5)
end


@changeprecision Float32 function pendcart(xd,x,p,t)
    g = 9.82; l = 1.0; d = 0.5
    u = controller(x, t)
    xd[1] = x[2]
    xd[2] = -g/l * sin(x[1]) + u/l * cos(x[1]) - d*x[2]
    xd
end

function centerangle(x)
    c = round(Int, median(x)/(2π))
    mod2pi.(x .- (2π*c - π)) .- π
end

@changeprecision Float32 function generate_data_pendcart(T,
    u0::AbstractVector = Float32[2pi*rand(), 6*2*(rand()-0.5)])
    tspan = (0f0,Float32(T))
    prob = ODEProblem(pendcart,u0,tspan)
    sol = solve(prob,Tsit5())
    z = reduce(hcat, sol(0:h:T).u)
    y = vcat(sin.(z[1:1,:]), cos.(z[1:1,:])) .+ 0.05 .* randn.()
    # y = cos.(z[1:1,:]) .+ 0.05 .* randn.()
    z[1,:] .= centerangle(z[1,:])
    z,y, controller.(eachcol(z), 0:h:T)'
end

function generate_data_pendcart(T::Number,N::Int)
    trajs_full = [generate_data_pendcart(T) for i = 1:N]
    # trajs_full = [generate_data_pendcart(5, [pi-0.1, 0]), generate_data_pendcart(5, [pi+0.1, 0])]
    Y = map(trajs_full) do (_,t,_)
        [copy(c) for c in eachcol(t)]
    end
    U = map(trajs_full) do (_,_,t)
        [copy(c) for c in eachcol(t)]
    end
    Z = map(trajs_full) do (t,_,_)
        [copy(c) for c in eachcol(t)]
    end
    Z,Y,U
end
