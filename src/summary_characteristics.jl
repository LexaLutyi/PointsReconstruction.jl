# ϕ -- point process

distance(x, y) = norm(x - y)
distance(x::SVector{N, T}, y::Tuple{Vararg{T, N}}) where T where N = norm(x .- y)
distance(x::Tuple{Vararg{T, N}}, y::Tuple{Vararg{T, N}}) where T where N = norm(x .- y)

distances(ϕ::AbstractVector{T}, x::T) where T = [distance(x, y) for y in ϕ]

distances(ϕ, ψ) = [distance(x, y) for x in ϕ, y in ψ]

distances(ϕ) = @inbounds [distance(ϕ[i], ϕ[j]) for i in 1:length(ϕ) for j in i + 1:length(ϕ)]


function K(rs, ϕ)
    d = distances(ϕ)
    [count(s -> s < r, d) for r in rs]
end


function normK(rs, ϕ)
    k = K(rs, ϕ)
    n = length(ϕ)
    k / (n * (n - 1) ÷ 2)
end


function diffK(rs, ϕ, point_ix, new_point)
    old_point = ϕ[point_ix]
    v = ϕ[1:end .!= point_ix]

    d1 = distances(v, old_point)
    d2 = distances(v, new_point)
    
    k1 = [count(s -> s < r, d1) for r in rs]
    k2 = [count(s -> s < r, d2) for r in rs]
    
    k2 - k1
end


function D(maxk, rs, ϕ)
    d = distances(ϕ, ϕ)
    cnt = zeros(Int, length(rs), maxk)
    
    for (i, r) in enumerate(rs)
        cnt_r = count(s -> s < r, d, dims=1) .- 1
        cnt[i, :] = [count(s -> s ≥ k, cnt_r) for k in 1:maxk]
    end
    cnt
end


function normD(maxk, rs, ϕ)
    d = D(maxk, rs, ϕ)
    d / max(length(ϕ), 1)
end


function T(rs, ϕ)
    n = length(ϕ)
    cnt = zeros(Int, length(rs))
    d = distances(ϕ, ϕ)
    
    for (i, r) in enumerate(rs)
        q = d .< r
        q[diagind(q)] .= false
        cnt[i] = sum(q^2 .* q)
    end
    cnt
end


function normT(rs, ϕ)
    n = length(ϕ)
    t = T(rs, ϕ)
    t / (n * (n - 1) * (n - 2))
end


min_distance(ϕ, x) = mapreduce(p -> distance(p, x), min, ϕ)


function H(rs, ϕ, step::Real)
    xs = 0.:step:1.
    H(rs, ϕ, xs)
end


function H(rs, ϕ, xs::AbstractRange=rs)
    dim = length(eltype(ϕ))
    ϕ0 = Iterators.product(fill(xs, dim)...)
    H(rs, ϕ, ϕ0)
end


function H(rs, ϕ, ϕ0)
    mapreduce(x -> rs .>= min_distance(ϕ, x), +, ϕ0)
end


number_of_points(ϕ, ϕ0) = length(ϕ0)
number_of_points(ϕ, step::Real) = (dim=length(eltype(ϕ)); length(0.:step:1.)^dim)
number_of_points(ϕ, xs::AbstractRange) = (dim=length(eltype(ϕ)); length(xs)^dim)



function normH(rs, ϕ, arg=rs)
    h = H(rs, ϕ, arg)
    n = number_of_points(ϕ, arg)
    h / n
end