function ϕ_uniform(dim, n)
    [rand(SVector{dim}) for _ in 1:n]
end


function ϕ_cluster(dim, n, k; delta=0.02)
    m = n ÷ k
    a = [fill(delta, dim) + (1 - 2delta) * p for p in ϕ_uniform(dim, m)]
    @views v = a[1:m]
    for _ in 2:k
        c = v .+ [2 * (p .- 0.5) * delta for p in ϕ_uniform(dim, m)]
        append!(a, c)
    end
    a[1:n]
end


function ϕ_uniform(dim, n, m; replace=false)
    ij = CartesianIndices(Tuple(fill(m, dim)))
    ϕ = sample(ij, n; replace) .|> Tuple .|> SVector{dim}
end


function ϕ_cluster(dim, n, k, m; delta = round(Int, sqrt(k)))
    nk = n ÷ k
    a = map(p -> p .+ delta, ϕ_uniform(dim, nk, m - 2*delta))
    v = view(a, 1:nk)
    
    for _ in 2:2k
        b = map(p -> p .- (delta + 1), ϕ_uniform(dim, nk, 2 * delta + 1; replace=true))
        append!(a, v + b)
        if length(unique(a)) > n
            break
        end
    end
    unique(a)[1:n]
end


function ϕ_net(nline, npoint)
    dx = 1 / npoint
    xs = range(dx / 2, length = npoint, step = dx)
    dy = 1 / nline
    ys = range(dy / 2, length = nline, step = dy)

    a = [[x, y] for x in xs, y in ys]
    b = [[y, x] for x in xs, y in ys]
    reduce(hcat, [a[:]; b[:]])
end


function ϕ_circles(nx, npoint, r=1 / 3nx)
    xs = [r * [cos(x), sin(x)] for x in range(0., length = npoint, step = 2π / npoint)]
    
    dy = 1 / nx
    ys = range(dy / 2, length = nx, step = dy)

    a = reduce(hcat, [Ref([x, y]) .+ xs for x in ys, y in ys])
    reduce(hcat, a[:])
end


# function replace_random_point(ϕ::AbstractMatrix{<:Real})
#     n, dim = size(ϕ)
#     ix = rand(1:n)
#     new_point = rand(dim)

#     ϕ[ix, :] = new_point
# end


function replace_random_point(ϕ::AbstractVector{T}) where T
    n = length(ϕ)
    ix = rand(1:n)
    new_point = rand(T)

    ϕ[ix] = new_point
end


function replace_random_point(ϕ, seed)
    n = length(ϕ)
    m = length(eltype(ϕ))
    ix = rand(1:n)
    new_point = rand(seed, m)

    ϕ[ix] = new_point
end


# funcitons to use with Optim
to_matrix(ϕ::AbstractVector{<:AbstractVector{T}}) where T = reinterpret(reshape, T, ϕ) |> Matrix
to_process(M::AbstractMatrix{T}) where T = reinterpret(reshape, SVector{size(M, 1), T}, M)

neighbor!(M0, M1) = (copyto!(M1, M0); replace_random_point(to_process(M1)))
neighbor_discr!(M0, M1, m) = (copyto!(M1, M0); replace_random_point(to_process(M1), 1:m))
