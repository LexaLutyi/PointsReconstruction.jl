distance(x, y; dist=Euclidean()) = dist(x, y)
distances(x::AbstractMatrix; dist=Euclidean()) = pairwise(dist, x, dims=2)


function K(rs, x; dist=Euclidean())
    n = size(x, 2)
    d = distances(x; dist)
    N = n * (n - 1)
    cnt = [count(s -> s < r, d) - n for r in rs]
    cnt / N
end


function D(rs, x, maxk; dist=Euclidean())
    n = size(x, 2)
    d = distances(x; dist)

    nr = length(rs)
    cnt = zeros(Int, nr, maxk)
    
    for (i, r) in enumerate(rs)
        cnt_r = count(s -> s < r, d, dims=1) .- 1
        cnt[i, :] = [count(s -> s ≥ k, cnt_r) for k in 1:maxk]
    end
    cnt / n
end


function T(rs, x; dist=Euclidean())
    n = size(x, 2)
    d = distances(x; dist)

    nr = length(rs)
    cnt = zeros(Int, nr)
    
    for (i, r) in enumerate(rs)
        q = d .< r
        q[diagind(q)] .= false
        cnt[i] = sum(q^2 .* q)
    end

    N = n * (n - 1) * (n - 2)
    cnt / N
end


min_distance(x, y; dist=Euclidean()) = minimum(pairwise(dist, x, reshape(y, length(y), 1), dims=2))


function H(rs, x::AbstractMatrix, step::Real; dist=Euclidean())
    xs = 0.:step:1.
    H(rs, x, xs; dist)
end


function H(rs, x::AbstractMatrix, xs::AbstractRange=rs; dist=Euclidean())
    dim, n = size(x)
    x0 = Iterators.product(fill(xs, dim)...)
    H(rs, x, x0; dist)
end


function H(rs, x::AbstractMatrix, x0; dist=Euclidean())
    n = length(x0)
    mapreduce(y -> rs .>= min_distance(x, collect(y); dist), +, x0) / n
end
