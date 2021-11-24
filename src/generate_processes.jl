uniform(dim, n) = rand(dim, n)


true_uniform(n, m) = reduce(hcat, collect.(Iterators.product((1:n) / n .- 1 / 2n, (1:m) / m .- 1 / 2m)))


function net(nline, npoint)
    dx = 1 / npoint
    xs = range(dx / 2, length = npoint, step = dx)
    dy = 1 / nline
    ys = range(dy / 2, length = nline, step = dy)

    a = [[x, y] for x in xs, y in ys]
    b = [[y, x] for x in xs, y in ys]
    reduce(hcat, [a[:]; b[:]])
end


function circle_cluster(n)
    ws = range(0., length=n, step=2π / n)
    [f(w) for f in (cos, sin), w in ws]
end


normal_cluster(dim, n) = randn(dim, n)


function cross_cluster(w)
    T = [
        cos(w) sin(w)
        -sin(w) cos(w)
    ]
    x = [
        -2 -1 0 1 2 0 0 0 0
        0 0 0 0 0 -2 -1 1 2
    ]
    mapreduce(x -> T * x, hcat, eachcol(x))
end


clusters(centers, cluster) = mod.(reduce(hcat, [cluster() .+ c for c in eachcol(centers)]), 1.)


function voronoi_metric(x, support)
    a = colwise(PeriodicEuclidean(ones(2)), support, x)
    b = partialsort!(a, 1:2)
    abs(b[2] - b[1])
end


function circle_metric(x, support, r)
    a = abs.(colwise(PeriodicEuclidean(ones(2)), support, x) .- r)
    partialsort!(a, 1)
end


function surface_points(ndims, npoints, metric)
    all_points = rand(ndims, 1000 * npoints)
    
    a = [metric(x) for x in eachcol(all_points)]
    
    ix = sortperm(a)
    points = all_points[:, ix[1:npoints]]
    
    points
end


"""
    voronoi(n, nsupport)

generate `n` points on the surface of Voronoi diagram
"""
function voronoi(npoints, nsupport)
    ndims = 2
    support = rand(ndims, nsupport)
    surface_points(ndims, npoints, x -> voronoi_metric(x, support))
end


"""
    circles(n, nsupport, radius)

generate `n` points on the surface of circles
"""
function circles(npoints, nsupport, radius)
    ndims = 2
    support = rand(ndims, nsupport)
    surface_points(ndims, npoints, x -> circle_metric(x, support, radius))
end
