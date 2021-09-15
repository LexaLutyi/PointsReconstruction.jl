function dDrs(d1, d2)
    n, m = map(min, size(d1), size(d2))
    weights = (m:-1:1) ./ m
    @views vd1 = d1[1:n, 1:m]
    @views vd2 = d2[1:n, 1:m]
    abs2.(vd1 - vd2) * weights ./ sum(weights) .|> sqrt
end


function analyze(rs, ϕs; fs = [normK, normT, normH], isD=true)
    label = ["ϕ$i" for i in 1:length(ϕs)]
    for (ϕ, lbl) in zip(ϕs, label)
        scatter(Tuple.(ϕ); title=lbl, label=nothing, markersize=2) |> display
    end

    for f in fs
        vs = [f(rs, ϕ) for ϕ in ϕs]
        plot(rs, vs; label=permutedims(label), title="$f(r)") |> display
    end

    if isD
        ds = [normD(length(ϕ) - 1, rs, ϕ) for ϕ in ϕs]
        [plot(ds[i], label=nothing, title=label[i]) for i in 1:length(ϕs)] .|> display

        d1 = ds[1]
        d2 = ds[end]

        dd = dDrs(d1, d2)
        plot(dd, title="∑Dk(r)/k") |> display
    end
    return
end


lossK(ϕ, p) = mapreduce((k, k0) -> abs2(k - k0), +, normK(p.rs, ϕ), p.k0) / length(p.rs)
lossH(ϕ, p) = mapreduce((h, h0) -> abs2(h - h0), +, normH(p.rs, ϕ), p.h0) / length(p.rs)

function errD(d1, d2)
    ddrs = PointsReconstruction.dDrs(d1, d2)
    sum(ddrs) / length(ddrs)
end

lossD(ϕ, p) = errD(normD(p.maxk, p.rs, ϕ), p.d0)

function default_loss_params(rs, ϕ; isK=false, isH=false, isD=false, isT=false)
    p = (;rs)
    
    if isK 
        k = (; k0 = normK(rs, ϕ))
        p = merge(p, k)
    end
    
    if isH
        h = (; h0 = normH(rs, ϕ))
        p = merge(p, h)
    end
    
    if isD
        maxk = length(ϕ) - 1
        d = (; maxk, d0 = normD(maxk, rs, ϕ))
        p = merge(p, d)
    end
    
    p
end
