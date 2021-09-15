function plotfdistr(rs, f; kwargs...)
    m = mean(f)
    v = sqrt.(var(f))
    plot(;kwargs...)
    plot!(rs, m, label="μ")
    plot!(rs, m + v, label="μ + σ")
    plot!(rs, m - v, label="μ - σ")
end


plotfdistr(f; kwargs...) = plotfdistr(length(f[1]), f; kwargs...)


plot_point_process(ϕ; label=nothing, xlims=(-0.1, 1.1), ylims=(-0.1, 1.1), kwargs...) = 
    scatter(Tuple.(ϕ); label, xlims, ylims, kwargs...)
