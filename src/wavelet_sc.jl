# 1
wavelet_matrices(ws, J, L; ξ0=maximum(ws) / 2, c=1) =
    [wavelet_matrix(ws, j, θ; ξ0, L, c) for j in 0:J - 1, θ in range(0., length=L, step=2π / L)]


# 2
e_θ(l, L) = (cos(π / 2 + 2π * l / L), sin(π / 2 + 2π * l / L))
τ_θ_j(l, j_, L) = round.(Int, 2^j_ .* e_θ(l, L))

# ! k > 1 doesn't work if not in Γ_H

function default_Γ_H(J, L, K; θtype=Float64)
    Γ_H = NamedTuple{
        (:j, :l, :θ, :k, :j_, :l_, :θ_, :k_, :τ_), 
        Tuple{Int, Int, θtype, Int, Int, 
        Int, θtype, Int, Tuple{Int, Int}}
    }[]
    p = Iterators.product(
        0:L - 1, 
        0:L - 1,
        0:J - 1,
        0:J - 1,
        [false, true],
        0:K - 1,
        0:K - 1
    )

    for (l, l_, j, j_, isshift, k, k_) in p
        l, l_, j, j_, isshift, k, k_
        if l_ > l
            continue
        end
        if (j > j_) || (j < j_ - 2)
            continue
        end
        if k > min(k_, 1)
            continue
        end

        if isshift
            τ_ = τ_θ_j(l, j_, L)
        else
            τ_ = (0, 0)
        end

        θ = 2π * l / L
        θ_ = 2π * l_ / L
        dl = abs(l - l_)

        if j == j_
            if k == 0 && k_ > 1
                continue
            end
            if k == 1
                if k_ != 1
                    continue
                elseif dl > 2
                    continue
                end
            end
        else
            if k == 0 && k_ == 4
                continue
            end
            if k == 1
                if k_ != 2^(j_ - j)
                    continue
                elseif dl > 2
                    continue
                end
            end
        end

        push!(Γ_H, (;j, l, θ, k, j_, l_, θ_, k_, τ_))
    end
    Γ_H
end


function get_ix_from_Γ_H(p, J, L)
    ix1 = (p.j + 1) + J * (p.l) + J * L * (p.k)
    ix2 = (p.j_ + 1) + J * (p.l_) + J * L * (p.k_)
    (p.τ_ == (0, 0), ix1, ix2, p.τ_)
end


function get_ix_subsets_from_Γ_H(Γ_H, J, L, K)
    ix_all = map(p -> get_ix_from_Γ_H(p, J, L), Γ_H)

    perm_ix = findall(s -> s[1] == true, ix_all)
    ix = map(s -> s[2:3], ix_all[perm_ix])

    perm_ix_shift = findall(s -> s[1] == false, ix_all)
    ix_shift = map(s -> s[2:4], ix_all[perm_ix_shift])

    ix_0 = map(i -> (1, i), 1:J * L * K)
    ix, ix_shift, ix_0, perm_ix, perm_ix_shift
end


function M(x::AbstractArray{T}, ws) where T <: AbstractFloat
    n = size(x, 2)
    N = length(ws)
    
    a = x[1, :]
    b = x[2, :]
    aw = a * ws'
    bw = b * ws'
    c = reshape(aw, n, 1, N) .+ reshape(bw, n, N, 1)

    m = sum(t -> cis(-T(2π) * t), c, dims=1)
    reshape(m, N, N)
end


struct WaveletParams{TF, Tw}
    s::TF
    N::Int   
    J::Int
    L::Int
    K::Int
    σ::TF

    ws::Tw
    FN::Function
    
    Ψs::Vector{Matrix{TF}}

    Γ_H::Vector{NamedTuple{
        (:j, :l, :θ, :k, :j_, :l_, :θ_, :k_, :τ_), 
        Tuple{Int, Int, TF, Int, Int, 
        Int, TF, Int, Tuple{Int, Int}}}
        }
    
    ix::Vector{Tuple{Int, Int}}
    ix_shift::Vector{Tuple{Int, Int, Tuple{Int, Int}}}
    ix_0::Vector{Tuple{Int, Int}}
end


function WaveletParams(s, N, J, L, K, σ, Γ_H = default_Γ_H(J, L, K, θtype=typeof(s)))
    dx = 2s / N
    xs = range(-s; length=N, step=dx)
    ws = fftfreq(N, 1 / dx)

    g = MvNormal([0, 0], σ^2 * I)
    gaus = [pdf(g, [x, y]) for x in xs, y in xs]
    Gaus = gaus |> ifftshift |> fft |> real

    FN(μ) = M(μ, ws)
    float_type = typeof(s)
    Ψs = map(Ψ -> float_type.(Ψ .* Gaus), reshape(wavelet_matrices(ws, J, L), :))

    ix, ix_shift, ix_0, perm_ix, perm_ix_shift = get_ix_subsets_from_Γ_H(Γ_H, J, L, K)
    WaveletParams(s, N, J, L, K, σ, ws, FN, Ψs, Γ_H, ix, ix_shift, ix_0)
end


function phase_harmonics(z::T, k::Int) where T
    if k < 0
        ph(conj.(z), -k)
    elseif k == 0
        T(@. abs(z))
    elseif k == 1
        copy(z)
    elseif k == 2
        @. z * z / abs(z)
    elseif k == 3
        @. z ^ 3 / abs2(z)
    else
        if k % 2 == 0
            @. z ^ k / abs(z) ^ (k - 1)
        else
            @. z ^ k / abs2(z) ^ (k ÷ 2)
        end
    end
end


all_phase_harmonics(w_jl, K) = mapreduce(k -> phase_harmonics.(w_jl, k), vcat, 0:K - 1)


function wavelet_phase_harmonics(μ, wp, vs)
    M = wp.FN(μ)
    
    w_jl = map(Ψ -> ifft(M .* Ψ), wp.Ψs)
    w_jlk = map((w, v) -> w .- v, all_phase_harmonics(w_jl, wp.K), vs[2])
    
    m = ifft(M) .- vs[1]
    
    m, w_jlk
end


function v_λ_k_all(μ, wp::WaveletParams{TF}) where TF <: AbstractFloat
    m, w_jlk = wavelet_phase_harmonics(μ, wp, (zero(TF), zeros(TF, wp.J * wp.L * wp.K)))
    mean(m), mean.(w_jlk)
end


cc(x, y::AbstractArray{<:Complex}) = mean(@. x * conj(y))
cc(x, y::AbstractArray{<:Real}) = mean(@. x * y)


cc_subset(w1, w2, ix) = map(ix) do (i, j)
    cc(w1[i], w2[j])
end


cc_subset(w1, w2, ix::Vector{Tuple{Int, Int, Tuple{Int, Int}}}) = map(ix) do (i, j, shift)
    cc(w1[i], circshift(w2[j], shift))
end


function K_all(μ, wp, vs)
    m, w = wavelet_phase_harmonics(μ, wp, vs)

    c1 = cc_subset(w, w, wp.ix)
    c2 = cc_subset(w, w, wp.ix_shift)
    c3 = cc_subset([m], w, wp.ix_0)

    [c1; c2; c3]
end


function K_all(x, y, wp, vs_x, vs_y)
    mx, wx = wavelet_phase_harmonics(x, wp, vs_x)
    my, wy = wavelet_phase_harmonics(y, wp, vs_y)

    # cy1 = cc_subset(wy, wy, wp.ix)
    # cy2 = cc_subset(wy, wy, wp.ix_shift)
    # cy3 = cc_subset([my], wy, wp.ix_0)

    cxy1 = cc_subset(wx, wy, wp.ix)
    cxy2 = cc_subset(wx, wy, wp.ix_shift)
    cxy3 = cc_subset([mx], wy, wp.ix_0)

    # [cy1; cy2; cy3; cxy1; cxy2; cxy3]
    [cxy1; cxy2; cxy3]
end


ww_subset(w1, w2, ix) = map(ix) do (i, j)
    x = w1[i]
    y = w2[j]
    std(@. x * conj(y))
end


ww_subset(w1, w2, ix::Vector{Tuple{Int, Int, Tuple{Int, Int}}}) = map(ix) do (i, j, shift)
    x = w1[i]
    y = circshift(w2[j], shift)
    std(@. x * conj(y))
end


function W_all(μ, wp, vs)
    m, w = wavelet_phase_harmonics(μ, wp, vs)

    c1 = ww_subset(w, w, wp.ix)
    c2 = ww_subset(w, w, wp.ix_shift)
    c3 = ww_subset([m], w, wp.ix_0)

    [c1; c2; c3]
end


function W_all(x, y, wp, vs_x, vs_y)
    mx, wx = wavelet_phase_harmonics(x, wp, vs_x)
    my, wy = wavelet_phase_harmonics(y, wp, vs_y)

    cxy1 = ww_subset(wx, wy, wp.ix)
    cxy2 = ww_subset(wx, wy, wp.ix_shift)
    cxy3 = ww_subset([mx], wy, wp.ix_0)

    [cxy1; cxy2; cxy3]
end


function lossW(x; wp, vs, K0, weights)
    k = K_all(x, wp, vs) .* weights
    sqL2dist(k, K0)
end
lossW(x, p) = lossW(x; p...)


function lossW_params(x, wp, weights=ones(length(wp.ix) + length(wp.ix_shift) + length(wp.ix_0)))
    vs = v_λ_k_all(x, wp)
    K0 = K_all(x, wp, vs) .* weights
    (; wp, vs, K0, weights)
end
