# 1
wavelet_matrices(ws, J, L; ξ0=maximum(ws) / 2, c=1) =
    [wavelet_matrix(ws, j, θ; ξ0, L, c) for j in 0:J - 1, θ in range(0., length=L, step=2π / L)]


# 2
e_θ(l, L) = (cos(π / 2 + 2π * l / L), sin(π / 2 + 2π * l / L))
τ_θ_j(l, j_, L) = round.(Int, 2^j_ .* e_θ(l, L))

# ! k > 1 doesn't work if not in Γ_H

function default_Γ_H(J, L, K)
    Γ_H = NamedTuple{
        (:j, :l, :θ, :k, :j_, :l_, :θ_, :k_, :τ_), 
        Tuple{Int64, Int64, Float64, Int64, Int64, 
        Int64, Float64, Int64, Tuple{Int64, Int64}}
    }[]
    p = Iterators.product(
        0:L - 1, 
        0:L - 1,
        0:J - 1,
        0:J - 1,
        [false, true],
        0:K,
        0:K
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

# 3
"Fourier transform of δ(x - x0)"
# Δ(w, x0) = exp(-2π * im * dot(x0, w))
"Fourier transform of point process"
# M(Ws, μ::AbstractMatrix) = [sum(Δ(w, x) for x in eachcol(μ)) for w in Ws]

function Δ(x0, ws)
    a = x0[1] .* ws
    b = x0[2] .* ws
    -2pi * (a' .+ b) .|> cis
end


M(x, ws) = sum(Δ(xi, ws) for xi in eachcol(x))


# main
struct WaveletParams{Tw}
    s::Float64
    N::Int   
    J::Int
    L::Int
    K::Int
    σ::Float64

    ws::Tw
    Ws::Matrix{Vector{Float64}}
    FN::Function

    
    Ψs::Matrix{Matrix{Float64}}

    Γ_H::Vector{NamedTuple{
        (:j, :l, :θ, :k, :j_, :l_, :θ_, :k_, :τ_), 
        Tuple{Int64, Int64, Float64, Int64, Int64, 
        Int64, Float64, Int64, Tuple{Int64, Int64}}}
        }
    jlk::Array{Tuple{Int64, Int64, Int64}, 3}
end


function WaveletParams(s, N, J, L, K, σ, Γ_H = default_Γ_H(J, L, K))
    dx = 2s / N
    xs = range(-s; length=N, step=dx)
    ws = fftfreq(N, 1 / dx)
    Ws = [[w1, w2] for w1 in ws, w2 in ws]

    g = MvNormal([0, 0], σ^2 * I)
    gaus = [pdf(g, [x, y]) for x in xs, y in xs]
    Gaus = gaus |> ifftshift |> fft |> real

    # FN(μ) = M(Ws, μ)
    FN(μ) = M(μ, ws)

    Ψs = [Ψ .* Gaus for Ψ in wavelet_matrices(ws, J, L)]

    jlk = collect(Iterators.product(1:J, 1:L, 1:K + 1))

    WaveletParams(s, N, J, L, K, σ, ws, Ws, FN, Ψs, Γ_H, jlk)
end


function phase_harmonics(z::T, k::Int) where T <: Complex
    if k < 0
        phase_harmonics(conj(z), -k)
    elseif k == 0
        abs(z) |> T
    elseif k == 1
        z
    elseif k == 2
        z * z / abs(z)
    elseif k == 3
        z ^ 3 / abs2(z)
    else
        if k % 2 == 0
            z ^ k / abs(z) ^ (k - 1)
        else
            z ^ k / abs2(z) ^ (k ÷ 2)
        end
    end
end
phase_harmonics(Z::AbstractArray{<:Complex}, k::Int) = phase_harmonics.(Z, k)


function wavelet_phase_harmonics(μ, wp, vs)
    M = wp.FN(μ)
    
    w_jl = map(Ψ -> ifft(M .* Ψ), wp.Ψs)
    w_jlk = map(.-, mapreduce(k -> phase_harmonics.(w_jl, k), vcat, 0:wp.K), vs[2])
    
    m = ifft(M) .- vs[1]
    
    m, w_jlk
end


function v_λ_k_all(μ, wp::WaveletParams)
    m, w_jlk = wavelet_phase_harmonics(μ, wp, (0., zeros(wp.J, wp.L, wp.K + 1)))
    mean(m), mean.(w_jlk)
end


function K_μ(w_jlk; j, l, k, j_, l_, k_, τ_)
    A = w_jlk[j + 1, l + 1, k + 1]
    B = circshift(w_jlk[j_ + 1, l_ + 1, k_ + 1], τ_)
    mean(@. A * conj(B))
end


function K_μ_0(m, w_jlk)
    map(c -> mean(m .* conj.(c)), w_jlk)
end


K_μ(w_jlk, p) = K_μ(w_jlk; p.j, p.l, p.k, p.j_, p.l_, p.k_, p.τ_)


function K_all(μ, wp::WaveletParams, vs)
    m, w_jlk = wavelet_phase_harmonics(μ, wp, vs)
    K_main = map(p -> K_μ(w_jlk, p), wp.Γ_H)
    K_0 = K_μ_0(m, w_jlk)[:]
    [K_main; K_0]
end


function lossW(x; wp, vs, K0, weights, index)
    k = (K_all(x, wp, vs) .* weights)[index]
    k0 = (K0 .* weights)[index]
    sqL2dist(k, k0)
end
lossW(x, p) = lossW(x; p...)


function lossW_params(x, wp, weights=ones(length(wp.Γ_H) + length(wp.jlk)), index=:)
    vs = v_λ_k_all(x, wp)
    K0 = K_all(x, wp, vs)
    (; wp, vs, K0, weights, index)
end
