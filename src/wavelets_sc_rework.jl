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
Δ(w, x0) = exp(-2π * im * dot(x0, w))
"Fourier transform of point process"
M(Ws, μ::AbstractMatrix) = [sum(Δ(w, x) for x in eachcol(μ)) for w in Ws]

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


function WaveletParams(s, N, J, L, K, σ)
    dx = 2s / N
    xs = range(-s; length=N, step=dx)
    ws = fftfreq(N, 1 / dx)
    Ws = [[w1, w2] for w1 in ws, w2 in ws]

    g = MvNormal([0, 0], σ^2 * I)
    gaus = [pdf(g, [x, y]) for x in xs, y in xs]
    Gaus = gaus |> ifftshift |> fft |> real

    FN(μ) = M(Ws, μ)

    Ψs = [Ψ .* Gaus for Ψ in wavelet_matrices(ws, J, L)]

    Γ_H = default_Γ_H(J, L, K)
    jlk = collect(Iterators.product(1:J, 1:L, 1:K + 1))

    WaveletParams(s, N, J, L, K, σ, ws, Ws, FN, Ψs, Γ_H, jlk)
end


phase_harmonics(z::Complex, k::Int) = abs(z) * exp(im * angle(z) * k)


function wavelet_phase_harmonics(μ, wp, vs)
    M = wp.FN(μ)
    cs = [ifft(M .* Ψ) for Ψ in wp.Ψs]
    m = ifft(M) .- vs[1]
    csk = [phase_harmonics.(cs[j, l], k - 1) .- vs[2][j, l, k] for (j, l, k) in wp.jlk]
    m, csk
end


function v_λ_k_all(μ, wp::WaveletParams)
    m, csk = wavelet_phase_harmonics(μ, wp, (0., zeros(wp.J, wp.L, wp.K + 1)))
    mean(m), mean.(csk)
end


function K_μ(csk; j, l, k, j_, l_, k_, τ_)
    A = csk[j + 1, l + 1, k + 1]
    B = circshift(csk[j_ + 1, l_ + 1, k_ + 1], τ_)
    mean(@. A * conj(B))
end


function K_μ_0(m, csk)
    [mean(m .* conj.(c)) for c in csk]
end


K_μ(csk, p) = K_μ(csk; p.j, p.l, p.k, p.j_, p.l_, p.k_, p.τ_)


function K_all(μ, wp::WaveletParams, vs)
    m, csk = wavelet_phase_harmonics(μ, wp, vs)
    K_main = [K_μ(csk, p) for p in wp.Γ_H]
    K_0 = K_μ_0(m, csk)[:]
    [K_main; K_0]
end


lossW(x; wp, vs, K0) = sum(abs2, K_all(x, wp, vs) - K0)
lossW(x, p) = lossW(x; p...)


function lossW_params(x, wp)
    vs = v_λ_k_all(x, wp)
    K0 = K_all(x, wp, vs)
    (; wp, vs, K0)
end


function lossW_params(x; s, N, J, L, K, σ)
    wp = WaveletParams(s, N, J, L, K, σ)
    lossW_params(x, wp)
end
