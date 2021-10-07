best_c(L) = (1 / 1.29) * 2^(L - 1) * factorial(L - 1) / sqrt(L * factorial(2 * (L - 1)))


function wavelet_matrices(J, L, ws)
    c = best_c(L)
    ξ0 = maximum(ws) / 2
    [[Ψ(scale(x + im * y, j, 2π * l / L); ξ0, L, c) for x in ws, y in ws] for j in 0:J-1, l in 0:L-1]
end


# function conv_fft(μ, ψ)
#     M = fft(μ)
#     Ψ = fft(ψ)
#     ifft(M .* Ψ)
# end


struct WaveletParams
    # FN: μ -> M
    # M - fft of fN(μ)
    s::Float64
    N::Int
    FN::Function
    # xs::Matrix{Vector{Float64}}
    Ws::Matrix{Vector{Float64}}

    J::Int
    L::Int
    K::Int
    Ψs::Matrix{Matrix{Float64}}

    Γ_H::Vector{NamedTuple{
        (:j, :l, :θ, :k, :j_, :l_, :θ_, :k_, :τ_), 
        Tuple{Int64, Int64, Float64, Int64, Int64, Int64, Float64, Int64, Tuple{Int64, Int64}}}
        }
end


function WaveletParams(s, N, J, L, K, σ = 2s / (N - 1))
    xs = range(-s, s; length=N)
    dx = xs.step.hi
    ws = fftfreq(N, 1 / dx)
    Ws = [[w1, w2] for w1 in ws, w2 in ws]

    g = MvNormal([0, 0], σ^2 * I)
    gaus = [pdf(g, [x, y]) for x in xs, y in xs]
    Gaus = gaus |> ifftshift |> fft

    FN(μ) = M(μ, Ws) .* Gaus

    Ψs = wavelet_matrices(J, L, ws)

    Γ_H = default_Γ_H(J, L, K)
    
    WaveletParams(s, N, FN, Ws, J, L, K, Ψs, Γ_H)
end


f(x, w) = exp(-2π * im * dot(x, w))
M(μ, Ws) = [sum(f(x, w) for x in eachcol(μ)) for w in Ws]


# fN(x, wp::WaveletParams) = [wp.f(x, y) for y in wp.xs]

phase_harmonics(z::Complex, k::Int) = rotate(angle(z) * (k - 1), z)
phase_harmonics(cs::Matrix{<:Complex}, k::Int) = phase_harmonics.(cs, k)
v_λ_k(cs, j, l, k) = mean(phase_harmonics(cs[j + 1, l + 1], k))


function v_λ_k_all(x, wp::WaveletParams)
    MG = wp.FN(x)
    cs = [ifft(MG .* Ψ) for Ψ in wp.Ψs]
    # cs = [conv_fft(ifft(fN(x, wp)), ψ) for ψ in wp.ψs]
    [v_λ_k(cs, j, l, k) for j in 0:wp.J - 1, l in 0:wp.L - 1, k in 0:wp.K]
end


e_θ(l) = (cos(π / 2 + π * l / 4), sin(π / 2 + π * l / 4))
τ_θ_j(l, j_) = round.(Int, 2^j_ .* e_θ(l))


function default_Γ_H(J, L, K)
    Γ_H = NamedTuple{
        (:j, :l, :θ, :k, :j_, :l_, :θ_, :k_, :τ_), 
        Tuple{Int64, Int64, Float64, Int64, Int64, Int64, Float64, Int64, Tuple{Int64, Int64}}
    }[]
    for l in 0:L-1
        for l_ in 0:l
            for j_ in 0:J-1
                for j in max(0, j_ - 2):j_
                    for τ_ in [(0, 0), τ_θ_j(l, j_)]
                        for τ_ in [(0, 0)]
                            for k_ in 0:K
                                for k in 0:min(k_, 1)
                                    θ = 2π * l / L
                                    θ_ = 2π * l_ / L
                                    dθ = abs(θ - θ_)

                                    if j == j_
                                        if k == 0 && k_ > 1
                                            continue
                                        end
                                        if k == 1
                                            if k_ != 1
                                                continue
                                            elseif dθ > 4π / L
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
                                            elseif dθ > 4π / L
                                                continue
                                            end
                                        end

                                    end
                                    push!(Γ_H, (;j, l, θ, k, j_, l_, θ_, k_, τ_))
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    Γ_H
end


function aaa(cs, vs, j, l, k, τ)
    A = cs[j + 1, l + 1]
    v = vs[j + 1, l + 1, k == 4 ? k : k + 1]
    phase_harmonics(circshift(A, τ), k) .- v
end


function K_μ(cs, vs, j, l, k, j_, l_, k_, τ_)
    A = aaa(cs, vs, j, l, k, (0, 0))
    B = aaa(cs, vs, j_, l_, k_, τ_)
    mean(@. A * conj(B))
end


K_μ(cs, vs, p) = K_μ(cs, vs, p.j, p.l, p.k, p.j_, p.l_, p.k_, p.τ_)


function K_all(x, wp::WaveletParams, vs)
    MG = wp.FN(x)
    # cs = [conv_fft(ifft(fN(x, wp)), ψ) for ψ in wp.ψs]
    cs = [ifft(MG .* Ψ) for Ψ in wp.Ψs]
    K_μ.(Ref(cs), Ref(vs), wp.Γ_H)
end