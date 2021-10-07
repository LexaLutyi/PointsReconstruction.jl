g(t) = -1 < t < 1 ? exp(-abs2(t) / (1 - abs2(t))) : 0.


Ψ(ω; ξ0, L, c) = -π / 2 < angle(ω) < π / 2 ? c * g((abs(ω) - ξ0) / ξ0) * cos(angle(ω))^(L - 1) : 0.

# fs(C, N) = C * N / (N - 1)


scale(x, j, θ) = 2^j * rotate(θ, x)


function get_ψ(j=0, θ=0.;c=1., L=8, C::AbstractFloat=1., ξ0=π / C, n=100)
    dx = C / n
    N = 2n + 1

    ωs = fftfreq(N, 1 / dx) |> fftshift

    Ψ_centered = [Ψ(scale(x + im * y, j, θ); ξ0, L, c) for y in ωs, x in ωs]

    Ψ_zero = ifftshift(Ψ_centered)
    ψ_zero = ifft(Ψ_zero)
    
    xs = -C:dx:C
    ψ_centered = fftshift(ψ_zero) / dx^2

    ψ_centered, xs
end


struct Wavelet{T}
    ext::T
    ξ::SVector{2, Float64}
    c::Float64
    L::Int
    C::Float64
    j::Int
    θ::Float64
end


function Wavelet(j::Int=0, θ::Float64=0.; c=1., L=8, C=1., ξ0=π / C, n=100)
    ψ, xs = get_ψ(j, θ; n, c, ξ0, L, C)
    itp = interpolate((xs, xs), ψ, Gridded(Linear()))
    ext = extrapolate(itp, 0.)
    Wavelet(ext, SVector(ξ0, 0.), c, L, C, j, θ)
end


(w::Wavelet)(x...) = w.ext(x...)
(w::Wavelet)(x) = w.ext(x...)

rotate(θ, ω::Complex) = ω * exp(im * θ)

rotate(θ, x) = [
    cos(θ) * x[1] - sin(θ) * x[2]
    sin(θ) * x[1] + cos(θ) * x[2]
]


ψ_λ(x, j::Int, θ; w::Wavelet) = 2. ^ (-2j) * w(2. ^ (-j) * rotate(θ, x))
