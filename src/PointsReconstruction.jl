module PointsReconstruction

using LinearAlgebra: norm, diagind, I, dot
using Distances: Euclidean, pairwise
# using Plots: plot, scatter, plot!, scatter!
using Statistics: mean, var
using StaticArrays: SVector
using StatsBase: sample, sqL2dist
using FFTW: ifft, ifftshift, fftshift, fftfreq, fft
# using Interpolations: interpolate, Gridded, Linear, extrapolate
using Distributions: IsoNormal, MvNormal, pdf
# using ThreadTools: tmap1

include("generate_processes.jl")
include("summary_characteristics.jl")
# include("plot_functions.jl")
# include("analyze.jl")
include("wavelets.jl")
include("wavelet_sc.jl")

export uniform, true_uniform, net
export circle_cluster, normal_cluster, cross_cluster
export clusters, voronoi

export distance, distances, K, D, T, H

# export plotfdistr, plot_point_process

# export analyze, lossK, lossD, lossH, default_loss_params

export WaveletParams
export lossW, lossW_params

end
