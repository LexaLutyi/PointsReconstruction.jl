module PointsReconstruction

using LinearAlgebra: norm, diagind
using Plots: plot, scatter, plot!, scatter!
using Statistics: mean, var
using StaticArrays: SVector
using StatsBase: sample

include("gen_point_process.jl")
include("summary_characteristics.jl")
include("plot_functions.jl")
include("analyze.jl")

export ϕ_uniform, ϕ_cluster, replace_random_point
export distance, distances, K, D, T, H
export normK, normD, normT, normH
export diffK
export plotfdistr, plot_point_process
export to_matrix, to_process, neighbor!, neighbor_discr!
export analyze, lossK, lossD, lossH, default_loss_params

end
