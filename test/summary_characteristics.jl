x = [
    0.0 0.3 0.6
    0.0 0.0 0.0
]
rs = 0.5:0.5:1.

dist = PeriodicEuclidean(ones(2))

@test K(rs, x) ≈ [2 / 3, 1.]
@test K(rs, x; dist) ≈ ones(2)


@test D(rs, x, 2) ≈ [1. 1/3; 1. 1.]
@test D(rs, x, 2; dist) ≈ ones(2, 2)


@test T(rs, x) ≈ [0., 1.]
@test T(rs, x; dist) ≈ [1., 1.]


@test H(rs, x, 0.5) ≈ [4 / 9, 7 / 9]
@test H(rs, x, 0.5; dist) ≈ [8 / 9, 1.]
