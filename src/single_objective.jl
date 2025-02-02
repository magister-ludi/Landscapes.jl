
export ackley, ackley_n2, adjiman, amgm, bartels_conn, beale, bent_cigar, bird
export bohachevsky_n1, bohachevsky_n2, bohachevsky_n3, booth, branin, brent, brown, bukin_n6
export camel_hump_3, camel_hump_6, carrom_table, chichinadze, chung_reynolds, colville
export cosine_mixture, cross_in_tray, csendes, cube, damavandi, dekker_aarts, dixon_price
export drop_wave, easom, eggholder, exponential, freudenstein_roth, goldstein_price
export griewank, himmelblau, holder_table, hosaki, keane, leon, levy_n13, matyas
export mccormick, michalewicz, parsopoulos, pen_holder, plateau, qing, quartic, rastrigin
export rosenbrock, rotated_hyper_ellipsoid, salomon, schaffer_n2, schaffer_n4, schwefel
export sphere, step_function, styblinski_tang, sum_of_different_powers, sum_of_squares
export trid, tripod, wolfe, zakharov

VecOrTuple = Union{AbstractVector{<:Real}, Tuple{Vararg{Real}}}

"""
    ackley(x)

Ackley function: https://en.wikipedia.org/wiki/Ackley_function

  - Dimensions: 2
  - Global minimum: f(0, 0) = 0
  - Bounds: -5 ≤ x₁, x₂ ≤ 5
"""
function ackley(xy::VecOrTuple)
    x, y = xy
    return -20 * exp(-0.2 * sqrt(0.5 * (x * x + y * y))) -
           exp(0.5 * (cos(2.0 * π * x) + cos(2 * π * y))) +
           ℯ +
           20
end

ackley(x::Real, y::Real) = ackley((x, y))

"""
    ackley_n2(x)

Ackley N.2 function: https://www.indusmic.com/post/ackleyn-2function

  - Dimensions: 2
  - Global minimum: f(0, 0) = -200
  - Bounds: -32 ≤ x₁, x₂ ≤ 32

### References

  - D. H. Ackley, "A Connectionist Machine for Genetic Hill-Climbing", Kluwer, 1987.
"""
ackley_n2(xy::VecOrTuple) = -200 * exp(-0.2 * sqrt(xy[1]^2 + xy[2]^2))

"""
    adjiman(x)

Adjiman function

  - Dimensions: 2

If the bounds (-X ≤ x₁, x₂ ≤ X) in the two dimensions are "similar" (see link
below), the minimum is at x₁ == X, the maximum is at x₁ == -X, e.g.

  - Global minimum: f(2.0, 0.10578) = -2.02180678
  - Bounds: -2 ≤ x₁, x₂ ≤ 2

### References

  - C. S. Adjiman, S. Sallwig, C. A. Flouda, A. Neumaier, "A Global Optimization
    Method, aBB for General Twice-Differentiable NLPs-1, Theoretical Advances, ”
    Computers Chemical Engineering, vol. 22, no. 9, pp. 1137-1158, 1998.
  - https://al-roomi.org/benchmarks/unconstrained/2-dimensions/113-adjiman-s-function
"""
function adjiman(xy::VecOrTuple)
    x, y = xy
    return cos(x) * sin(y) - (x / (y^2.0 + 1.0))
end

"""
    amgm(x)

Arithmetic Mean-Geometric Mean (Mishra's function No.11):
https://al-roomi.org/benchmarks/unconstrained/n-dimensions/182-mishra-s-function-no-11-or-amgm-function

  - Dimension: ≥2
  - Global minimum(*):  f(x₁, x₂, x₃...) = 0 if x₁ == x₂ == x₃ == ...
  - Bounds: 0 ≤ xᵢ ≤ 10 for i=2, ..., n

(*) There is an infinite number of global minima when all values of x are
non-negative and equal.

    f(1, 1) = 0
    f(6, 6, 6) = 0
    f(0.5, 0.5, 0.5, 0.5) = 0
    etc...
"""
function amgm(x::VecOrTuple)
    d = length(x)
    if d < 2
        error("argument requires more than 1 element")
    end
    a = sum(abs(v) for v in x) / d
    b = abs(prod(x))^(1 / d)
    return (a - b)^2
end

"""
    bartels_conn(x)

Bartels Conn function: https://www.indusmic.com/post/bartels-conn-function

  - Dimensions: 2
  - Global minimum: f(0, 0) = 1
  - Bounds: -500 ≤ x₁, x₂ ≤ 500

### References

Momin Jamil and Xin-She Yang, A literature survey of benchmark functions for
global optimization problems, Int. Journal of Mathematical Modelling and
Numerical Optimisation, Vol. 4, Issue. 2 (2013).
"""
function bartels_conn(xy::VecOrTuple)
    x, y = xy
    return abs(x^2 + y^2 + x * y) + abs(sin(x)) + abs(cos(y))
end

"""
    beale(x)

Beale function: https://www.sfu.ca/~ssurjano/beale.html

  - Dimensions: 2
  - Global minimum: f(3, 0.5) = 0
  - Bounds: -4.5 ≤ x₁, x₂ ≤ 4.5
"""
function beale(xy::VecOrTuple)
    x, y = xy
    return (1.500 - x + x * y)^2 + (2.250 - x + x * y^2)^2 + (2.625 - x + x * y^3)^2
end

"""
    bent_cigar(x::VecOrTuple)

Bent Cigar function

  - Dimensions: 1, 2, 3,...
  - Global minimum: f(0, ..., 0) = 0
  - Bounds: -100 ≤ xᵢ ≤ 100 for i=1, ..., n

### References

https://al-roomi.org/benchmarks/unconstrained/n-dimensions/164-bent-cigar-function
"""
bent_cigar(x::VecOrTuple) = x[1] * x[1] + 1e6 * sum([v * v for v in @view(x[2:end])])

"""
    bird(xy)

Bird function

  - Dimensions: 2
  - Global minimum(s): f(4.70105, 3.15294) = f(-1.58214, -3.13024) = -106.76453
  - Bounds: -2π ≤ x₁, x₂ ≤ 2π

### References

S. K. Mishra, "Global Optimization By Differential Evolution and Particle
Swarm Methods: Evaluation On Some Benchmark Functions", Munich Research
Papers in Economics: http://mpra.ub.uni-muenchen.de/1005/
"""
function bird(xy::VecOrTuple)
    x, y = xy
    return sin(x) * exp((1 - cos(y))^2) + cos(y) * exp((1 - sin(x))^2) + (x - y)^2
end

"""
    bohachevsky_n1(xy)

Bohachevsky function N.1

  - Dimensions: 2
  - Global minimim: f(0, 0) = 0
  - Bounds: -100 ≤ x₁, x₂ ≤ 100

### References

[1] I. O. Bohachevsky, M. E. Johnson, M. L. Stein, "General Simulated Annealing
for Function Optimization", Technometrics, vol. 28, no. 3, pp. 209-217, 1986.
[2] http://www.sfu.ca/~ssurjano/boha.html
"""
function bohachevsky_n1(xy::VecOrTuple)
    x, y = xy
    return x^2 + 2 * y^2 - 0.3 * cos(3.0 * π * x) - 0.4 * cos(4 * π * y) + 0.7
end

"""
    bohachevsky_n2(xy)

Bohachevsky function N.2

  - Dimensions: 2
  - Global minimim: f(0, 0) = 0
  - Bounds: -100 ≤ x₁, x₂ ≤ 100

### References

[1] I. O. Bohachevsky, M. E. Johnson, M. L. Stein, "General Simulated Annealing
for Function Optimization", Technometrics, vol. 28, no. 3, pp. 209-217, 1986.
[2] http://www.sfu.ca/~ssurjano/boha.html
"""
function bohachevsky_n2(xy::VecOrTuple)
    x, y = xy
    return x^2 + 2 * y^2 - 0.3 * cos(3 * π * x) * cos(4 * π * y) + 0.3
end

"""
    bohachevsky_n3(xy)

Bohachevsky function N.3

  - Dimensions: 2
  - Global minimim: f(0, 0) = 0
  - Bounds: -100 ≤ x₁, x₂ ≤ 100

### References

[1] I. O. Bohachevsky, M. E. Johnson, M. L. Stein, "General Simulated Annealing
for Function Optimization", Technometrics, vol. 28, no. 3, pp. 209-217, 1986.
[2] http://www.sfu.ca/~ssurjano/boha.html
"""
function bohachevsky_n3(xy::VecOrTuple)
    x, y = xy
    return x^2 + 2 * y^2 - 0.3 * cos(3 * π * x + 4 * π * y) + 0.3
end

"""
    booth(x)

Booth function, https://www.sfu.ca/~ssurjano/booth.html

  - Dimensions: 2
  - Global minimum: f(1, 3) = 0
  - Bounds: 10 ≤ x₁, x₂ ≤ 10
"""
function booth(xy::VecOrTuple)
    x, y = xy
    return (x + 2 * y - 7)^2 + (2 * x + y - 5)^2
end

"""
    branin(x)

Branin function, https://www.sfu.ca/~ssurjano/branin.html

  - Dimensions: 2
  - Global minimum(s): f(-π, 12.275) = f( π,  2.275) = f(3π,  2.475) = 0.397887
  - Bounds: -5 ≤ x₁ ≤ 10, 0 ≤ x₂ ≤ 15
"""
function branin(xy::VecOrTuple)
    a = 1
    b = 5.1 / (4 * π^2)
    c = 5 / π
    r = 6
    s = 10
    t = 1 / (8 * π)
    x, y = xy
    return a * (y - b * x^2 + c * x - r)^2 + s * (1 - t) * cos(x) + s
end

"""
    brent(x)

Brent's function

  - Dimensions: 2
  - Global minimum: f(-10, -10) = 0
  - Bounds: -10 ≤ x₁, x₂ ≤ 2

### References

  - S. K. Mishra, "Global Optimization By Differential Evolution and Particle
    Swarm Methods: Evaluation On Some Benchmark Functions", Munich Research
    Papers in Economics: http://mpra.ub.uni-muenchen.de/1005/
  - https://al-roomi.org/benchmarks/unconstrained/2-dimensions/112-brent-s-function
"""
function brent(xy::VecOrTuple)
    x, y = xy
    return (x + 10.0)^2.0 + (y + 10.0)^2.0 + exp(-(x^2.0) - y^2.0)
end

"""
    brown(x)

Brown's function

  - Dimensions: 1, 2  3,...
  - Global minimum: f(0, 0,..., 0) = 0
  - Bounds: -1 ≤ xᵢ ≤ 4 for i=1, ..., n

### References

  - O. Begambre, J. E. Laier, "A hybrid Particle Swarm Optimization - Simplex
    Algorithm (PSOS) for Structural Damage Identification", Journal of Advances
    in Engineering Software, vol. 40, no. 9, pp. 883-891, 2009.
  - https://al-roomi.org/benchmarks/unconstrained/n-dimensions/241-brown-s-function
  - https://www.indusmic.com/post/brown-function
"""
function brown(x::VecOrTuple)
    l = length(x)
    l == 1 && return 0.0
    return sum(((x[i]^2)^(x[i + 1]^2 + 1.0)) + ((x[i + 1]^2)^(x[i]^2 + 1.0)) for i = 1:(l - 1))
end

"""
    bukin_n6(x)

Bukin function N.6: https://www.sfu.ca/~ssurjano/bukin6.html

  - Dimensions: 2
  - Global minimum: f(-10, 1) = 0
  - Bounds: -15 ≤ x₁ ≤ -5, -3 ≤ x₂ ≤ 3
"""
function bukin_n6(xy::VecOrTuple)
    x, y = xy
    return 100 * sqrt(abs(y - 0.01 * x^2)) + 0.01 * abs(x + 10)
end

"""
    camel_hump_3(x)

Three-Hump Camel function: https://www.sfu.ca/~ssurjano/camel3.html

  - Dimensions: 2
  - Global minimum: f(0, 0) = 0
  - Bounds: -5 ≤ x₁, x₂ ≤ 5
"""
function camel_hump_3(xy::VecOrTuple)
    x, y = xy
    return 2.0 * x^2 - 1.05 * x^4 + (x^6 / 6.0) + x * y + y^2
end

"""
    camel_hump_6(x)

Six-hump camel function: https://www.sfu.ca/~ssurjano/camel6.html

The six-hump camel function is usually evaluated on the rectangle bounded
by: -3 ≤ x₁ ≤ 3, -2 ≤ x₂ ≤ 2

  - Dimensions: 2
  - Bounds: -3 ≤ x₁ ≤ 3, -2 ≤ x₂ ≤ 2
  - Global minimum(s):  f( 0.0898, -0.7126) = f(-0.0898, 0.7126) = -1.0316

### References

Molga, M., & Smutnicki, C. Test functions for optimization needs (2005)
"""
function camel_hump_6(xy::VecOrTuple)
    x, y = xy
    a = (4 - 2.1 * x^2 + x^4 / 3.0) * x^2
    b = x * y
    c = (-4 + 4 * y^2) * y^2
    return a + b + c
end

"""
    carrom_table(x)

Carrom Table function

  - Dimensions: 2
  - Bounds -10 ≤ x₁, x₂ ≤ 10
  - Global minima: f(±9.646157, ±9.646157) = -24.1568

### References

[1]  S. K. Mishra, "Global Optimization By Differential Evolution and
Particle Swarm Methods: Evaluation On Some Benchmark Functions", Munich
Research Papers in Economics:  http://mpra.ub.uni-muenchen.de/1005/
[2]  https://al-roomi.org/benchmarks/unconstrained/2-dimensions/32-carrom-table-function
"""
function carrom_table(xy::VecOrTuple)
    x, y = xy
    return (exp(2 * abs(1 - (sqrt(x^2 + y^2) / π))) * cos(x)^2 * cos(y)^2) / -30.0
end

"""
    chichinadze(x)

Chichinadze function: https://al-roomi.org/benchmarks/unconstrained/2-dimensions/42-chichinadze-s-function

  - Dimensions: 2
  - Bounds: -30 ≤ x₁, x₂ ≤ 30
  - Global minimum: f(6.189866586965680, 0.5) = -42.9443870
"""
function chichinadze(xy::VecOrTuple)
    x, y = xy
    return x^2 - 12 * x + 11 + 10 * cos(π * x / 2) + 8 * sin(5 * π * x / 2) -
           1.0 / sqrt(5) * exp(-((y - 0.5)^2) / 2)
end

"""
    chung_reynolds(x)

Chung Reynolds function: https://al-roomi.org/benchmarks/unconstrained/n-dimensions/165-chung-reynolds-function

  - Dimensions: 1, 2, 3,...
  - Global minimum: f(0, 0,..., 0) = 0
  - Bounds: -100 ≤ xᵢ ≤ 100 for i=1, ..., n

### References

C. J. Chung, R. G. Reynolds, "CAEP: An Evolution-Based Tool for Real-Valued
Function Optimization Using Cultural Algorithms", International Journal on
Artificial Intelligence Tool, vol. 7, no. 3, pp. 239-291, 1998.
"""
chung_reynolds(x::VecOrTuple) = sphere(x)^2

"""
Colville function

  - Dimensions: 4
  - Bounds: The Colville function is a 4-dimensional function usually evaluated
    on the hypercube defined by -10 ≤ x₁, x₂, x₃, x₄ ≤ 10
  - Global minimum: f(1, 1, 1, 1) = 0

### References

  - A.-R. Hedar, "Global Optimization Test Problems”
  - https://www.sfu.ca/~ssurjano/colville.html
"""
function colville(xy::VecOrTuple)
    x1, x2, x3, x4 = xy
    a = 100 * (x1^2 - x2)^2
    b = (x1 - 1)^2
    c = (x3 - 1)^2
    d = 90 * (x3^2 - x4)^2
    e = 10.1 * ((x2 - 1)^2 + (x4 - 1)^2)
    f = 19.8 * (x2 - 1) * (x4 - 1)
    return a + b + c + d + e + f
end

"""
    cosine_mixture(x)

Cosine mixture function

  - Dimensions: 1, 2, 3,...
  - Global minimum: f(x₁, x₂,..., xᵢ) = -0.1n for i=1, 2,..., n
  - Bounds: -1 ≤ xᵢ ≤ 1 i = 1,..., n

### References

  - M. M. Ali, C. Khompatraporn, Z. B. Zabinsky, "A Numerical Evaluation of
    Several Stochastic Algorithms on Selected Continuous Global Optimization
    Test Problems", Journal of Global Optimization, vol. 31, pp. 635-672, 2005.
  - https://al-roomi.org/benchmarks/unconstrained/n-dimensions/166-cosine-mixture-function
"""
cosine_mixture(x::VecOrTuple) = -0.1 * sum([cos(5 * π * v) for v in x]) + sum([v^2 for v in x])

"""
Cross-in-tray function: https://www.sfu.ca/~ssurjano/crossit.html

  - Dimensions: 2
  - Bounds: -10 ≤ x₁, x₂ ≤ 10
  - Global minimum: f(±1.34941, ±1.34941) = -2.06261
"""
function cross_in_tray(xy::VecOrTuple)
    x, y = xy
    return -0.0001 * (abs(sin(x) * sin(y) * exp(abs(100 - (sqrt(x^2 + y^2) / π)))) + 1)^0.1
end

"""
    csendes(x)

Csendes function

  - Dimensions: 1, 2,..., n
  - Bounds: -1 ≤ xᵢ ≤ 1 for i=1,..., n
  - Global minimum f(0, 0,..., 0) = 0

### References

  - T. Csendes, D. Ratz, "Subdivision Direction Selection in Interval Methods
    for Global Optimization", SIAM Journal on Numerical Analysis, vol. 34,
    no. 3, pp. 922-938.
  - https://al-roomi.org/benchmarks/unconstrained/n-dimensions/167-ex3-csendes-or-infinity-function
"""
csendes(x::VecOrTuple) = return sum(v^6 * (2 + sin(safe_division(1, v))) for v in x)

"""
    cube(x)

Cube function: https://al-roomi.org/benchmarks/unconstrained/2-dimensions/119-cube-function

  - Dimensions: 2
  - Global minimum: f(1, 1) = 0
  - Bounds: -10 ≤ x₁, x₂ ≤ 10
"""
function cube(xy::VecOrTuple)
    x, y = xy
    return 100 * (y - x^3)^2 + (1 - x)^2
end

"""
    damavandi(x)

Damavandi's function

  - Dimensions: 2
  - Bounds: 0 ≤ x₁, x₂ ≤ 14
  - Global minimum: f(2, 2) = 0

### References

  - N. Damavandi, S. Safavi-Naeini, "A Hybrid Evolutionary Programming Method
    for Circuit Optimization", IEEE Transaction on Circuit and Systems I,
    vol. 52, no. 5, pp.902-910, 2005.
  - https://al-roomi.org/benchmarks/unconstrained/2-dimensions/120-damavandi-s-function
"""
function damavandi(xy::VecOrTuple)
    x, y = xy
    # division by zero causes errors...
    if x == 2 && y == 2
        return 0.0
    end
    n = sin(π * (x - 2)) * sin(π * (y - 2))
    d = π^2 * (x - 2) * (y - 2)
    a = 1.0 - abs(n / d)^5
    b = 2.0 + (x - 7)^2 + 2 * (y - 7)^2
    return a * b
end

"""
    dekker_aarts(x)

Dekker-Aarts' function

  - Dimensions: 2
  - Bounds: -20 ≤ x₁, x₂ ≤ 20'
  - Global minimum: f(0, ±15) = -24771.093749

### References

  - M. M. Ali, C. Khompatraporn, Z. B. Zabinsky, "A Numerical Evaluation of
    Several Stochastic Algorithms on Selected Continuous Global Optimization
    Test Problems", Journal of Global Optimization, vol. 31, pp. 635-672, 2005.
  - https://al-roomi.org/benchmarks/unconstrained/2-dimensions/53-dekker-aarts-function
"""
function dekker_aarts(xy::VecOrTuple)
    x, y = xy
    return 10^5 * x^2 + y^2 - (x^2 + y^2)^2 + 10^(-5) * (x^2 + y^2)^4
end

"""
Dixon-Price function

  - Dimensions: 1, 2, 3,...
  - Bounds: x_i in [-10, 10] for i=1, ..., n
  - Global minimum: f(x₁, x₂,...) = 0 at xᵢ = 2^-((2^i-2)/(2^i))

### References

  - L. C. W. Dixon, R. C. Price, "The Truncated Newton Method for Sparse
    Unconstrained Optimisation Using Automatic Differentiation", Journal of
    Optimization Theory and Applications, vol. 60, no. 2, pp. 261-275, 1989.
  - https://www.sfu.ca/~ssurjano/dixonpr.html
"""
function dixon_price(x::VecOrTuple)
    l = length(x)
    f = (x[1] - 1.0)^2.0
    if l > 1
        f += sum(i * (2.0 * x[i]^2.0 - x[i - 1])^2.0 for i = 2:l)
    end
    return f
end

"""
    drop_wave(x)

Drop-Wave function: https://www.sfu.ca/~ssurjano/drop.html

  - Dimensions: 2
  - Bounds: -5.12 ≤ x₁, x₂ ≤ 5.12
  - Global minimum: f(0, 0) = -1
"""
function drop_wave(xy::VecOrTuple)
    x, y = xy
    return -(1 + cos(12 * sqrt(x^2 + y^2))) / (0.5 * (x^2 + y^2) + 2)
end

"""
    easom(x)

Easom function: https://www.sfu.ca/~ssurjano/easom.html

  - Dimensions: 2
  - Bounds: -100 ≤ x₁, x₂ ≤ 100
  - Global minimum: f(π, π) = -1
"""
function easom(xy::VecOrTuple)
    x, y = xy
    return -cos(x) * cos(y) * exp(-(x - π)^2 - (y - π)^2)
end

"""
    eggholder(x)

Eggholder function: https://www.sfu.ca/~ssurjano/egg.html

  - Dimensions: 2
  - Bounds: -512 ≤ x₁, x₂ ≤ 512
  - Global minimum: f(512, 404.2319) = -959.6407
"""
function eggholder(xy::VecOrTuple)
    x, y = xy
    return -(y + 47) * sin(sqrt(abs((x / 2.0) + (y + 47)))) - x * sin(sqrt(abs(x - (y + 47))))
end

"""
    exponential(x)

Exponential function

  - Dimensions: 1, 2, 3,...
  - Bounds: -1 ≤ xᵢ ≤ 1 for i=1, 2
  - Global minimum: f(0, 0,...) = -1

### References

S. Rahnamyan, H. R. Tizhoosh, N. M. M. Salama, "Opposition-Based
Differential Evolution (ODE) with Variable Jumping Rate", IEEE Sympousim
Foundations Computation Intelligence, Honolulu, HI, pp. 81-88, 2007.
"""
exponential(x::VecOrTuple) = -exp(-0.5 * sum(v^2 for v in x))

"""
    freudenstein_roth(x)

Freudenstein-Roth function: https://al-roomi.org/benchmarks/unconstrained/2-dimensions/55-freudenstein-roth-s-function

  - Dimensions: 2
  - Bounds: -10 ≤ x₁, x₂ ≤ 10
  - Global minimum: f(5, 4) = 0

### References

S. S. Rao, "Engineering Optimization: Theory and Practice, ”
John Wiley & Sons, 2009.
"""
function freudenstein_roth(xy::VecOrTuple)
    x, y = xy
    a = (x - 13 + ((5 - y) * y - 2) * y)^2
    b = (x - 29 + ((y + 1) * y - 14) * y)^2
    return a + b
end

"""
    goldstein_price(x)

Goldstein-Price function: https://www.sfu.ca/~ssurjano/goldpr.html

  - Dimensions: 2
  - Bounds: -2 ≤ x₁, x₂ ≤ 2
  - Global minimum: f(0, -1) = 3
"""
function goldstein_price(xy::VecOrTuple)
    x, y = xy
    return (1 + (x + y + 1)^2 * (19 - 14 * x + 3 * x^2 - 14 * y + 6 * x * y + 3 * y^2)) *
           (30 + (2 * x - 3 * y)^2 * (18 - 32 * x + 12 * x^2 + 48 * y - 36 * x * y + 27 * y^2))
end

"""
    griewank(x)

Griewank function: https://www.sfu.ca/~ssurjano/griewank.html

  - Dimensions: 1, 2, 3,...
  - Bounds: -600 ≤ xᵢ ≤ 600 for i=1,..., d
  - Global minimum: f(0, 0,..., 0) = 0

### References

A. O. Griewank, "Generalized Descent for Global Optimization", Journal of
Optimization Theory and Applications, vol. 34, no. 1, pp. 11-39, 1981.
"""
function griewank(xy::VecOrTuple)
    a, b = 0, 1
    for (i, v) in enumerate(xy)
        a += v^2 / 4000.0
        b *= cos(v / sqrt(i))
    end
    return a - b + 1
end

"""
    himmelblau(x)

Himmelblau's function: https://en.wikipedia.org/wiki/Himmelblau%27s_function

  - Dimensions: 2
  - Bounds: -5 ≤ x₁, x₂ ≤ 5
    global minimum(s):
    f(3.0, 2.0) = f(-2.805118, 3.131312) = f(-3.779310, -3.283186) = f(3.584428, -1.848126) = 0
"""
function himmelblau(xy::VecOrTuple)
    x, y = xy
    return (x^2 + y - 11)^2 + (x + y^2 - 7)^2
end

"""
    holder_table(x)

Hölder Table function: https://www.sfu.ca/~ssurjano/holder.html

  - Dimensions: 2
  - Bounds: -10 ≤ x₁, x₂ ≤ 10
  - Global minimum: f(±8.05502, ±9.66459) = -19.2085
"""
function holder_table(xy::VecOrTuple)
    x, y = xy
    return -abs(sin(x) * cos(y) * exp(abs(1 - (sqrt(x^2 + y^2) / π))))
end

"""
    hosaki(x)

Hosaki function: https://al-roomi.org/benchmarks/unconstrained/2-dimensions/58-hosaki-s-function

  - Dimensions: 2
  - Bounds: 0 ≤ x₁ ≤ 5, 0 ≤ x₂ ≤ 6
  - Global minimum: f(4, 2) = -2.345811

### References

G. A. Bekey, M. T. Ung, "A Comparative Evaluation of Two Global Search
Algorithms", IEEE Transaction on Systems, Man and Cybernetics, vol. 4,
no. 1, pp. 112-116, 1974.
"""
function hosaki(xy::VecOrTuple)
    x, y = xy
    return (1 - 8 * x + 7 * x^2 - (7.0 / 3.0) * x^3 + 0.25 * x^4) * y^2 * exp(-y)
end

"""
    keane(x)

Keane function: https://al-roomi.org/benchmarks/unconstrained/2-dimensions/135-keane-s-function

  - Dimensions: 2
  - Bounds: -10 ≤ x₁, x₂ ≤ 10
  - Global minimum: f(0, 1.3932490) = f(1.3932490, 0) = -0.6736675

### References

Momin Jamil and Xin-She Yang, A literature survey of benchmark functions for
global optimization problems, Int. Journal of Mathematical Modelling and
Numerical Optimisation}, Vol. 4, No. 2, pp. 150–194 (2013), arXiv:1308.4008
"""
function keane(xy::VecOrTuple)
    x, y = xy
    n = sin(x - y)^2 * sin(x + y)^2
    d = sqrt(x^2 + y^2)
    return -n / d
end

"""
    leon(x)

Leon function: https://al-roomi.org/benchmarks/unconstrained/2-dimensions/125-leon-s-function

  - Dimensions: 2
  - Bounds: -10 ≤ x₁, x₂ ≤ 10
  - Global minimum: f(1, 1) = 0

### References

A. Lavi, T. P. Vogel (eds), "Recent Advances in Optimization Techniques, ”
John Wliley & Sons, 1966.
"""
function leon(xy::VecOrTuple)
    x, y = xy
    return 100 * (y - x^3)^2 + (1 - x)^2
end

"""
    levy_n13(x)

Levy function N.13: https://www.sfu.ca/~ssurjano/levy13.html

  - Dimensions: 2
  - Bounds: -10 ≤ x₁, x₂ ≤ 10
  - Global minimum: f(1, 1) = 0
"""
function levy_n13(xy::VecOrTuple)
    x, y = xy
    return (
        sin(3.0 * π * x)^2 +
        (x - 1)^2 * (1 + sin(3.0 * π * y)^2) +
        (y - 1)^2 * (1 + sin(2.0 * π * y)^2)
    )
end

"""
    matyas(x)

Matyas function: https://www.sfu.ca/~ssurjano/matya.html

  - Dimensions: 2
  - Bounds: -10 ≤ x₁, x₂ ≤ 10
  - Global minimum: f(0, 0) = 0
"""
function matyas(xy::VecOrTuple)
    x, y = xy
    return 0.26 * (x^2 + y^2) - 0.48 * x * y
end

"""
    michalewicz(x, m = 10)

Michalewicz function: https://www.sfu.ca/~ssurjano/michal.html

The parameter m defines the steepness of they valleys and ridges; a larger m leads to a more difficult search. The recommended value of m is m = 10.

  - Dimensions: 1, 2, 3,...
  - Bounds: 0 ≤ xᵢ ≤ π for i=1, 2, 3,...
  - Global minimum (2D): f([2.20, 1.57]) = -1.8013
"""
function michalewicz(x::VecOrTuple, m = 10)
    d = length(x)
    result = 0.0
    for i = 1:d
        result -= sin(x[i]) * (sin(i * x[i]^2 / π))^(2 * m)
    end
    return result
end

"""
    mccormick(x)

McCormick function: https://www.sfu.ca/~ssurjano/mccorm.html

  - Dimensions: 2
  - Global minimum: f(-0.54719, -1.54719) = -1.9133
  - Bounds: -1.5 ≤ x₁ ≤ 4, -3 ≤ x₂ ≤ 4
"""
function mccormick(xy::VecOrTuple)
    x, y = xy
    return sin(x + y) + (x - y)^2 - 1.5 * x + 2.5 * y + 1
end

"""
    parsopoulos(x)

Parsopoulos function: https://al-roomi.org/benchmarks/unconstrained/2-dimensions/252-parsopoulos-function

  - Bounds: -11 ≤ xᵢ ≤ 11 for i=1, 2
  - There are infinitely many global minimums in R²:
    f(k * π / 2, l * π) = 0, where k = ±1, ±3,..., l = 0, ±1, ±2,...
"""
parsopoulos(xy::VecOrTuple) = cos(xy[1])^2 + sin(xy[2])^2

"""
    pen_holder(x)

Pen holder function

  - Dimensions: 2
  - Bounds: -11 ≤ x₁, x₂ ≤ 11
  - Global minimum(s): f(±9.6461677, ±9.6461677) = -0.9635348

### References

  - S. K. Mishra, "Global Optimization By Differential Evolution and Particle
    Swarm Methods: Evaluation On Some Benchmark Functions", Munich Research
    Papers in Economics, [Available Online]:
    http://mpra.ub.uni-muenchen.de/1005/
  - https://al-roomi.org/benchmarks/unconstrained/2-dimensions/64-pen-holder-function
"""
pen_holder(xy::VecOrTuple) =
    -exp(-((abs(cos(xy[1]) * cos(xy[2]) * exp(abs(1 - sqrt(xy[1]^2 + xy[2]^2) / π))))^-1))

"""
    plateau(x)

Plateau function

  - Dimensions: 1, 2, 3,...
  - Bounds: -5.12 ≤ xᵢ ≤ 5.12 for i=1, ..., n
  - Global minimum: f(0, 0,..., 0) = 30

### References

???
"""
plateau(x::VecOrTuple) = 30 + sum([floor(abs(v)) for v in x])

"""
Qing function

  - Dimensions: 1, 2, 3,...
  - Global minimum: f(±sqrt(1), ±sqrt(2),..., ±sqrt(n)) = 0
  - Bounds: -500 ≤ xᵢ ≤ 500

### References

  - A. Qing, "Dynamic Differential Evolution Strategy and Applications in
    Electromagnetic Inverse Scattering Problems", IEEE Transactions on
    Geoscience and remote Sensing, vol. 44, no. 1, pp. 116-125, 2006.
  - https://al-roomi.org/benchmarks/unconstrained/n-dimensions/185-qing-s-function
"""
qing(x::VecOrTuple) = sum((v^2 - i)^2 for (i, v) in enumerate(x))

"""
    quartic(x::VecOrTuple, rand_term = false)

Quartic function

  - Dimensions: 1, 2, 3,...
  - Global minimum: f(0, 0,..., 0) = 0
  - Bounds: -1.28 ≤ xᵢ ≤ 1.28

### References

R. Storn, K. Price, "Differntial Evolution - A Simple and Efficient Adaptive
Scheme for Global Optimization over Continuous Spaces", Technical Report
no. TR-95-012, International Computer Science Institute, Berkeley, CA, 1996:
https://www.researchgate.net/publication/227242104_Differential_Evolution_-_A_Simple_and_Efficient_Heuristic_for_Global_Optimization_over_Continuous_Spaces
"""
function quartic(x::VecOrTuple, rand_term = false)
    l = length(x)
    r = rand_term ? rand(l) : zeros(l)
    return sum(i * v^4 + r[i] for (i, v) in enumerate(x))
end

"""
    rastrigin(x, safe_mode=false)

Rastrigin function

  - Dimensions: 1, 2, 3,...
  - Bounds: -5.12 ≤ xᵢ ≤ 5.12
  - Global minimum: f(0,..., 0) = 0
  - Global maximum near f(±4.52299366, ±4.52299366,..., ±4.52299366)

### References

  - https://en.wikipedia.org/wiki/Rastrigin_function
"""
function rastrigin(x::VecOrTuple, safe_mode = false)
    if safe_mode
        if any(v -> 5.12 < v < -5.12, x)
            error("input exceeds bounds of (-5.12, 5.12)")
        end
    end
    return length(x) * 10.0 + sum(v * v - 10.0 * cos(2.0 * π * v) for v in x)
end

"""
    rotated_hyper_ellipsoid(x)

Rotated hyper-ellipsoid function

  - Dimensions: 1, 2, 3,...
  - Bounds: -65.536 ≤ xᵢ ≤ 65.536
  - Global minimum: f(0,..., 0) = 0

### References

  - Molga, M., & Smutnicki, C. Test functions for optimization needs (2005)
  - https://www.sfu.ca/~ssurjano/rothyp.html
"""
function rotated_hyper_ellipsoid(xy::VecOrTuple)
    a = 0.0
    for i = 1:length(xy)
        a += sum(xy[j]^2 for j = 1:i)
    end
    return a
end

"""
    rosenbrock(x::VecOrTuple; a = 1.0, b = 100.0)

Rosenbrock function: https://en.wikipedia.org/wiki/Rosenbrock_function

  - Dimensions: 2, 3,...
  - Bounds: -∞ ≤ xᵢ ≤ ∞
  - Global minimum, for default a, b: f(1, 1,..., 1) = 0
"""
function rosenbrock(x::VecOrTuple; a = 1.0, b = 100.0)
    total = 0.0
    for i = 1:(length(x) - 1)
        total += b * (x[i + 1] - x[i] * x[i])^2 + (a - x[i])^2
    end
    return total
end

"""
    salomon(x)

Salomon's function

  - Dimensions: 1, 2, 3,...
  - Bounds: -100 ≤ xᵢ ≤ 100
  - Global minimum: f(0, ..., 0) = 0

### Reference

  - R. Salomon, "Re-evaluating Genetic Algorithm Performance Under Corodinate
    Rotation of Benchmark Functions: A Survey of Some Theoretical and Practical
    Aspects of Genetic Algorithms", BioSystems, vol. 39, no. 3, pp. 263-278, 1996. 
  - https://al-roomi.org/benchmarks/unconstrained/n-dimensions/184-salomon-s-function
"""
function salomon(x::VecOrTuple)
    s = sqrt(sum(v^2 for v in x))
    return 1 - cos(2 * π * s) + 0.1 * s
end

"""
    schaffer_n2(x)

Schaffer function N.2: https://www.sfu.ca/~ssurjano/schaffer2.html

  - Dimensions: 2
  - Bounds: -100 ≤ x₁, x₂ ≤ 100
  - Global minimum: f(0, 0) = 0
"""
function schaffer_n2(xy::VecOrTuple)
    x, y = xy
    return 0.5 + (sin(x^2 - y^2)^2 - 0.5) / (1 + 0.001 * (x^2 + y^2))^2
end

"""
    schaffer_n4(x)

Schaffer function N.4: https://www.sfu.ca/~ssurjano/schaffer4.html

  - Dimensions: 2
  - Bounds: -100 ≤ x₁, x₂ ≤ 100
  - Global minimum: f(0, 1.25313) = f(0, -1.25313) = 0.292579
"""
function schaffer_n4(xy::VecOrTuple)
    x, y = xy
    return 0.5 + (cos(sin(abs(x^2 - y^2)))^2 - 0.5) / (1 + 0.001 * (x^2 + y^2))^2
end

"""
    schwefel(x)

Schwefel function: https://www.sfu.ca/~ssurjano/schwef.html

  - Dimensions: 1, 2, 3,...
  - Bounds: -500 ≤ xᵢ ≤ 500
  - Global minimum: f(420.9687, ..., 420.9687) = 0
"""
schwefel(x::VecOrTuple) = 418.9829 * length(x) - sum([v * sin(sqrt(abs(v))) for v in x])

"""
    sphere(x)

Sphere function: https://www.sfu.ca/~ssurjano/spheref.html

  - Dimensions: 1, 2, 3,...
  - Bounds: -5.12 ≤
  - Global minimum: f(0, 0,..., 0) = 0
"""
sphere(x::VecOrTuple) = sum(v -> v * v, x)

"""
    step_function(x)

Step function

  - Dimensions: 1, 2, 3,...
  - Bounds: -100 ≤ xᵢ ≤ 100
  - Global minimum: f(0, 0,..., 0) = 0

### References

Momin Jamil and Xin-She Yang, A literature survey of benchmark functions for
global optimization problems, Int. Journal of Mathematical Modelling and
Numerical Optimisation, Vol. 4, Issue. 2 (2013).
"""
step_function(xy::VecOrTuple) = sum(floor(abs(v)) for v in xy)

"""
    styblinski_tang(x)

Styblinski-Tang function: https://www.sfu.ca/~ssurjano/stybtang.html

  - Dimensions: 1, 2, 3,...
  - Bounds: -5 ≤ xᵢ ≤ 5
  - Global minimum: f(-2.903534, ..., -2.903534) = -39.16616d
"""
styblinski_tang(x::VecOrTuple) = sum([v^4 - 16 * v^2 + 5 * v for v in x]) / 2.0

"""
    sum_of_different_powers(x)

Sum of different powers: https://www.sfu.ca/~ssurjano/sumpow.html

  - Dimensions: 1, 2, 3,...
  - Bounds: -1 ≤ xᵢ ≤ 1
  - Global minimum: f(0, 0,..., 0)=0
"""
sum_of_different_powers(x::VecOrTuple) = sum(abs(v)^(i + 1) for (i, v) in enumerate(x))

"""
    sum_of_squares(x)

Sum of Squares function

  - Dimensions: 1, 2, 3,...
  - Bounds: -10 ≤ xᵢ ≤ 10
  - Global minimum: f(0, 0,..., 0) = 0

### References

  - A.-R. Hedar, "Global Optimization Test Problems”
  - https://www.sfu.ca/~ssurjano/sumsqu.html
"""
sum_of_squares(xy::VecOrTuple) = sum(i * v^2 for (i, v) in enumerate(xy))

"""
    trid(x)

Trid function

  - Dimensions: 1, 2, 3,...
  - Bounds: -d² ≤ xᵢ ≤ d²
  - Global minimum: f(x₁, x₂,...,xₙ)=-n(n+4)(n-1)/6 with xᵢ = i(n+1-i)

### References

  - A.-R. Hedar, "Global Optimization Test Problems”
  - https://www.sfu.ca/~ssurjano/trid.html
"""
function trid(xy::VecOrTuple)
    l = length(xy)
    a = sum((v - 1)^2 for v in xy)
    b = l < 2 ? 0.0 : sum((xy[i + 1] * xy[i]) for (i, v) in enumerate(@view(xy[2:end])))
    return a - b
end

"""
    tripod(x)

Tripod function: https://doi.org/10.1016/j.camwa.2006.07.013

  - Dimensions: 2
  - Bounds: -100 ≤ xᵢ ≤ 100
  - Global minimum: f(0, 50) = 0

### References

  - S. Rahnamyan, H. R. Tizhoosh, N. M. M. Salama, "A Novel Population
    Initialization Method for Accelerating Evolutionary Algorithms”
    Computers and Mathematics with Applications, vol. 53, no. 10,
    pp. 1605-1614, 2007.
"""
function tripod(xy::VecOrTuple)
    x, y = xy
    p_x = x >= 0 ? 1 : 0
    p_y = y >= 0 ? 1 : 0
    a = p_y * (1 + p_x)
    b = abs(x + 50 * p_y * (1 - 2 * p_x))
    c = abs(y + 50 * (1 - 2 * p_y))
    return a + b + c
end

"""
    wolfe(x)

Wolfe function

  - Dimensions: 3
  - Bounds: 0 ≤ xᵢ ≤ 2
  - Global minimum: f(0, 0, 0) = 0

### References

  - H. P. Schwefel, "Numerical Optimization for Computer Models", John Wiley
    Sons, 1981.
  - https://al-roomi.org/benchmarks/unconstrained/3-dimensions/251-wolfe-s-function
"""
function wolfe(xyz::VecOrTuple)
    x, y, z = xyz
    return (4.0 / 3.0) * (x^2.0 + y^2.0 - x * y)^0.75 + z
end

"""
    zakharov(x)

Zakharov function
The Zakharov function has no local minima, but has a single global minimum.

  - Dimensions: 1, 2, 3,...
  - Bounds: -5 ≤ xᵢ ≤ 10
  - Global minimum: f(0, 0,..., 0) = 0

### References

  - Shahryar Rahnamayan, Hamid R. Tizhoosh, Magdy M.A. Salama,
    "A novel population initialization method for accelerating evolutionary
    algorithms" - Computers & Mathematics with Applications
    Volume 53, Issue 10, 2007, Pages 1605-1614, ISSN 0898-1221
  - https://www.sfu.ca/~ssurjano/zakharov.html
"""
function zakharov(x::VecOrTuple)
    a, b = 0, 0
    for (i, val) in enumerate(x)
        a += val^2
        b += 0.5 * i * val
    end
    return a + b^2 + b^4
end
