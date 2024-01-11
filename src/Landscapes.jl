module Landscapes

include("single_objective.jl")

"""
    safe_division(n, d)

Taking care of zero denominators.
"""
safe_division(n, d) = iszero(d) ? 0.0 : n / d

end # module Landscapes
