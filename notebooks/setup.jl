
#=
This file holds the plotting code and the setup code for the
notebook files in this directory.
=#

using CairoMakie
using Landscapes
CairoMakie.activate!()

"""
    landscape(f, bounds; ex_pts = nothing, elevation=0.26, azimuth=-2.2)

Plot the first two dimentions of `f` as a 3D surface plot, and a heatmap
plot. Return the constructed Makie plot.

  - `bounds`: a tuple, `(min, max)` containing the minimum and maximum values
    in both dimensions.
  - `ex_pts`: a vector of tuples, `[(x₁, y₁), (x₂, y₂), ...]`. The locations
    are added to the heatmap as dots. The name is `ex_pts` becuse the intention is
    to mark the known extremal points of `f`.
  - `elevation`, `azimuth` are used to modify the default projection of the
    surface plot (functions have better and worse projections).
"""
function landscape(f, bounds; ex_pts = nothing, elevation = 0.26, azimuth = -2.2)
    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), size = (800, 400))
    xs = ys = LinRange(bounds[1], bounds[2], 100)
    zs = [f([x, y]) for x in xs, y in ys]
    ax1, sf = surface(
        fig[1, 1],
        xs,
        ys,
        zs;
        axis = (type = Axis3, zlabel = "", elevation = elevation, azimuth = azimuth),
        colormap = :rainbow2,
    )
    ax2, hm = heatmap(fig[1, 2], xs, ys, zs; axis = (aspect = 1,), colormap = :rainbow2)
    fig[:, 3] = Colorbar(fig[1, 2], limits = (minimum(zs), maximum(zs)))
    if !isnothing(ex_pts)
        for zr in ex_pts
            scatter!(fig[1, 2], zr[1], zr[2], color = :yellow, markersize = 6.0)
        end
    end
    return fig
end
