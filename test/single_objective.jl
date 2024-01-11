
const D_MIN = 1
const D_MAX = 20

distance2(a, b) = sum((x - y)^2 for (x, y) in zip(a, b))

@testset "Single objective" begin
    @testset "ackley" begin
        x = [0, 0]
        @test ackley(x) == 0
        res = bboptimize(ackley; SearchRange = (-5, 5), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "ackley_n2" begin
        x = [0, 0]
        @test -200 ≈ ackley_n2(x) atol = 1e-6
        res = bboptimize(ackley_n2; SearchRange = (-32, 32), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "adjiman" begin
        x = [2.0, 0.105783]
        @test -2.02180678 ≈ adjiman(x) atol = 1e-6
        res = bboptimize(adjiman; SearchRange = (-2, 2), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-6
        x = [5.0, -0.0284005]
        @test -5.004025373 ≈ adjiman(x) atol = 1e-6
        res = bboptimize(adjiman; SearchRange = (-5, 5), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "amgm" begin
        @test amgm([1, 1, 1]) == 0
        @test amgm([1.5, 1.5, 1.5]) == 0
        @test amgm([0, 0, 0]) == 0
        @test amgm([1, 2, 3, 4, 5, 6, 7]) != 0
    end

    @testset "bartels_conn" begin
        x = [0, 0]
        @test bartels_conn([0, 0]) == 1
        res = bboptimize(
            bartels_conn;
            SearchRange = (-500, 500),
            NumDimensions = 2,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "beale" begin
        x = [3, 0.5]
        @test beale(x) == 0
        res = bboptimize(beale; SearchRange = (-4.5, 4.5), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "bent_cigar" begin
        for d = D_MIN:D_MAX
            x = zeros(d)
            @test bent_cigar(x) == 0
            res = bboptimize(
                bent_cigar;
                MaxSteps = 1e6,
                SearchRange = (-4.5, 4.5),
                NumDimensions = d,
                TraceMode = :silent,
            )
            @test x ≈ best_candidate(res) atol = 1e-1
        end
    end

    @testset "bird" begin
        x1 = [4.701055, 3.152946]
        x2 = [-1.582142, -3.130246]
        minval = -106.764536
        @test minval ≈ bird(x1) atol = 1e-6
        @test minval ≈ bird(x2) atol = 1e-6
        counts = [0, 0]
        runs = 10
        tol = 1e-4
        for _ = 1:runs
            res = bboptimize(bird; SearchRange = (-2π, 2π), NumDimensions = 2, TraceMode = :silent)
            m = best_candidate(res)
            if distance2(m, x1) < distance2(m, x2)
                @test m ≈ x1 atol = tol
                counts[1] += 1
            else
                @test m ≈ x2 atol = tol
                counts[2] += 1
            end
        end
        @test all(>(0), counts)
    end

    @testset "bohachevsky_n1" begin
        x = [0, 0]
        @test bohachevsky_n1(x) == 0
        res = bboptimize(
            bohachevsky_n1;
            SearchRange = (-100, 100),
            NumDimensions = 2,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "bohachevsky_n2" begin
        x = [0, 0]
        @test bohachevsky_n2(x) == 0
        res = bboptimize(
            bohachevsky_n2;
            SearchRange = (-100, 100),
            NumDimensions = 2,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "bohachevsky_n3" begin
        x = [0, 0]
        @test bohachevsky_n3(x) == 0
        res = bboptimize(
            bohachevsky_n3;
            SearchRange = (-100, 100),
            NumDimensions = 2,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "booth" begin
        x = [1, 3]
        @test booth(x) == 0
        res = bboptimize(booth; SearchRange = (-10, 10), NumDimensions = 2, TraceMode = :silent)
        @test best_candidate(res) ≈ x atol = 1e-6
    end

    @testset "branin" begin
        x1 = [-π, 12.275]
        x2 = [π, 2.275]
        x3 = [3π, 2.475]
        minval = 0.397887
        @test minval ≈ branin(x1) atol = 1e-6
        @test minval ≈ branin(x2) atol = 1e-6
        @test minval ≈ branin(x3) atol = 1e-6
        counts = [0, 0, 0]
        runs = 30
        tol = 1e-6
        for _ = 1:runs
            res = bboptimize(branin; SearchRange = (-5, 15), NumDimensions = 2, TraceMode = :silent)
            m = best_candidate(res)
            d1, d2, d3 = distance2(m, x1), distance2(m, x2), distance2(m, x3)
            if d1 < d2 && d1 < d3
                @test m ≈ x1 atol = tol
                counts[1] += 1
            elseif d2 < d1 && d2 < d3
                @test m ≈ x2 atol = tol
                counts[2] += 1
            elseif d3 < d1 && d3 < d2
                @test m ≈ x3 atol = tol
                counts[3] += 1
            end
        end
        @test all(>(0), counts)
    end

    @testset "brent" begin
        x = [-10, -10]
        @test 0 ≈ brent(x) atol = 1e-6
        res = bboptimize(brent; SearchRange = (-10, 2), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "brown" begin
        for d = D_MIN:D_MAX
            x = zeros(d)
            @test brown(x) == 0
            # for d=1 the function is always zero, so the search is unstable
            # for large d, the search doesn't converge quickly, one day we can try to
            #           stabilise it...
            if 1 < d < 8
                res =
                    bboptimize(brown; SearchRange = (-1, 4), NumDimensions = d, TraceMode = :silent)
                @test x ≈ best_candidate(res) atol = 1e-4
            end
        end
    end

    @testset "bukin_n6" begin
        x = [-10, 1]
        @test bukin_n6(x) == 0
        # This doesn't work. Try other methods?
        #res = bboptimize(bukin_n6; SearchRange = (-15, 3), NumDimensions = 2, TraceMode = :silent)
        #@test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "camel_hump_3" begin
        x = [0, 0]
        @test camel_hump_3(x) == 0
        res =
            bboptimize(camel_hump_3; SearchRange = (-5, 5), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-3
    end

    @testset "camel_hump_6" begin
        x1 = [0.0898, -0.7126]
        x2 = [-0.0898, 0.7126]
        minval = -1.0316
        @test minval ≈ camel_hump_6(x1) atol = 1e-4
        @test minval ≈ camel_hump_6(x2) atol = 1e-4
        runs = 6
        counts = [0, 0]
        tol = 1e-3
        for _ = 1:runs
            res = bboptimize(
                camel_hump_6;
                SearchRange = (-5, 5),
                NumDimensions = 2,
                TraceMode = :silent,
            )
            m = best_candidate(res)
            if distance2(m, x1) < distance2(m, x2)
                @test x1 ≈ m atol = tol
                counts[1] += 1
            else
                @test x2 ≈ m atol = tol
                counts[2] += 1
            end
        end
        @test all(>(0), counts)
    end

    @testset "carrom_table" begin
        xy = 9.6461572
        minval = -24.1568155
        @test minval ≈ carrom_table([+xy, +xy]) atol = 1e-6
        @test minval ≈ carrom_table([+xy, -xy]) atol = 1e-6
        @test minval ≈ carrom_table([-xy, +xy]) atol = 1e-6
        @test minval ≈ carrom_table([-xy, -xy]) atol = 1e-6
        runs = 20
        counts = [0, 0, 0, 0]
        tol = 1e-4
        for _ = 1:runs
            res = bboptimize(
                carrom_table;
                SearchRange = (-10, 10),
                NumDimensions = 2,
                TraceMode = :silent,
            )
            x1, x2 = best_candidate(res)
            if x1 > 0
                if x2 > 0
                    @test [+xy, +xy] ≈ [x1, x2] atol = tol
                    counts[1] += 1
                else
                    @test [+xy, -xy] ≈ [x1, x2] atol = tol
                    counts[2] += 1
                end
            else
                if x2 > 0
                    @test [-xy, +xy] ≈ [x1, x2] atol = tol
                    counts[3] += 1
                else
                    @test [-xy, -xy] ≈ [x1, x2] atol = tol
                    counts[4] += 1
                end
            end
        end
        @test all(>(0), counts)
    end

    @testset "chichinadze" begin
        x = [6.189866586965680, 0.5]
        @test -42.9443870 ≈ chichinadze(x) atol = 1e-6
        res =
            bboptimize(chichinadze; SearchRange = (-30, 30), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "chung_reynolds" begin
        for d = D_MIN:D_MAX
            x = zeros(d)
            @test chung_reynolds(x) == 0
            if d < 10
                res = bboptimize(
                    chung_reynolds;
                    SearchRange = (-100, 100),
                    NumDimensions = d,
                    TraceMode = :silent,
                )
                @test x ≈ best_candidate(res) atol = 1e-1
            end
        end
    end

    @testset "colville" begin
        x = [1, 1, 1, 1]
        @test colville(x) == 0
        res = bboptimize(colville; SearchRange = (-10, 10), NumDimensions = 4, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-2
    end

    @testset "cosine_mixture" begin
        for d = D_MIN:D_MAX
            g_min = -0.1 * d
            x = zeros(d)
            @test g_min ≈ cosine_mixture(x) atol = 1e-6
            res = bboptimize(
                cosine_mixture;
                MaxSteps = 100000,
                SearchRange = (-1, 1),
                NumDimensions = d,
                TraceMode = :silent,
            )
            @test x ≈ best_candidate(res) atol = 1e-6
        end
    end

    @testset "cross_in_tray" begin
        xy = 1.34941
        minval = -2.06261
        @test minval ≈ cross_in_tray([xy, xy]) atol = 1e-5
        @test minval ≈ cross_in_tray([xy, -xy]) atol = 1e-5
        @test minval ≈ cross_in_tray([-xy, xy]) atol = 1e-5
        @test minval ≈ cross_in_tray([-xy, -xy]) atol = 1e-5
        runs = 20
        counts = [0, 0, 0, 0]
        tol = 1e-4
        for _ = 1:runs
            res = bboptimize(
                cross_in_tray;
                SearchRange = (-10, 10),
                NumDimensions = 2,
                TraceMode = :silent,
            )
            x1, x2 = best_candidate(res)
            if x1 > 0
                if x2 > 0
                    @test [+xy, +xy] ≈ [x1, x2] atol = tol
                    counts[1] += 1
                else
                    @test [+xy, -xy] ≈ [x1, x2] atol = tol
                    counts[2] += 1
                end
            else
                if x2 > 0
                    @test [-xy, +xy] ≈ [x1, x2] atol = tol
                    counts[3] += 1
                else
                    @test [-xy, -xy] ≈ [x1, x2] atol = tol
                    counts[4] += 1
                end
            end
        end
        @test all(>(0), counts)
    end

    @testset "csendes" begin
        for d = D_MIN:D_MAX
            x = zeros(d)
            @test csendes(x) == 0
            if d < 8
                res = bboptimize(
                    csendes;
                    SearchRange = (-10, 10),
                    NumDimensions = d,
                    TraceMode = :silent,
                )
                @test x ≈ best_candidate(res) atol = 1e-2
            end
        end
    end

    @testset "cube" begin
        x = [1, 1]
        @test cube(x) == 0
        res = bboptimize(cube; SearchRange = (-10, 10), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "damavandi" begin
        x = [2, 2]
        @test damavandi(x) == 0
    end

    @testset "dekker_aarts" begin
        minval = -24771.09375
        x, y = 0, 15
        @test minval ≈ dekker_aarts([x, +y]) atol = 1e-5
        @test minval ≈ dekker_aarts([x, -y]) atol = 1e-5
        runs = 10
        counts = [0, 0]
        tol = 1e-1
        for _ = 1:runs
            res = bboptimize(
                dekker_aarts;
                SearchRange = (-20, 20),
                NumDimensions = 2,
                TraceMode = :silent,
            )
            x1, x2 = best_candidate(res)
            if x2 > 0
                @test [x, +y] ≈ [x1, x2] atol = tol
                counts[1] += 1
            else
                @test [x, -y] ≈ [x1, x2] atol = tol
                counts[2] += 1
            end
        end
        @test all(>(0), counts)
    end

    @testset "dixon_price" begin
        for d = D_MIN:D_MAX
            g_min = [2^(-(2^i - 2) / 2.0^i) for i = 1:d]
            @test 0 ≈ dixon_price(g_min) atol = 1e-6
        end
    end

    @testset "drop_wave" begin
        x = [0, 0]
        @test drop_wave(x) == -1
        res = bboptimize(
            drop_wave;
            SearchRange = (-5.12, 5.12),
            NumDimensions = 2,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "easom" begin
        x = [π, π]
        @test easom(x) == -1
        res = bboptimize(
            easom;
            MaxSteps = 1000000,
            Method = :generating_set_search,
            SearchRange = (-10, 10),
            NumDimensions = 2,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "eggholder" begin
        x = [512, 404.2319]
        @test -959.6407 ≈ eggholder(x) atol = 1e-4
        res = bboptimize(
            eggholder;
            MaxSteps = 1000000,
            Method = :generating_set_search,
            SearchRange = (-512, 512),
            NumDimensions = 2,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-3
    end

    @testset "exponential" begin
        for d = D_MIN:D_MAX
            x = zeros(d)
            @test exponential(x) == -1
            if d < 9
                res = bboptimize(
                    exponential;
                    SearchRange = (-1, 1),
                    NumDimensions = d,
                    TraceMode = :silent,
                )
                @test x ≈ best_candidate(res) atol = 1e-4
            end
        end
    end

    @testset "freudenstein_roth" begin
        x = [5, 4]
        @test freudenstein_roth(x) == 0
        res = bboptimize(
            freudenstein_roth;
            SearchRange = (-10, 10),
            NumDimensions = 2,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "goldstein_price" begin
        x = [0, -1]
        @test goldstein_price(x) == 3
        res = bboptimize(
            goldstein_price;
            MaxSteps = 1000000,
            Method = :generating_set_search,
            SearchRange = (-2, 2),
            NumDimensions = 2,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "griewank" begin
        @test griewank([0]) == 0
        @test griewank([0, 0, 0, 0, 0]) == 0
    end

    @testset "himmelblau" begin
        x1 = [3, 2]
        x2 = [3.584428, -1.848126]
        x3 = [-2.805118, 3.131312]
        x4 = [-3.779310, -3.283186]
        @test 0 ≈ himmelblau(x1) atol = 1e-10
        @test 0 ≈ himmelblau(x2) atol = 1e-10
        @test 0 ≈ himmelblau(x3) atol = 1e-10
        @test 0 ≈ himmelblau(x4) atol = 1e-10
        runs = 100 # x4 is not found often, need lots of runs
        counts = [0, 0, 0, 0]
        for _ = 1:runs
            res = bboptimize(
                himmelblau;
                SearchRange = (-5, 5),
                NumDimensions = 2,
                TraceMode = :silent,
            )
            x, y = best_candidate(res)
            if x > 0
                if y > 0
                    @test x1 ≈ best_candidate(res) atol = 1e-6
                    counts[1] += 1
                else
                    @test x2 ≈ best_candidate(res) atol = 1e-6
                    counts[2] += 1
                end
            else
                if y > 0
                    @test x3 ≈ best_candidate(res) atol = 1e-6
                    counts[3] += 1
                else
                    @test x4 ≈ best_candidate(res) atol = 1e-6
                    counts[4] += 1
                end
            end
        end
        @test all(>(0), counts)
    end

    @testset "holder_table" begin
        x, y = 8.05502, 9.66459
        minval = -19.2085
        @test minval ≈ holder_table([+x, +y]) atol = 1e-5
        @test minval ≈ holder_table([+x, -y]) atol = 1e-5
        @test minval ≈ holder_table([-x, +y]) atol = 1e-5
        @test minval ≈ holder_table([-x, -y]) atol = 1e-5
        runs = 20
        counts = [0, 0, 0, 0]
        for _ = 1:runs
            res = bboptimize(
                holder_table;
                SearchRange = (-10, 10),
                NumDimensions = 2,
                TraceMode = :silent,
            )
            mx, my = best_candidate(res)
            if mx > 0
                if my > 0
                    @test [+x, +y] ≈ best_candidate(res) atol = 1e-4
                    counts[1] += 1
                else
                    @test [+x, -y] ≈ best_candidate(res) atol = 1e-4
                    counts[2] += 1
                end
            else
                if my > 0
                    @test [-x, +y] ≈ best_candidate(res) atol = 1e-4
                    counts[3] += 1
                else
                    @test [-x, -y] ≈ best_candidate(res) atol = 1e-4
                    counts[4] += 1
                end
            end
        end
        @test all(>(0), counts)
    end

    @testset "hosaki" begin
        x = [4, 2]
        @test -2.345811 ≈ hosaki(x) atol = 1e-5
        res = bboptimize(hosaki; SearchRange = (0, 10), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "keane" begin
        minval = -0.6736675
        xy = 1.3932490
        @test minval ≈ keane([0, xy]) atol = 1e-6
        @test minval ≈ keane([xy, 0]) atol = 1e-6
        runs = 10
        counts = [0, 0]
        for _ = 1:runs
            res = bboptimize(keane; SearchRange = (0, 10), NumDimensions = 2, TraceMode = :silent)
            x, y = best_candidate(res)
            if x > y
                @test [xy, 0] ≈ best_candidate(res) atol = 1e-6
                counts[1] += 1
            else
                @test [0, xy] ≈ best_candidate(res) atol = 1e-6
                counts[2] += 1
            end
        end
        @test all(>(0), counts)
    end

    @testset "levy_n13" begin
        x = [1, 1]
        @test 0 ≈ levy_n13(x) atol = 1e-10
        res = bboptimize(levy_n13; SearchRange = (0, 10), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "leon" begin
        x = [1, 1]
        @test leon(x) == 0
        res = bboptimize(leon; SearchRange = (-10, 10), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "matyas" begin
        x = [0, 0]
        @test matyas(x) == 0
        res = bboptimize(matyas; SearchRange = (-10, 10), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "michalewicz" begin
        x = [2.20291, 1.5708]
        @test -1.8013 ≈ michalewicz(x) atol = 1e-5
        res = bboptimize(
            michalewicz;
            MaxSteps = 1000000,
            SearchRange = (0, π),
            NumDimensions = 2,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-5
    end

    @testset "mccormick" begin
        x = [-0.54719, -1.54719]
        @test -1.9133 ≈ mccormick(x) atol = 1e-4
        res = bboptimize(mccormick; SearchRange = (-3, 4), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-4
    end

    @testset "parsopoulos" begin
        for k in [-1, 1, -3, 3]
            for l in [0, -1, 1, -2, 2]
                @test 0 ≈ parsopoulos([k * π / 2, l * π]) atol = 1e-6
            end
        end
    end

    @testset "pen_holder" begin
        xy = 9.6461677
        minval = -0.9635348
        @test minval ≈ pen_holder([+xy, +xy]) atol = 1e-6
        @test minval ≈ pen_holder([+xy, -xy]) atol = 1e-6
        @test minval ≈ pen_holder([-xy, +xy]) atol = 1e-6
        @test minval ≈ pen_holder([-xy, -xy]) atol = 1e-6
        runs = 20
        counts = [0, 0, 0, 0]
        for _ = 1:runs
            res = bboptimize(
                pen_holder;
                SearchRange = (-11, 11),
                NumDimensions = 2,
                TraceMode = :silent,
            )
            mx, my = best_candidate(res)
            if mx > 0
                if my > 0
                    @test [+xy, +xy] ≈ best_candidate(res) atol = 1e-4
                    counts[1] += 1
                else
                    @test [+xy, -xy] ≈ best_candidate(res) atol = 1e-4
                    counts[2] += 1
                end
            else
                if my > 0
                    @test [-xy, +xy] ≈ best_candidate(res) atol = 1e-4
                    counts[3] += 1
                else
                    @test [-xy, -xy] ≈ best_candidate(res) atol = 1e-4
                    counts[4] += 1
                end
            end
        end
        @test all(>(0), counts)
    end

    @testset "plateau" begin
        for d = D_MIN:D_MAX
            x = zeros(d)
            @test plateau(x) == 30
        end
    end

    @testset "qing" begin
        for d = D_MIN:D_MAX
            x = [sqrt(i) for i = 1:d]
            @test 0 ≈ qing(x) atol = 1e-6
            if d < 4
                for _ = 1:10
                    res = bboptimize(
                        qing;
                        SearchRange = (-500, 500),
                        NumDimensions = d,
                        TraceMode = :silent,
                    )
                    @test abs.(x) ≈ abs.(best_candidate(res)) atol = 1e-2
                end
            end
        end
    end

    @testset "quartic, no noise" begin
        for d = D_MIN:D_MAX
            x = zeros(d)
            @test quartic(x) == 0
            if d < 8
                res = bboptimize(
                    quartic;
                    SearchRange = (-1.28, 1.28),
                    NumDimensions = d,
                    TraceMode = :silent,
                )
                @test x ≈ best_candidate(res) atol = 1e-4
            end
        end
    end

    @testset "rastrigin" begin
        x = [0]
        @test rastrigin(x) == 0
        res = bboptimize(
            rastrigin;
            SearchRange = (-5.12, 5.12),
            NumDimensions = 1,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-6

        x = [0, 0, 0, 0, 0]
        @test rastrigin(x) == 0
        res = bboptimize(
            rastrigin;
            SearchRange = (-5.12, 5.12),
            NumDimensions = 5,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-3
    end

    @testset "rotated_hyper_ellipsoid" begin
        x = [0]
        @test rotated_hyper_ellipsoid(x) == 0
        res = bboptimize(
            rotated_hyper_ellipsoid;
            SearchRange = (-65.536, 65.536),
            NumDimensions = 1,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-6

        x = [0, 0, 0, 0, 0]
        @test rotated_hyper_ellipsoid(x) == 0
        res = bboptimize(
            rotated_hyper_ellipsoid;
            SearchRange = (-65.536, 65.536),
            NumDimensions = 5,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-2
    end

    @testset "rosenbrock" begin
        x = [1]
        @test rosenbrock(x) == 0
        x = [1, 1, 1, 1, 1]
        @test rosenbrock(x) == 0
    end

    @testset "salomon" begin
        for d = D_MIN:D_MAX
            x = zeros(d)
            @test 0 ≈ salomon(x) atol = 1e-6
        end
    end

    @testset "schaffer_n2" begin
        x = [0, 0]
        @test schaffer_n2(x) == 0
        res = bboptimize(
            schaffer_n2;
            SearchRange = (-100, 100),
            NumDimensions = 2,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-6
    end

    @testset "schaffer_n4" begin
        yy = 1.25313
        minval = 0.292579
        @test minval ≈ schaffer_n4([0, yy]) atol = 1e-6
        @test minval ≈ schaffer_n4([0, -yy]) atol = 1e-6
    end

    @testset "schwefel" begin
        for d = D_MIN:6
            x = fill(420.96874, d)
            @test 0 ≈ schwefel(x) atol = 1e-4
            res = bboptimize(
                schwefel;
                SearchRange = (-500, 500),
                NumDimensions = d,
                TraceMode = :silent,
            )
            @test x ≈ best_candidate(res) atol = 1e-2
        end
    end

    @testset "sphere" begin
        x = [0]
        @test sphere(x) == 0
        res =
            bboptimize(sphere; SearchRange = (-5.12, 5.12), NumDimensions = 1, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-2
        x = [0, 0, 0, 0, 0]
        @test sphere(x) == 0
        res =
            bboptimize(sphere; SearchRange = (-5.12, 5.12), NumDimensions = 5, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-2
    end

    @testset "step_function" begin
        for d = D_MIN:D_MAX
            x = zeros(d)
            @test step_function(x) == 0
            if d < 8
                res = bboptimize(
                    sphere;
                    SearchRange = (-100, 100),
                    NumDimensions = d,
                    TraceMode = :silent,
                )
                @test x ≈ best_candidate(res) atol = 1e-2
            end
        end
    end

    @testset "styblinski_tang" begin
        pos = -2.903534
        minval = -39.16617

        d = 1
        x = fill(pos, d)
        @test minval * d ≈ styblinski_tang(x) atol = 0.00001
        res = bboptimize(
            styblinski_tang;
            SearchRange = (-5, 5),
            NumDimensions = d,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-4

        d = 5
        x = fill(pos, d)
        @test minval * d ≈ styblinski_tang(x) atol = 0.00001 * d
        res = bboptimize(
            styblinski_tang;
            SearchRange = (-5, 5),
            NumDimensions = d,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-4
    end

    @testset "sum_of_different_powers" begin
        d = 1
        x = zeros(d)
        @test sum_of_different_powers(x) == 0
        res = bboptimize(
            sum_of_different_powers;
            SearchRange = (-1, 1),
            NumDimensions = d,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-4

        d = 5
        x = zeros(d)
        @test sum_of_different_powers(x) == 0
        res = bboptimize(
            sum_of_different_powers;
            SearchRange = (-1, 1),
            NumDimensions = d,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-3
    end

    @testset "sum_of_squares" begin
        d = 1
        x = zeros(d)
        @test sum_of_squares(x) == 0
        res = bboptimize(
            sum_of_squares;
            SearchRange = (-5, 5),
            NumDimensions = d,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-4

        d = 5
        x = zeros(d)
        @test sum_of_squares(x) == 0
        res = bboptimize(
            sum_of_squares;
            SearchRange = (-5, 5),
            NumDimensions = d,
            TraceMode = :silent,
        )
        @test x ≈ best_candidate(res) atol = 1e-4
    end

    @testset "trid" begin
        for d in (1, 5)
            x = [i * (d + 1 - i) for i = 1:d]
            sol = -d * (d + 4) * (d - 1) / 6.0
            @test trid(x) == sol
            res =
                bboptimize(trid; SearchRange = (-d^2, d^2), NumDimensions = d, TraceMode = :silent)
            @test x ≈ best_candidate(res) atol = 1e-3
        end
    end

    @testset "tripod" begin
        x = [0, -50]
        @test tripod(x) == 0
        res = bboptimize(tripod; SearchRange = (-100, 100), NumDimensions = 2, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-3
    end

    @testset "wolfe" begin
        x = [0, 0, 0]
        @test wolfe(x) == 0
        res = bboptimize(wolfe; SearchRange = (0, 2), NumDimensions = 3, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-3
    end

    @testset "zakharov" begin
        d = 1
        x = zeros(d)
        @test zakharov(x) == 0
        res = bboptimize(zakharov; SearchRange = (-5, 5), NumDimensions = d, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-3

        d = 5
        x = zeros(d)
        @test zakharov(x) == 0
        res = bboptimize(zakharov; SearchRange = (-5, 5), NumDimensions = d, TraceMode = :silent)
        @test x ≈ best_candidate(res) atol = 1e-3
    end
end
