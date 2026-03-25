# ═══════════════════════════════════════════════════════════════
# Water Fixture Classification — Raw Signal HMM (RxInfer)
# ═══════════════════════════════════════════════════════════════
#
# Instead of hand-crafted features, this model operates directly
# on the discretized magnetometer time series. Each fixture type
# gets its own HMM that learns:
#   - Transition matrix A: how hidden phases flow into each other
#   - Emission matrix B: what freq+vibration pattern each phase emits
#
# Classification = run all HMMs on a new signal, pick lowest free energy.
#
# ═══════════════════════════════════════════════════════════════
#
# TUNABLE PARAMETERS
# ═══════════════════════════════════════════════════════════════
#
# N_STATES
#   Number of hidden states (phases) per HMM.
#   Default: 5
#   Increase: model can capture more complex phase structure
#     (e.g. multiple plateau levels). Risk of overfitting with few events.
#   Decrease: simpler model, needs less data to train.
#
# FREQ_BINS / VIB_BINS
#   Bin edges for discretizing continuous readings.
#   Adjust these if your sensor's range differs from the defaults.
#   More bins = finer resolution but sparser observations.
#
# N_ITERATIONS
#   Number of variational inference iterations.
#   Default: 20
#   Increase if free energy hasn't converged (check convergence plot).
#
# EMISSION_SMOOTHING
#   Pseudo-count added to emission prior (Dirichlet).
#   Default: 1.0 (uniform smoothing)
#   Increase: more regularisation, less overfitting to single events.
#   Decrease: model fits training data more tightly.
#
# ═══════════════════════════════════════════════════════════════

using RxInfer
using JSON
using CSV
using DataFrames

# ── Parameters ────────────────────────────────────────────────

const N_STATES = 5

const N_BINS_PER_CHANNEL = 3
const N_CHANNELS = 4  # dx, dy, dz, |d|
const N_OBS = N_BINS_PER_CHANNEL ^ N_CHANNELS  # 81

const N_ITERATIONS = 20
const EMISSION_SMOOTHING = 1.0

# ── Discretization ────────────────────────────────────────────
# Uses first-differences of raw x/y/z magnetometer readings.
# dx[t] = x[t] - x[t-1] encodes oscillation speed and direction.
# Global bin edges must be provided (computed from training data).

function bin_channel(values::Vector{Float64}, edges::Vector{Float64})
    return [clamp(searchsortedlast(edges, v), 0, N_BINS_PER_CHANNEL - 1) for v in values]
end

function discretize_signal(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64},
                           dx_edges::Vector{Float64}, dy_edges::Vector{Float64},
                           dz_edges::Vector{Float64}, dmag_edges::Vector{Float64})
    dx = diff(x)
    dy = diff(y)
    dz = diff(z)
    dmag = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)

    dxb = bin_channel(dx, dx_edges)
    dyb = bin_channel(dy, dy_edges)
    dzb = bin_channel(dz, dz_edges)
    dmb = bin_channel(dmag, dmag_edges)

    obs = Int[]
    for i in eachindex(dx)
        o = dxb[i] * N_BINS_PER_CHANNEL^3 + dyb[i] * N_BINS_PER_CHANNEL^2 +
            dzb[i] * N_BINS_PER_CHANNEL + dmb[i] + 1  # 1-indexed for Julia
        push!(obs, o)
    end
    return obs
end

function obs_to_onehot(obs::Vector{Int})
    return [begin v = zeros(N_OBS); v[o] = 1.0; v end for o in obs]
end

# ── Model ─────────────────────────────────────────────────────

@model function fixture_hmm(y, prior_A, prior_B)
    A ~ DirichletCollection(prior_A)
    B ~ DirichletCollection(prior_B)

    s_prev ~ Categorical(fill(1.0 / N_STATES, N_STATES))

    for t in eachindex(y)
        s[t] ~ DiscreteTransition(s_prev, A)
        y[t] ~ DiscreteTransition(s[t], B)
        s_prev = s[t]
    end
end

@constraints function fixture_hmm_constraints()
    q(s_prev, s, A, B) = q(s_prev, s)q(A)q(B)
end

# ── Training ──────────────────────────────────────────────────

mutable struct TrainedHMM
    name::String
    transition_counts::Matrix{Float64}  # Dirichlet pseudo-counts for A
    emission_counts::Matrix{Float64}    # Dirichlet pseudo-counts for B
    n_signals::Int
end

function make_default_hmm(name::String)
    A_prior = ones(N_STATES, N_STATES)
    B_prior = fill(EMISSION_SMOOTHING, N_STATES, N_OBS)
    return TrainedHMM(name, A_prior, B_prior, 0)
end

"""Train an HMM on one or more observation sequences. Returns updated pseudo-counts."""
function train_hmm!(hmm::TrainedHMM, sequences::Vector{Vector{Int}};
                    n_iter::Int=N_ITERATIONS)
    for obs in sequences
        y_onehot = obs_to_onehot(obs)

        init = @initialization begin
            q(A) = DirichletCollection(ones(N_STATES, N_STATES))
            q(B) = DirichletCollection(ones(N_STATES, N_OBS))
            q(s) = Categorical(fill(1.0 / N_STATES, N_STATES))
        end

        result = infer(
            model          = fixture_hmm(
                prior_A = hmm.transition_counts,
                prior_B = hmm.emission_counts,
            ),
            data           = (y = y_onehot,),
            constraints    = fixture_hmm_constraints(),
            initialization = init,
            returnvars     = (s = KeepLast(), A = KeepLast(), B = KeepLast()),
            iterations     = n_iter,
            free_energy    = true,
            options        = (limit_stack_depth = 500,),
        )

        # Update pseudo-counts from posteriors
        A_post = mean(result.posteriors[:A])
        B_post = mean(result.posteriors[:B])

        # Accumulate: posterior counts become next prior
        hmm.transition_counts .= A_post .* N_STATES  # scale back to pseudo-counts
        hmm.emission_counts .= B_post .* N_OBS
        hmm.n_signals += 1

        fe = result.free_energy[end]
        println("  Signal trained: free_energy=$(round(fe, digits=2))")
    end
    return hmm
end

"""Score a single observation sequence against a trained HMM. Returns free energy."""
function score_hmm(hmm::TrainedHMM, obs::Vector{Int}; n_iter::Int=N_ITERATIONS)
    y_onehot = obs_to_onehot(obs)

    init = @initialization begin
        q(A) = DirichletCollection(ones(N_STATES, N_STATES))
        q(B) = DirichletCollection(ones(N_STATES, N_OBS))
        q(s) = Categorical(fill(1.0 / N_STATES, N_STATES))
    end

    result = infer(
        model          = fixture_hmm(
            prior_A = hmm.transition_counts,
            prior_B = hmm.emission_counts,
        ),
        data           = (y = y_onehot,),
        constraints    = fixture_hmm_constraints(),
        initialization = init,
        returnvars     = (s = KeepLast(),),
        iterations     = n_iter,
        free_energy    = true,
        options        = (limit_stack_depth = 500,),
    )

    fe = result.free_energy[end]
    state_posteriors = result.posteriors[:s]
    labels = [argmax(probvec(sp)) for sp in state_posteriors]

    return fe, labels
end

# ── Classification ────────────────────────────────────────────

"""
    classify_raw(obs, models_dict)

Classify a discretized observation sequence by scoring against all models.
Returns (predicted_type, free_energies_dict).
"""
function classify_raw(obs::Vector{Int}, models::Dict{String, TrainedHMM})
    free_energies = Dict{String, Float64}()
    for (name, hmm) in models
        fe, _ = score_hmm(hmm, obs; n_iter=10)
        free_energies[name] = fe
    end
    predicted = argmin(free_energies)
    return first(predicted), free_energies
end

# ── Serialization ─────────────────────────────────────────────

function save_models(models::Dict{String, TrainedHMM}, path::String)
    data = Dict{String, Any}()
    for (name, hmm) in models
        data[name] = Dict(
            "name" => hmm.name,
            "transition_counts" => hmm.transition_counts,
            "emission_counts" => hmm.emission_counts,
            "n_signals" => hmm.n_signals,
        )
    end
    data["_meta"] = Dict(
        "n_states" => N_STATES,
        "n_obs" => N_OBS,
        "n_freq_bins" => N_FREQ_BINS,
        "n_vib_bins" => N_VIB_BINS,
    )
    open(path, "w") do f
        JSON.print(f, data, 2)
    end
    println("Saved $(length(models)) model(s) to $path")
end

function load_models(path::String)
    raw = JSON.parsefile(path)
    models = Dict{String, TrainedHMM}()
    for (key, val) in raw
        key == "_meta" && continue
        models[key] = TrainedHMM(
            val["name"],
            Float64.(hcat(val["transition_counts"]...)'),
            Float64.(hcat(val["emission_counts"]...)'),
            val["n_signals"],
        )
    end
    return models
end

# ── Entry point ───────────────────────────────────────────────

println("Raw signal fixture model loaded.")
println("Available functions:")
println("  train_hmm!(hmm, sequences)         — train on observation sequences")
println("  score_hmm(hmm, obs)                 — score a sequence, returns (FE, states)")
println("  classify_raw(obs, models)            — classify by lowest free energy")
println("  save_models(models, path)            — save to JSON")
println("  load_models(path)                    — load from JSON")
