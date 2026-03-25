# ═══════════════════════════════════════════════════════════════
# Water Fixture Classification — RxInfer Generative Model
# ═══════════════════════════════════════════════════════════════
#
# FEATURES USED FOR CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
#
# The model uses 9 features per event:
#
# Frequency features (vibration-gated):
#   duration_seconds        — total event duration
#   mean_plateau_freq_hz    — mean frequency in middle 60% of event
#                             (only readings with vibration ≥ 5)
#   estimated_cycles        — integrated freq×time (gated)
#   ramp_duration_seconds   — time to reach 80% of peak freq
#   peak_freq_hz            — max frequency (real signal only)
#   freq_stability          — coefficient of variation of plateau freq
#
# Vibration intensity features:
#   mean_vibration_intensity — mean band_energy_10s across all readings
#   signal_fraction          — fraction of readings above vibration
#                              threshold (≥5). Low = mostly noise.
#   mean_plateau_vibration   — mean vibration in middle 60% window
#                              with real signal. Characterises the
#                              "strength" of the fixture's flow.
#
# Why vibration matters:
#   dominant_freq_hz alone can't distinguish real cog rotation from
#   noise. When vibration intensity is low (< 5), the frequency
#   reading is just measuring background noise. The signal_fraction
#   feature tells the model what proportion of the event had real
#   detectable flow. Different fixtures have different vibration
#   profiles — a toilet flush has high vibration for 30-60s, while
#   a slow sink tap may have lower sustained vibration.
#
# ═══════════════════════════════════════════════════════════════
#
# TUNABLE PARAMETERS — read this before modifying the model
# ═══════════════════════════════════════════════════════════════
#
# OBSERVATION_NOISE_SCALE
#   Multiplier on the std of each feature's Gaussian likelihood.
#   Default: 1.0 (use std exactly as computed from data)
#   Increase (e.g. 2.0): model becomes more tolerant of unusual
#     signals — wider acceptance region per fixture type.
#     Use this when you get too many false negatives.
#   Decrease (e.g. 0.5): model becomes stricter — only signals
#     very close to the learned mean are accepted.
#     Use this when two fixture types are being confused.
#
# PRIOR_CONFIDENCE
#   Precision multiplier on the fixture type Categorical prior.
#   Default: 1.0 (uniform prior across fixture types)
#   Increase: model is more stubborn, needs more evidence to change
#     its fixture type belief. Good when priors are well-established.
#   Decrease: model updates more easily from a single signal.
#     Good in early stages with little data.
#
# FORGETTING_FACTOR
#   Used in online update mode. Between 0.0 and 1.0.
#   Default: 0.95
#   Controls how much past observations are discounted when new
#   data arrives. 1.0 = no forgetting (all history weighted equally).
#   0.9 = recent observations matter more than old ones.
#   Lower values make the model adapt faster to drift but less stable.
#   Only applies when calling online_update!().
#
# MIN_EVENTS_FOR_RELIABLE_PRIOR
#   If a fixture type has fewer events than this in the priors file,
#   the model prints a warning when classifying that type.
#   Default: 5
#   This is informational only — does not change model behaviour.
#
# ═══════════════════════════════════════════════════════════════
#
# FREE ENERGY — what it tells you
#
# free_energy is returned by every classify() call.
# It measures how surprised the model was by the signal.
#
# Low free energy (e.g. < 5.0):
#   The signal matched the model's expectations well.
#   High confidence classification.
#
# Medium free energy (5.0 – 15.0):
#   Some mismatch. The classification is probably right but
#   the signal had unusual characteristics.
#
# High free energy (> 15.0):
#   The model was very surprised. Either:
#   a) The fixture type is not in the model yet
#   b) The fixture's behaviour has drifted from its prior
#   c) Two fixtures were running simultaneously
#   d) The signal window was mislabeled
#
# Threshold values above are approximate starting points.
# After collecting more data, inspect your free energy
# distribution per fixture type and adjust thresholds to
# match your actual signal characteristics.
#
# ═══════════════════════════════════════════════════════════════

using RxInfer
using JSON
using CSV
using DataFrames
using LinearAlgebra

# ── Parameters ────────────────────────────────────────────────

const OBSERVATION_NOISE_SCALE = 1.0
const PRIOR_CONFIDENCE = 1.0
const FORGETTING_FACTOR = 0.95
const MIN_EVENTS_FOR_RELIABLE_PRIOR = 5

const FEATURE_NAMES = [
    "duration_seconds",
    "mean_plateau_freq_hz",
    "estimated_cycles",
    "ramp_duration_seconds",
    "peak_freq_hz",
    "freq_stability",
    "mean_vibration_intensity",
    "signal_fraction",
    "mean_plateau_vibration",
    "mean_autocorr_lag1",
    "mean_autocorr_lag5",
    "mean_zero_crossing_rate",
    "mean_peak_to_peak",
]

# ── Prior loading ─────────────────────────────────────────────

function resolve_prior(feature_dict)
    # If heuristic_override is present and not null, use it
    if haskey(feature_dict, "heuristic_override") && feature_dict["heuristic_override"] !== nothing
        override = feature_dict["heuristic_override"]
        return (mean=Float64(override["mean"]), std=Float64(override["std"]))
    end
    return (mean=Float64(feature_dict["mean"]), std=Float64(feature_dict["std"]))
end

function load_priors(path::String)
    raw = JSON.parsefile(path)
    priors = Dict{String, Dict{String, NamedTuple{(:mean, :std), Tuple{Float64, Float64}}}}()
    n_events = Dict{String, Int}()

    for (ftype, fdata) in raw
        priors[ftype] = Dict{String, NamedTuple{(:mean, :std), Tuple{Float64, Float64}}}()
        n_events[ftype] = get(fdata, "n_events", 0)

        for feat in FEATURE_NAMES
            if haskey(fdata, feat)
                priors[ftype][feat] = resolve_prior(fdata[feat])
            else
                priors[ftype][feat] = (mean=0.0, std=1.0)
            end
        end
    end
    return priors, n_events
end

# ── Model definition ─────────────────────────────────────────
# Uses the NormalMixture/Categorical pattern from the Gaussian
# Mixture example in this repo.

@model function fixture_model(observed, means, precisions, n_types)
    # Prior over fixture types — uniform scaled by PRIOR_CONFIDENCE
    fixture_type ~ Categorical(fill(PRIOR_CONFIDENCE / n_types, n_types))

    # Each feature is generated by a Gaussian mixture component
    # selected by fixture_type
    for i in eachindex(observed)
        observed[i] ~ NormalMixture(switch = fixture_type, m = means[i], p = precisions[i])
    end
end

# ── Build model parameters from priors ────────────────────────

function build_model_params(priors::Dict, fixture_types::Vector{String})
    n_types = length(fixture_types)
    n_features = length(FEATURE_NAMES)

    # means[feature_idx] = Vector of means, one per fixture type
    # precisions[feature_idx] = Vector of precisions, one per fixture type
    means = Vector{Vector{Float64}}(undef, n_features)
    precisions = Vector{Vector{Float64}}(undef, n_features)

    for (fi, feat) in enumerate(FEATURE_NAMES)
        m_vec = Float64[]
        p_vec = Float64[]
        for ftype in fixture_types
            prior = priors[ftype][feat]
            push!(m_vec, prior.mean)
            scaled_std = prior.std * OBSERVATION_NOISE_SCALE
            push!(p_vec, 1.0 / (scaled_std^2))
        end
        means[fi] = m_vec
        precisions[fi] = p_vec
    end

    return means, precisions
end

# ── Classification ────────────────────────────────────────────

"""
    classify(features::Dict, priors_path="priors_general.json")

Classify a single event. `features` is a Dict with keys matching FEATURE_NAMES.
Returns (predicted_type, posterior_probs, free_energy).
"""
function classify(features::Dict, priors_path::String="priors_general.json")
    priors, n_events_map = load_priors(priors_path)
    fixture_types = sort(collect(keys(priors)))
    n_types = length(fixture_types)

    # Warn about low-data types
    for ftype in fixture_types
        n = get(n_events_map, ftype, 0)
        if n < MIN_EVENTS_FOR_RELIABLE_PRIOR
            @warn "Fixture type '$ftype' has only $n event(s) — prior may be unreliable"
        end
    end

    means, precisions = build_model_params(priors, fixture_types)

    # Observed feature vector
    obs = [Float64(get(features, feat, 0.0)) for feat in FEATURE_NAMES]

    # Initialization
    init = @initialization begin
        q(fixture_type) = Categorical(fill(1.0 / n_types, n_types))
    end

    result = infer(
        model = fixture_model(means=means, precisions=precisions, n_types=n_types),
        data = (observed = obs,),
        constraints = MeanField(),
        initialization = init,
        iterations = 20,
        free_energy = true,
    )

    # Extract posterior
    posterior = result.posteriors[:fixture_type]
    probs = probvec(posterior)
    best_idx = argmax(probs)
    predicted_type = fixture_types[best_idx]
    fe = result.free_energy[end]

    return predicted_type, probs, fe
end

"""
    batch_classify(events_csv_path, priors_path="priors_general.json")

Classify all events in events.csv. Prints results table and accuracy.
"""
function batch_classify(events_csv_path::String="events.csv",
                        priors_path::String="priors_general.json")
    df = CSV.read(events_csv_path, DataFrame)
    priors, _ = load_priors(priors_path)
    fixture_types = sort(collect(keys(priors)))

    results = DataFrame(
        signal_id = Int[],
        true_label = String[],
        predicted = String[],
        confidence = Float64[],
        free_energy = Float64[],
        correct = Bool[],
    )

    for row in eachrow(df)
        features = Dict(feat => row[Symbol(feat)] for feat in FEATURE_NAMES)
        predicted, probs, fe = classify(features, priors_path)
        conf = maximum(probs)
        push!(results, (
            signal_id = row.signal_id,
            true_label = row.fixture_type,
            predicted = predicted,
            confidence = round(conf, digits=3),
            free_energy = round(fe, digits=2),
            correct = predicted == row.fixture_type,
        ))
    end

    println("\n" * "="^70)
    println(" Classification Results — $(basename(priors_path))")
    println("="^70)
    println(results)

    n_correct = sum(results.correct)
    n_total = nrow(results)
    acc = n_total > 0 ? round(n_correct / n_total * 100, digits=1) : 0.0
    println("\nAccuracy: $n_correct / $n_total ($acc%)")

    return results
end

# ── Online update ─────────────────────────────────────────────

"""
    online_update!(features, true_label, priors_path="priors_general.json")

Update priors for `true_label` using Welford's running mean/variance.
Applies FORGETTING_FACTOR. Does NOT override heuristic_override values.
"""
function online_update!(features::Dict, true_label::String,
                        priors_path::String="priors_general.json")
    raw = JSON.parsefile(priors_path)

    if !haskey(raw, true_label)
        raw[true_label] = Dict("n_events" => 0)
        for feat in FEATURE_NAMES
            raw[true_label][feat] = Dict("mean" => 0.0, "std" => 1.0)
        end
    end

    entry = raw[true_label]
    n_old = get(entry, "n_events", 0)
    n_new = n_old + 1

    # Effective n with forgetting
    n_eff = n_old * FORGETTING_FACTOR + 1.0

    for feat in FEATURE_NAMES
        # Skip features with heuristic override
        if haskey(entry, feat) && isa(entry[feat], Dict)
            if haskey(entry[feat], "heuristic_override") && entry[feat]["heuristic_override"] !== nothing
                continue
            end
        end

        x = Float64(get(features, feat, 0.0))
        old_mean = Float64(get(entry[feat], "mean", 0.0))
        old_std = Float64(get(entry[feat], "std", 1.0))
        old_var = old_std^2

        if n_old == 0
            new_mean = x
            new_var = max(abs(x) * 0.3, 1.0)^2
        else
            # Welford-style with forgetting
            new_mean = old_mean + (x - old_mean) / n_eff
            new_var = old_var * FORGETTING_FACTOR + (x - old_mean) * (x - new_mean) / n_eff
        end

        entry[feat]["mean"] = round(new_mean, digits=4)
        entry[feat]["std"] = round(sqrt(max(new_var, 1e-8)), digits=4)
    end

    entry["n_events"] = n_new
    raw[true_label] = entry

    open(priors_path, "w") do f
        JSON.print(f, raw, 2)
    end

    println("Updated priors for '$true_label' (n=$n_new) in $priors_path")
end

# ── Model comparison ──────────────────────────────────────────

"""
    compare_models(events_csv_path="events.csv")

Run both general and heuristic models on all events.
Prints side-by-side comparison.
"""
function compare_models(events_csv_path::String="events.csv")
    df = CSV.read(events_csv_path, DataFrame)

    println("\n" * "="^80)
    println(" Model Comparison: General vs Heuristic")
    println("="^80)
    println()

    header = lpad("Signal", 8) * lpad("True", 10) * lpad("General", 10) *
             lpad("Heuristic", 12) * lpad("Winner", 10)
    println(header)
    println("-"^50)

    n_gen = 0
    n_heur = 0

    for row in eachrow(df)
        features = Dict(feat => row[Symbol(feat)] for feat in FEATURE_NAMES)

        pred_gen, _, _ = classify(features, "priors_general.json")
        pred_heur, _, _ = classify(features, "priors_heuristic.json")

        gen_correct = pred_gen == row.fixture_type
        heur_correct = pred_heur == row.fixture_type

        n_gen += gen_correct
        n_heur += heur_correct

        winner = if gen_correct && !heur_correct
            "general"
        elseif !gen_correct && heur_correct
            "heuristic"
        elseif gen_correct && heur_correct
            "both"
        else
            "neither"
        end

        println(lpad(string(row.signal_id), 8) *
                lpad(row.fixture_type, 10) *
                lpad(pred_gen, 10) *
                lpad(pred_heur, 12) *
                lpad(winner, 10))
    end

    n_total = nrow(df)
    println("\nGeneral:   $n_gen / $n_total correct")
    println("Heuristic: $n_heur / $n_total correct")
end

# ── Quick test ────────────────────────────────────────────────

println("Fixture classification model loaded.")
println("Available functions:")
println("  classify(features, priors_path)       — classify one event")
println("  batch_classify(csv_path, priors_path)  — classify all events")
println("  online_update!(features, label, path)  — update priors with new data")
println("  compare_models(csv_path)               — compare general vs heuristic")
