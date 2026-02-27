#!/usr/bin/env julia
#
# Flow Fingerprint — Inference-Only Server Script
#
# Loads a pre-trained model (flow_model.json) and runs streaming inference
# on new sensor data from Hasura. No RxInfer or batch training needed.
#
# Dependencies: HTTP, JSON3, Distributions, LinearAlgebra, Statistics
#
# Usage:
#   julia inference_server.jl                    # run once on latest data
#   julia inference_server.jl --watch 30         # poll every 30 seconds
#   julia inference_server.jl --model path.json  # custom model path
#

using HTTP, JSON3, LinearAlgebra, Statistics
import Distributions: Normal, pdf

# ── Parse CLI args ──────────────────────────────────────────────────────
function parse_args()
    args = Dict{String,Any}(
        "model"    => joinpath(@__DIR__, "flow_model.json"),
        "watch"    => 0,
        "sensor"   => "1",
    )
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--model" && i < length(ARGS)
            args["model"] = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--watch" && i < length(ARGS)
            args["watch"] = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--sensor" && i < length(ARGS)
            args["sensor"] = ARGS[i+1]; i += 2
        else
            i += 1
        end
    end
    return args
end

# ── Load model ──────────────────────────────────────────────────────────
function load_model(path::String)
    raw = JSON3.read(read(path, String))

    n_states = Int(raw.n_states)
    means    = Float64.(raw.learned_means)
    stds     = Float64.(raw.learned_stds)
    A        = vcat([Float64.(row)' for row in raw.learned_A]...)

    noise_floor = Float64(raw.noise_floor)
    n_signals   = Int(raw.n_signals)
    TLEN        = Int(raw.TLEN)
    overlap_thresh      = Float64(raw.overlap_thresh)
    min_dur_for_overlap = Int(raw.min_dur_for_overlap)

    cl = raw.clustering
    flo   = Float64.(cl.flo)
    fspan = Float64.(cl.fspan)
    order = Int.(cl.order)
    centroids = vcat([Float64.(c)' for c in cl.centroids]...)

    templates = Dict{Int, Vector{Float64}}()
    for (k, v) in pairs(raw.templates)
        templates[parse(Int, String(k))] = Float64.(v)
    end

    return (;
        n_states, means, stds, A, noise_floor, n_signals, TLEN,
        overlap_thresh, min_dur_for_overlap,
        flo, fspan, order, centroids, templates,
    )
end

# ── Hasura data fetching ────────────────────────────────────────────────
const HASURA_URL = "https://hasura.pipestuesday.org/v1/graphql"
const HASURA_ADMIN_SECRET = "PIPE_SUPERMMMSECRET_PIPE"

function fetch_sensor_data(sensor_id::String)
    query = """
    query {
      report(where: {sensor_id: {_eq: "$sensor_id"}}, order_by: {created_at: asc}) {
        id
        flow_value
        created_at
      }
    }
    """
    response = HTTP.post(
        HASURA_URL,
        ["Content-Type" => "application/json", "x-hasura-admin-secret" => HASURA_ADMIN_SECRET],
        JSON3.write(Dict("query" => query))
    )
    data = JSON3.read(String(response.body))
    reports = data.data.report
    flow = Float64[r.flow_value for r in reports]
    timestamps = [r.created_at for r in reports]
    return flow, timestamps
end

# ── Forward algorithm ───────────────────────────────────────────────────
function forward_step(alpha::Vector{Float64}, y::Float64,
                      A::Matrix{Float64}, μ::Vector{Float64}, σ::Vector{Float64})
    K = length(μ)
    α_new = [pdf(Normal(μ[k], σ[k]), y) * dot(alpha, @view A[:, k]) for k in 1:K]
    s = sum(α_new)
    return s > 0 ? α_new ./ s : fill(1.0/K, K)
end

# ── Event detection ─────────────────────────────────────────────────────
function detect_events(flow::Vector{Float64}, model)
    n_s = model.n_states
    baseline_set = Set(k for k in 1:n_s if abs(model.means[k]) < model.noise_floor)
    isempty(baseline_set) && push!(baseline_set, argmin(abs.(model.means)))

    alpha = fill(1.0/n_s, n_s)
    events = Vector{NamedTuple}()
    in_ev = false; ev_start = 0
    ev_flow = Float64[]; ev_st = Int[]

    for (t, y) in enumerate(flow)
        alpha = forward_step(alpha, y, model.A, model.means, model.stds)
        s = argmax(alpha)

        if !(s in baseline_set) && !in_ev
            in_ev = true; ev_start = t
            ev_flow = [y]; ev_st = [s]
        elseif !(s in baseline_set) && in_ev
            push!(ev_flow, y); push!(ev_st, s)
        elseif (s in baseline_set) && in_ev
            in_ev = false
            dur = length(ev_flow)
            mi = argmin(ev_flow)
            push!(events, (
                start=ev_start, stop=t-1,
                flow=copy(ev_flow), states=copy(ev_st),
                depth=minimum(ev_flow), duration=dur,
                volume=sum(abs, ev_flow),
                mean_flow=mean(ev_flow),
                rise_frac=mi/dur,
            ))
        end
    end
    if in_ev
        dur = length(ev_flow); mi = argmin(ev_flow)
        push!(events, (
            start=ev_start, stop=length(flow),
            flow=copy(ev_flow), states=copy(ev_st),
            depth=minimum(ev_flow), duration=dur,
            volume=sum(abs, ev_flow), mean_flow=mean(ev_flow),
            rise_frac=mi/dur,
        ))
    end
    return events
end

# ── Event classification using saved centroids ──────────────────────────
function classify_events(events, model)
    N_ev = length(events)
    N_ev == 0 && return Int[]

    F = zeros(N_ev, 4)
    for (i, ev) in enumerate(events)
        F[i, :] = [ev.depth, log(max(1, ev.duration)), ev.volume, ev.rise_frac]
    end

    # Normalize using saved normalization params
    Fn = (F .- model.flo') ./ model.fspan'

    # Assign to nearest centroid
    labels = [argmin([sum((Fn[i, :] .- model.centroids[c, :]).^2)
                      for c in 1:model.n_signals]) for i in 1:N_ev]

    # Apply depth-sort remap
    remap = Dict(model.order[i] => i for i in 1:model.n_signals)
    return [remap[l] for l in labels]
end

# ── Template matching + overlap detection ───────────────────────────────
function resample(sig, n)
    [sig[clamp(round(Int, t * length(sig) / n), 1, length(sig))] for t in 1:n]
end

function xcorr(a, b)
    az, bz = a .- mean(a), b .- mean(b)
    d = sqrt(sum(az.^2) * sum(bz.^2))
    d > 0 ? sum(az .* bz) / d : 0.0
end

function check_overlaps(events, labels, model)
    flagged = Int[]
    for (i, ev) in enumerate(events)
        ev.duration < model.min_dur_for_overlap && continue
        haskey(model.templates, labels[i]) || continue
        score = xcorr(resample(ev.flow, model.TLEN), model.templates[labels[i]])
        score < model.overlap_thresh && push!(flagged, i)
    end
    return flagged
end

# ── Main pipeline ───────────────────────────────────────────────────────
function run_inference(model, sensor_id::String)
    println("\n", "="^60)
    println("Fetching data for sensor $sensor_id...")
    flow, timestamps = fetch_sensor_data(sensor_id)
    println("  $(length(flow)) readings ($(first(timestamps)) to $(last(timestamps)))")

    println("Running streaming inference...")
    t0 = time()
    events = detect_events(flow, model)
    labels = classify_events(events, model)
    flagged = check_overlaps(events, labels, model)
    dt = time() - t0

    println("  $(length(events)) events detected in $(round(dt * 1000, digits=1)) ms")
    println("  $(model.n_signals) signal types, $(length(flagged)) potential overlaps")

    println("\nEvents by type:")
    for c in 1:model.n_signals
        mask = labels .== c
        evs = events[mask]
        isempty(evs) && continue
        avg_depth = round(mean(e.depth for e in evs), sigdigits=3)
        avg_dur   = round(mean(e.duration for e in evs), digits=1)
        println("  signal_$c: $(length(evs)) events (depth ≈ $avg_depth, duration ≈ $avg_dur steps)")
    end

    if !isempty(flagged)
        println("\nPotential overlaps:")
        for i in flagged
            ev = events[i]
            println("  event $i: t=$(ev.start)–$(ev.stop), depth=$(round(ev.depth, sigdigits=3)), type=signal_$(labels[i])")
        end
    end

    return events, labels, flagged
end

# ── Entry point ─────────────────────────────────────────────────────────
function main()
    args = parse_args()

    println("Loading model from: $(args["model"])")
    model = load_model(args["model"])
    println("  $(model.n_states) HMM states, $(model.n_signals) signal types")

    if args["watch"] > 0
        println("Watch mode: polling every $(args["watch"]) seconds (Ctrl-C to stop)")
        while true
            try
                run_inference(model, args["sensor"])
            catch e
                println("Error: $e")
            end
            sleep(args["watch"])
        end
    else
        run_inference(model, args["sensor"])
    end
end

main()
