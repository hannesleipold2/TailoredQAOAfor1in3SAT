using SparseArrays
using LinearAlgebra
using KrylovKit
using MAT
using Glob
using Random
using Optim 
using JSON 
# using StructTypes

import Unmarshal

include("bitvec_conversions_cuda.jl")
include("sat_problem_cuda.jl")
using .SatGenerator
include("measures_n_reps_cuda.jl")
include("../tailor_space.jl")
include("init_operators_cuda.jl")
include("apply_operators_cuda.jl")
include("training_cuda.jl")
# include("qaoa_circ.jl")


function simple_alphabeta()
    p_rounds    = 10
    pi_co       = pi / 2
    alpha_co    = 1/2
    beta_co     = 1/8
    alphas      = [ alpha_co * pi_co * (p/p_rounds)         for p = 1 : p_rounds ]
    betas       = [ beta_co * pi_co * (1 - (p/p_rounds))    for p = 1 : p_rounds ]
    return p_rounds, alphas, betas
end

function run_t_qaoa(wave_func, costop_vec, U_clause_mixers, p_rounds, alphas, betas, sol_vecs, num_bits, num_states)
    copy_wave_func = copy(wave_func)
    for p = 1 : p_rounds
        fin_sup = calc_sol_support(sol_vecs, wave_func)
        fin_eng = CUDA.dot(wave_func, costop_vec * wave_func )
        #fin_eng = dot(wave_func, wave_func .* costop_vec)
        println(string("SOLUTION SUPPORT: \t", string(fin_sup+0.00001)[1:5]))
        println(string("EXP COST: \t\t", string(fin_eng + 0.00001)[1:5]))
        println(string("NORM: \t\t\t", string(norm(wave_func) + 0.00001)[1:5]))
        phase_energy!(wave_func, costop_vec, alphas[ p ])
        wave_func = apply_all_clause_mixers(wave_func, copy_wave_func, U_clause_mixers, num_bits, betas[ p ])
    end

    fin_sup = calc_sol_support(sol_vecs, wave_func)
    fin_eng = CUDA.dot(wave_func, costop_vec * wave_func )
    #fin_eng = dot(wave_func, wave_func .* costop_vec)
    println()
    println(string("SOLUTION SUPPORT: \t", string(fin_sup+0.00001)[1:5]))
    println(string("EXP COST: \t\t", string(fin_eng + 0.00001)[1:5]))
    println(string("NORM: \t\t\t", string(norm(wave_func)+0.00001)[1:5]))
    return fin_eng, fin_sup
end

function simple_run_t_qaoa(new_sat_prob)
    p_rounds, alphas, betas = simple_alphabeta()
    ### READ ###
    num_bits        = new_sat_prob.num_red_variables
    red_sols        = new_sat_prob.red_solutions
    vars_per_clause = new_sat_prob.vars_per_clause 
    clauses         = new_sat_prob.red_clauses
    dis_clauses     = new_sat_prob.red_max_disjoint_clauses
    uncov_vars      = new_sat_prob.red_variables_uncovered
    ### DEFI ###
    num_states      = 2^(num_bits)
    sol_vecs        = init_sol_space(red_sols, num_states)
    reds_to_acts    = find_tailor_subspace(new_sat_prob)        # TAILOR SPECIFIC
    sols_per_clause = find_sols_per_clause(num_bits, dis_clauses)
    # println(reds_to_acts)
    # println(find_num_tailor_states(dis_clauses, uncov_vars))
    # println(length(reds_to_acts))
    # throw(DomainError)
    wave_func       = init_t_wavefunc(num_states, reds_to_acts) 
    costop_vec      = init_cost_oper(num_bits, clauses)
    U_clause_mixers = init_clause_mixers(num_bits, reds_to_acts, sols_per_clause, uncov_vars)
    ### CALL ###
    return run_t_qaoa(wave_func, costop_vec, U_clause_mixers, p_rounds, alphas, betas, sol_vecs, num_bits, num_states)
end

function run_ut_qaoa_memcon(new_sat_prob)
    return 0 
end

function run_ut_qaoa(given_wave_func, costop_vec, U_xmixers, p_rounds, alphas, betas, sol_vecs, num_bits, num_states, DO_PRINT=0)
    wave_func       = copy(given_wave_func)
    copy_wave_func  = copy(wave_func)
    for p = 1 : p_rounds
        fin_sup = calc_sol_support(sol_vecs, wave_func)
        #cpu_intermediete = wave_func .* costop_vec
        fin_eng = CUDA.dot(wave_func, costop_vec * wave_func )
        if DO_PRINT == 1
            println(string("SOLUTION SUPPORT: \t", string(fin_sup+0.00001)[1:5]))
            println(string("EXP COST: \t\t", string(fin_eng + 0.00001)[1:5]))
            println(string("NORM: \t\t\t", string(norm(wave_func) + 0.00001)[1:5]))
        end
        phase_energy!(wave_func, costop_vec, alphas[ p ])
        wave_func = apply_all_xmixers(wave_func, copy_wave_func, U_xmixers, num_bits, betas[ p ])
    end

    fin_sup = calc_sol_support(sol_vecs, wave_func)
    #fin_eng = dot(wave_func, wave_func .* costop_vec)
    fin_eng = abs(CUDA.dot(wave_func, costop_vec * wave_func ))
    #fin_eng = abs(dot(wave_func, wave_func .* costop_vec))
    if DO_PRINT == 1
        println()
        println(string("SOLUTION SUPPORT: \t", string(fin_sup+0.00001)[1:5]))
        println(string("EXP COST: \t\t", string(fin_eng + 0.00001)[1:5]))
        println(string("NORM: \t\t\t", string(norm(wave_func)+0.00001)[1:5]))
    end
    return fin_eng, fin_sup
end

function simple_run_ut_qaoa(new_sat_prob)
    p_rounds, alphas, betas = simple_alphabeta()
    ### READ ###
    num_bits        = new_sat_prob.num_red_variables
    red_sols        = new_sat_prob.red_solutions
    vars_per_clause = new_sat_prob.vars_per_clause 
    clauses         = new_sat_prob.red_clauses
    ### DEFI ###
    num_states      = 2^(num_bits)
    sol_vecs        = init_sol_space(red_sols, num_states)  
    wave_func       = init_ut_wavefunc(num_states) 
    costop_vec      = init_cost_oper(num_bits, clauses)
    U_xmixers       = init_xmixers(num_bits)
    ### CALL ###
    return run_ut_qaoa(wave_func, costop_vec, U_xmixers, p_rounds, alphas, betas, sol_vecs, num_bits, num_states)
end


function run_general_qaoa(nbits, clen, mclauses, kinsts)
    upp_dir_str = string("./sat_1in", clen, "/")
    dir_str = string(upp_dir_str, "rand_insts", "_nbits=", nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "/")
    open(string(dir_str,"inst_", 1,".json"), "r") do f 
        json_string = JSON.read(f, String)
        new_sat_prob = Unmarshal.unmarshal(SatProblem, JSON.parse(json_string))
        # simple_run_ut_qaoa(new_sat_prob)
        simple_run_t_qaoa(new_sat_prob)
    end
    return 0 
end

function train_general_qaoa(nbits, clen, mclauses, kinsts)
    upp_dir_str = string("./sat_1in", clen, "/")
    dir_str = string(upp_dir_str, "rand_insts", "_nbits=", nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "/")
    new_sat_probs = Array{SatProblem, 1}()
    for i = 1 : kinsts
        open(string(dir_str,"inst_", 1,".json"), "r") do f 
            json_string = JSON.read(f, String)
            new_sat_prob = Unmarshal.unmarshal(SatProblem, JSON.parse(json_string))
            push!(new_sat_probs, new_sat_prob)
        end
    end
    #simple_run_ut_qaoa(new_sat_probs[1])
    train_ut_qaoa(new_sat_probs)
    return 0 
end