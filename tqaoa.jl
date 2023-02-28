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

include("base_structs.jl")
include("bitvec_conversions.jl")
include("sat_problem.jl")
using .SatGenerator
include("measures_n_reps.jl")
include("tailor_space.jl")
include("init_operators.jl")
include("apply_operators.jl")
include("training.jl")
# include("qaoa_circ.jl")


function simple_alphabeta(pdepth=10)
	p_rounds    = pdepth
	pi_co		= pi / 2
	alpha_co	= 1/2
	beta_co		= 1/8
	alphas      = [ alpha_co * pi_co * (p/p_rounds)			for p = 1 : p_rounds ]
	betas       = [ beta_co * pi_co * (1 - (p/p_rounds)) 	for p = 1 : p_rounds ]
	return p_rounds, alphas, betas
end

function run_t_qaoa(given_wave_func, costop_vec, U_clause_mixers, reds_to_acts, p_rounds, alphas, betas, sol_vecs, num_bits, num_states, DO_PRINT=0)
	wave_func 		= copy(given_wave_func)
	copy_wave_func 	= copy(wave_func)
	for p = 1 : p_rounds
		if DO_PRINT == 1
			fin_sup = calc_sol_support(sol_vecs, wave_func)
			fin_eng = dot(wave_func, wave_func .* costop_vec)
			println(string("SOLUTION SUPPORT: \t", string(fin_sup+0.00001)[1:5]))
			println(string("EXP COST: \t\t", string(fin_eng + 0.00001)[1:5]))
			println(string("NORM: \t\t\t", string(norm(wave_func) + 0.00001)[1:5]))
		end
		phase_subspace_energy!(wave_func, costop_vec, reds_to_acts, alphas[ p ])
		# println(string("NORM: \t\t\t", string(norm(wave_func) + 0.00001)[1:5]))
		wave_func = apply_all_clause_mixers(wave_func, copy_wave_func, U_clause_mixers, num_bits, betas[ p ])
	end

	fin_sup = calc_sol_support(sol_vecs, wave_func)
	fin_eng = abs(dot(wave_func, wave_func .* costop_vec))
	if DO_PRINT == 1
		println()
		println(string("SOLUTION SUPPORT: \t", string(fin_sup+0.00001)[1:5]))
		println(string("EXP COST: \t\t", string(fin_eng + 0.00001)[1:5]))
		println(string("NORM: \t\t\t", string(norm(wave_func)+0.00001)[1:5]))
	end
	if norm(wave_func) < 0.99
		println("RUN NORM DECAY")
		throw(DomainError)
	end
	return fin_eng, fin_sup
end

function simple_run_t_qaoa(new_sat_prob, p_rounds=10, alphas=Array{Float64, 1}(), betas=Array{Float64, 1}())
	if length(alphas) == 0
		p_rounds, alphas, betas = simple_alphabeta(p_rounds)
	end
	### READ ###
	num_bits        = new_sat_prob.num_red_variables
	red_sols        = new_sat_prob.red_solutions
	vars_per_clause = new_sat_prob.vars_per_clause 
	clauses         = new_sat_prob.red_clauses
	dis_clauses 	= new_sat_prob.red_max_disjoint_clauses
	rem_clauses 	= new_sat_prob.red_remaining_clauses 
	uncov_vars		= new_sat_prob.red_variables_uncovered
	### DEFI ###
	num_states      = 2^(num_bits)
	# sol_vecs 		= init_full_sol_space(red_sols, num_states)
	sol_vecs 		= init_sol_space(red_sols, num_states)
	reds_to_acts	= find_tailor_subspace(new_sat_prob)		# TAILOR SPECIFIC
	sols_per_clause = find_sols_per_clause(num_bits, dis_clauses)
	wave_func 		= init_t_wavefunc(num_states, reds_to_acts)
	costop_vec  	= init_subspace_cost_oper(num_bits, rem_clauses, reds_to_acts)
	U_clause_mixers	= init_clause_mixers(num_bits, reds_to_acts, sols_per_clause, uncov_vars)
	### CALL ###
	ret_eng, ret_sup = run_t_qaoa(wave_func, costop_vec, U_clause_mixers, reds_to_acts, p_rounds, alphas, betas, sol_vecs, num_bits, num_states)
	return ret_eng, ret_sup, length(reds_to_acts)
end

function run_ut_qaoa_memcon(new_sat_prob)
	return 0 
end

function run_ut_qaoa(given_wave_func, costop_vec, U_xmixers, p_rounds, alphas, betas, sol_vecs, num_bits, num_states, DO_PRINT=0)
	wave_func		= copy(given_wave_func)
	copy_wave_func 	= copy(wave_func)
	for p = 1 : p_rounds
		fin_sup = calc_sol_support(sol_vecs, wave_func)
		fin_eng = dot(wave_func, wave_func .* costop_vec)
		if DO_PRINT == 1
			println(string("SOLUTION SUPPORT: \t", string(fin_sup+0.00001)[1:5]))
			println(string("EXP COST: \t\t", string(fin_eng + 0.00001)[1:5]))
			println(string("NORM: \t\t\t", string(norm(wave_func) + 0.00001)[1:5]))
		end
		phase_energy!(wave_func, costop_vec, alphas[ p ])
		wave_func = apply_all_xmixers(wave_func, copy_wave_func, U_xmixers, num_bits, betas[ p ])
	end

	fin_sup = calc_sol_support(sol_vecs, wave_func)
	fin_eng = abs(dot(wave_func, wave_func .* costop_vec))
	if DO_PRINT == 1
		println()
		println(string("SOLUTION SUPPORT: \t", string(fin_sup+0.00001)[1:5]))
		println(string("EXP COST: \t\t", string(fin_eng + 0.00001)[1:5]))
		println(string("NORM: \t\t\t", string(norm(wave_func)+0.00001)[1:5]))
	end
	return fin_eng, fin_sup
end

function simple_run_ut_qaoa(new_sat_prob, p_rounds=10, alphas=Array{Float64, 1}(), betas=Array{Float64, 1}())
	if length(alphas) == 0
		p_rounds, alphas, betas = simple_alphabeta(p_rounds)
	end
	### READ ###
	num_bits        = new_sat_prob.num_red_variables
	red_sols        = new_sat_prob.red_solutions
	vars_per_clause = new_sat_prob.vars_per_clause 
	clauses         = new_sat_prob.red_clauses
	### DEFI ###
	num_states      = 2^(num_bits)
	sol_vecs 		= init_sol_space(red_sols, num_states)	
	wave_func 		= init_ut_wavefunc(num_states) 
	costop_vec  	= init_cost_oper(num_bits, clauses)
	U_xmixers   	= init_xmixers(num_bits)
	### CALL ###
	ret_eng, ret_sup = run_ut_qaoa(wave_func, costop_vec, U_xmixers, p_rounds, alphas, betas, sol_vecs, num_bits, num_states) 
	return ret_eng, ret_sup, num_states
end


function run_general_qaoa(nbits, clen, mclauses, kinsts)
	upp_dir_str = string("./sat_1in", clen, "/")
	dir_str = string(upp_dir_str, "rand_insts", "_nbits=", nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "/")
	open(string(dir_str,"inst_", 1,".json"), "r") do f 
		json_string = JSON.read(f, String)
		new_sat_prob = Unmarshal.unmarshal(SatProblem, JSON.parse(json_string))
		simple_run_t_qaoa(new_sat_prob)
	end
	return 0 
end

function train_and_run_qaoa(nbits, clen, mclauses, kinsts, RUN_OPT=1)
	upp_dir_str = string("./sat_1in", clen, "/")
	dir_str = string(upp_dir_str, "rand_insts", "_nbits=", nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "/")
	new_sat_probs = Array{SatProblem, 1}()
	for i = 1 : kinsts
		open(string(dir_str,"inst_", i,".json"), "r") do f 
			json_string = JSON.read(f, String)
			new_sat_prob = Unmarshal.unmarshal(SatProblem, JSON.parse(json_string))
			push!(new_sat_probs, new_sat_prob)
		end
	end
	println(RUN_OPT == 1, typeof(RUN_OPT))
	if 		RUN_OPT == 1
		train_ut_qaoa(new_sat_probs)
	elseif 	RUN_OPT == 2
		train_t_qaoa(new_sat_probs)
	elseif 	RUN_OPT == 3
		simple_run_ut_qaoa(new_sat_probs[ 1 ])
	elseif 	RUN_OPT == 4
		simple_run_t_qaoa(new_sat_probs[ 1 ])
	end
	return 0 
end

function load_and_run_qaoa(train_nbits, pdepth, all_nbits, clen, mclauses, kinsts, kmin, kmax, APP_OPT=1)
	app_str 	= APP_OPT == 1 ? "_ut" : "_t"
	### LOAD TRAIN ###
	train_dir 	= string("./train_angles", app_str, "_clen=", clen, "/")
	train_str 	= string("rand_insts", "_nbits=", train_nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "_pdepth=", pdepth)
	train_path 	= string(train_dir, train_str, ".json")
	
	json_string = string()
	open(train_path, "r") do f
		json_string = JSON.read(f, String)
	end 
	angle_params 	= Unmarshal.unmarshal(AngleParams, JSON.parse(json_string))
	alphas 			= angle_params.alphas 
	betas 			= angle_params.betas 
	println(alphas)
	println(betas)
	### SET UP LOAD/SAVE ###
	upp_out_dir = string("./results_qaoa", app_str, "_clen=", clen , "_pdepth=", pdepth, "/")
	if !isdir(upp_out_dir)
		mkdir(upp_out_dir)
	end
	upp_indir_str 	= string("./sat_1in", clen, "/")
	
	### OPEN INSTS ###
	start_time = Dates.now()
	for nbit_id = 1 : length(all_nbits)
		nbits 			= all_nbits[ nbit_id ]
		println(nbits, " Runtime: ", (Dates.now() - start_time).value/1000, "s")
		mclauses 		= Int(ceil(nbits/clen))
		indir_str 		= string(upp_indir_str, "rand_insts", "_nbits=", nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "/")
		new_sat_probs = Array{SatProblem, 1}()
		for i = kmin : kmax
			open(string(indir_str,"inst_", i,".json"), "r") do f 
				json_string = JSON.read(f, String)
				new_sat_prob = Unmarshal.unmarshal(SatProblem, JSON.parse(json_string))
				push!(new_sat_probs, new_sat_prob)
			end
			out_dir		= string(upp_out_dir, "inst_nbits=", nbits, "/")
			if !isdir(out_dir)
				mkdir(out_dir)
			end		
			out_file_str 		= string("rand_inst_", i, ".json")
			fin_eng, fin_sup	= [ 0, 0, 0 ]
			if APP_OPT == 1
				fin_eng, fin_sup, size 	= simple_run_ut_qaoa(new_sat_probs[ length(new_sat_probs) ], pdepth, alphas, betas)
			else 
				fin_eng, fin_sup, size = simple_run_t_qaoa(new_sat_probs[ length(new_sat_probs) ], pdepth, alphas, betas)
			end
			exp_res 			= ExpResults(pdepth, fin_eng, fin_sup, size)
			open(string(out_dir, out_file_str), "w") do f
				JSON.print(f, exp_res, 4)
			end
		end 
	end
end

RUN_OPT = parse(Int64, ARGS[ 1 ])
CHC_OPT	= parse(Int64, ARGS[ 2 ])
# run_general_qaoa(12, 3, Int(ceil(12/3)), 100)
if 		CHC_OPT == 1
	@time train_and_run_qaoa(12, 3, Int(ceil(12/3)), 100, RUN_OPT)
elseif	CHC_OPT == 2
	@time load_and_run_qaoa(12, 14, [12,13,14,15,16,17,18,19,20,21], 3, Int(ceil(12/3)), 100, 1, 100, RUN_OPT)
else 
	@time load_and_run_qaoa(12, 60, [12,13,14,15,16,17,18,19,20,21], 3, Int(ceil(12/3)), 100, 1, 100, RUN_OPT)
end


