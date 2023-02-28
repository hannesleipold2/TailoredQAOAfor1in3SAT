using Random 
using Dates
using JSON 
include("apply_operators.jl")
include("sat_problem.jl")

using .SatGenerator

import Unmarshal

struct AngleParams
	pdepth::Int64
	avg_fin_eng::Float64
	avg_fin_sup::Float64
	num_epochs::Int64
	num_runs::Int64 
	epoch_engs::Array{Float64, 1}
	epoch_sups::Array{Float64, 1}
	alphas::Array{Float64, 1}
	betas::Array{Float64, 1}
end

function simple_param_shift(	wave_func, costop_vecs, U_xmixer, pdepth, alphas, betas, 
								all_sol_vecs, num_bits, num_states, shift_size=0.002, grad_coef=0.010, num_shifts=4, DO_PRINT=0)
	rand_id 	= rand(1:pdepth)
	rand_bit 	= rand(0:1)
	if rand_bit == 1
		alphas[rand_id] += shift_size
	else 
		betas[rand_id] 	+= shift_size
	end
	avg_fin_eng1 = 0
	avg_fin_sup1 = 0 
	for i = 1 : length(costop_vecs)
		fin_eng1, fin_sup1 = run_ut_qaoa(wave_func, costop_vecs[ i ], U_xmixer, pdepth, copy(alphas), copy(betas), all_sol_vecs[ i ], num_bits, num_states)
		avg_fin_eng1 += fin_eng1 / length(costop_vecs)
		avg_fin_sup1 += fin_sup1 / length(costop_vecs)
	end
	if rand_bit == 1
		alphas[rand_id] -= (2 * shift_size)
	else 
		betas[rand_id] 	-= (2 * shift_size)
	end
	avg_fin_eng2 = 0
	avg_fin_sup2 = 0 
	for i = 1 : length(costop_vecs)
		fin_eng2, fin_sup2 = run_ut_qaoa(wave_func, costop_vecs[ i ], U_xmixer, pdepth, alphas, betas, all_sol_vecs[ i ], num_bits, num_states)
		avg_fin_eng2 += fin_eng2 / length(costop_vecs)
		avg_fin_sup2 += fin_sup2 / length(costop_vecs)
	end
	appro_grad = (avg_fin_eng2 - avg_fin_eng1)
	if rand_bit == 1
		alphas[rand_id] += shift_size + (grad_coef * appro_grad)
	else 
		betas[rand_id] 	+= shift_size + (grad_coef * appro_grad)
	end
	return appro_grad
end

function simple_subspace_param_shift(	wave_func, costop_vecs, U_xmixer, reds_to_acts, pdepth, alphas, betas, 
								all_sol_vecs, num_bits, num_states, shift_size=0.002, grad_coef=0.010, num_shifts=4, DO_PRINT=0)
	rand_id 	= rand(1:pdepth)
	rand_bit 	= rand(0:1)
	if rand_bit == 1
		alphas[rand_id] += shift_size
	else 
		betas[rand_id] 	+= shift_size
	end
	avg_fin_eng1 = 0
	avg_fin_sup1 = 0 
	for i = 1 : length(costop_vecs)
		fin_eng1, fin_sup1 = run_t_qaoa(wave_func, costop_vecs[ i ], U_xmixer, reds_to_acts, pdepth, copy(alphas), copy(betas), all_sol_vecs[ i ], num_bits, num_states)
		avg_fin_eng1 += fin_eng1 / length(costop_vecs)
		avg_fin_sup1 += fin_sup1 / length(costop_vecs)
	end
	if rand_bit == 1
		alphas[rand_id] -= (2 * shift_size)
	else 
		betas[rand_id] 	-= (2 * shift_size)
	end
	avg_fin_eng2 = 0
	avg_fin_sup2 = 0 
	for i = 1 : length(costop_vecs)
		fin_eng2, fin_sup2 = run_t_qaoa(wave_func, costop_vecs[ i ], U_xmixer, reds_to_acts, pdepth, alphas, betas, all_sol_vecs[ i ], num_bits, num_states)
		avg_fin_eng2 += fin_eng2 / length(costop_vecs)
		avg_fin_sup2 += fin_sup2 / length(costop_vecs)
	end
	appro_grad = (avg_fin_eng2 - avg_fin_eng1)
	if rand_bit == 1
		alphas[rand_id] += shift_size + (grad_coef * appro_grad)
	else 
		betas[rand_id] 	+= shift_size + (grad_coef * appro_grad)
	end
	return appro_grad
end


function train_t_qaoa(sat_probs::Array{SatProblem, 1}, alphas, betas, pdepth=10, batch_size=10)
	## filter batches to have the same overall reduced size
	first_sat_id = rand(1:length(sat_probs))
	
	return 0 
end

function init_alphas_n_betas(pdepth, alpha_co, beta_co, INIT_CHOICE=1)
	if 		INIT_CHOICE == 1
		alphas      = [ alpha_co * pi * (p/pdepth)			for p = 1 : pdepth ]
		betas       = [ beta_co  * pi * (1 - (p/pdepth)) 	for p = 1 : pdepth ]
		return alphas, betas
	elseif  INIT_CHOICE == 2 
		alphas      = [ alpha_co * pi 						for p = 1 : pdepth ]
		betas       = [ beta_co  * pi 					 	for p = 1 : pdepth ]
		return alphas, betas
	end 
end


function train_ut_qaoa(sat_probs::Array{SatProblem, 1}, num_runs=50000, pre_runs=500, num_epochs=10, pdepth=14, batch_size=1)
	## filter batches to have the same overall reduced size
	clen 			= sat_probs[ 1 ].vars_per_clause
	unred_num_bits 	= sat_probs[ 1 ].num_variables
	num_clauses 	= sat_probs[ 1 ].num_clauses
	kinsts 			= length(sat_probs)
	# handle different reduced problem sizes
	all_snum_bit_set = Set([ sat_probs[ i ].num_red_variables for i = 1 : length(sat_probs) ])
	all_snum_bits	= [ i for i in all_snum_bit_set ]
	sort!(all_snum_bits)
	bit_to_ind 		= Dict([ all_snum_bits[ i ] => i for i = 1 : length(all_snum_bits) ])
	### 	SET-UP 		###
	println(all_snum_bits)
	all_U_xmixers	= [ Array{SparseMatrixCSC{Complex{Float64}, Int64}, 1}() 	for i = 1 : length(bit_to_ind) ]
	all_wave_funcs	= [ Array{Complex{Float64}, 1}() 							for i = 1 : length(bit_to_ind) ]
	for i = 1 : length(all_snum_bits)
		all_U_xmixers[ i ] 		= init_xmixers(all_snum_bits[ i ])
		all_wave_funcs[ i ] 	= init_ut_wavefunc(2^(all_snum_bits[ i ]))
	end

	all_num_bits	= [ sat_probs[ i ].num_red_variables 									for i = 1 : kinsts ]
	all_num_states 	= [ 2^(all_num_bits[ i ]) 												for i = 1 : kinsts ]
	all_clauses 	= [ sat_probs[ i ].red_clauses 											for i = 1 : kinsts ]
	all_costop_vecs = [ init_cost_oper(all_num_bits[ i ], all_clauses[ i ]) 				for i = 1 : kinsts ]
	all_sol_vecs	= [ init_sol_space(sat_probs[ i ].red_solutions, all_num_states[ i ]) 	for i = 1 : kinsts ]

	###		EVAL		###
	function apply_epoch(curr_alphas, curr_betas)
		avg_fin_eng	= 0
		avg_fin_sup	= 0
		for k = 1 : kinsts
			num_bits 			= all_num_bits[ k ]
			num_states			= 2^(num_bits)
			ind_id 				= bit_to_ind[ num_bits ]
			fin_eng, fin_sup 	= run_ut_qaoa(all_wave_funcs[ ind_id ], all_costop_vecs[ k ], all_U_xmixers[ ind_id ], pdepth, 
											curr_alphas, curr_betas, all_sol_vecs[ k ], num_bits, num_states)
			avg_fin_eng 		+= fin_eng/kinsts
			avg_fin_sup 		+= fin_sup/kinsts
		end	 
		return avg_fin_eng, avg_fin_sup
	end
	
	### 	RUN 		###
	SHIFT_SIZE 		= 0.05
	GRAD_COEF 		= 0.05 
	epoch_engs 		= Array{Float64, 1}()
	epoch_sups 		= Array{Float64, 1}()
	best_alphas		= [ 0.0 for i = 1 : pdepth ]
	best_betas 		= [ 0.0 for i = 1 : pdepth ]
	best_avg_fin_eng= 2^(unred_num_bits)  
	best_avg_fin_sup= 0
	best_num_alpha	= 0
	best_num_beta 	= 0
	num_alphas		= 1
	num_betas		= 1
	alpha_cos 		= [ 0.20 * i/num_alphas for i = 1 : num_alphas ]
	beta_cos 		= [ 0.05 * i/num_betas 	for i = 1 : num_betas  ]
	BEST_CHOICE		= 0
	NUM_INIT_CHOICES= 2
	for i = 1 : num_alphas
		for j = 1 : num_betas
			println("SWEEPING: ", i, " ", j)
			for INIT_CHOICE = 1 : NUM_INIT_CHOICES
				curr_alphas, curr_betas = init_alphas_n_betas(pdepth, alpha_cos[ i ], beta_cos[ j ], INIT_CHOICE)
				## OPT ##
				grads 		= Array{Float64, 1}()
				start_time 	= Dates.now()
				avg_fin_eng = 0
				avg_fin_sup	= 0
				for run_id = 1 : pre_runs
					rand_id 	= rand(1:kinsts)
					num_bits 	= all_num_bits[ rand_id ]
					num_states	= 2^(num_bits)
					ind_id 		= bit_to_ind[ num_bits ]
					cost_ids	= sort(push!([ rand(1:kinsts) for i = 1 : batch_size - 1 ], rand_id))
					for i = 1 : length(cost_ids)
						if all_num_bits[ length(cost_ids) - i + 1 ] != num_bits
							deleteat!(cost_ids, length(cost_ids) - i + 1)
						end
					end
					appro_grad 	= simple_param_shift(all_wave_funcs[ ind_id ], [ all_costop_vecs[ cost_id ] for cost_id in cost_ids ], all_U_xmixers[ ind_id ], pdepth, 
														curr_alphas, curr_betas, [ all_sol_vecs[ cost_id ] for cost_id in cost_ids ], num_bits, num_states, SHIFT_SIZE, GRAD_COEF)
					push!(grads, appro_grad)
				end
				avg_fin_eng, avg_fin_sup = apply_epoch(curr_alphas, curr_betas)
				if avg_fin_eng < best_avg_fin_eng
					best_avg_fin_eng 	= avg_fin_eng 
					best_avg_fin_sup 	= avg_fin_sup
					best_num_alpha		= i 
					best_num_beta 		= j
					BEST_CHOICE			= INIT_CHOICE
					copyto!(best_alphas, curr_alphas)
					copyto!(best_betas,  curr_betas)
				end
			end
		end
	end
	curr_alphas, curr_betas = [ best_alphas, best_betas ]
	## OPT ##
	grads 		= Array{Float64, 1}()
	start_time 	= Dates.now()
	avg_fin_eng = 0
	avg_fin_sup	= 0
	for run_id = 1 : num_runs # num_runs
		if run_id%Int(num_runs/10) == 0
			print()
			println(Int(run_id/Int(num_runs/10)), "0% ", Dates.now() - start_time)
			avg_fin_eng, avg_fin_sup = apply_epoch(curr_alphas, curr_betas)
			push!(epoch_engs, avg_fin_eng)
			push!(epoch_sups, avg_fin_sup)
			println("EPOCH ENG: \t", avg_fin_eng, "\t EPOCH SUP: ", avg_fin_sup)
		end
		rand_id 	= rand(1:kinsts)
		num_bits 	= all_num_bits[ rand_id ]
		num_states	= 2^(num_bits)
		ind_id 		= bit_to_ind[ num_bits ]
		cost_ids	= sort(push!([ rand(1:kinsts) for i = 1 : batch_size - 1 ], rand_id))
		for i = 1 : length(cost_ids)
			if all_num_bits[ length(cost_ids) - i + 1 ] != num_bits
				deleteat!(cost_ids, length(cost_ids) - i + 1)
			end
		end
		appro_grad 	= simple_param_shift(all_wave_funcs[ ind_id ], [ all_costop_vecs[ cost_id ] for cost_id in cost_ids ], all_U_xmixers[ ind_id ], pdepth, 
											curr_alphas, curr_betas, [ all_sol_vecs[ cost_id ] for cost_id in cost_ids ], num_bits, num_states, SHIFT_SIZE, GRAD_COEF)
		push!(grads, appro_grad)
	end
	avg_fin_eng, avg_fin_sup = apply_epoch(curr_alphas, curr_betas)
	if avg_fin_eng < best_avg_fin_eng
		best_avg_fin_eng 	= avg_fin_eng 
		best_avg_fin_sup 	= avg_fin_sup
		copyto!(best_alphas, curr_alphas)
		copyto!(best_betas,  curr_betas)
	end
	println("BEST ALPHAS: ", best_alphas)
	println("BEST BETAS: ", best_betas)
	println("BEST FIN ENG: ", best_avg_fin_eng)
	println("BEST FIN SUP: ", best_avg_fin_sup)
	###		SAVE		###
	res_params		= AngleParams(pdepth, best_avg_fin_eng, best_avg_fin_sup, num_epochs, num_runs, epoch_engs, epoch_sups, best_alphas, best_betas)
	upp_dir_str		= string("./train_angles_ut_clen=", clen, "/")
	if !isdir(upp_dir_str)
		mkdir(upp_dir_str)
	end
	out_path		= string(upp_dir_str, "rand_insts", "_nbits=", unred_num_bits, "_mclauses=", num_clauses, "_kinsts=", kinsts,"_pdepth=", pdepth, ".json")
	open(out_path, "w") do f
		JSON.print(f, res_params, 4)
	end
	open(out_path, "r") do f
		json_string = JSON.read(f, String)
		read_params = Unmarshal.unmarshal(AngleParams, JSON.parse(json_string))
		println(read_params)
	end
	return 0 
end


function train_t_qaoa(sat_probs::Array{SatProblem, 1}, num_runs=50000, pre_runs=500, num_epochs=10, pdepth=14)
	## filter batches to have the same overall reduced size
	clen 				= sat_probs[ 1 ].vars_per_clause
	unred_num_bits 		= sat_probs[ 1 ].num_variables
	num_clauses 		= sat_probs[ 1 ].num_clauses
	kinsts 				= length(sat_probs)
	### 	SET-UP 		###
	all_num_bits		= [ sat_probs[ i ].num_red_variables 									for i = 1 : kinsts ]
	all_num_states 		= [ 2^(all_num_bits[ i ]) 												for i = 1 : kinsts ]
	all_clauses 		= [ sat_probs[ i ].red_clauses 											for i = 1 : kinsts ]
	all_dis_clauses 	= [ sat_probs[ i ].red_max_disjoint_clauses 							for i = 1 : kinsts ]
	all_rem_clauses 	= [ sat_probs[ i ].red_remaining_clauses 								for i = 1 : kinsts ] 
	all_uncov_vars		= [ sat_probs[ i ].red_variables_uncovered								for i = 1 : kinsts ]

	###		ADV SET-UP	###
	all_sols_per_clause	= [ find_sols_per_clause(all_num_bits[ i ], all_dis_clauses[ i ])		for i = 1 : kinsts ]
	all_reds_to_acts 	= [ find_tailor_subspace(sat_probs[ i ]) 								for i = 1 : kinsts ]

	all_clause_mixers 	= [ init_clause_mixers(all_num_bits[ i ], all_reds_to_acts[ i ], all_sols_per_clause[ i ], 
							all_uncov_vars[ i ]) 												for i = 1 : kinsts ]
	all_wave_funcs		= [ init_t_wavefunc(all_num_states[ i ], all_reds_to_acts[ i ]) 		for i = 1 : kinsts ] 
	all_costop_vecs		= [ init_cost_oper(all_num_bits[ i ], all_rem_clauses[ i ]) 			for i = 1 : kinsts ]
	all_sol_vecs		= [ init_sol_space(sat_probs[ i ].red_solutions, all_num_states[ i ]) 	for i = 1 : kinsts ]


	###		EVAL		###
	function apply_epoch(curr_alphas, curr_betas)
		avg_fin_eng	= 0
		avg_fin_sup	= 0
		for k = 1 : kinsts
			num_bits 			= all_num_bits[ k ]
			num_states			= 2^(num_bits)
			fin_eng, fin_sup 	= run_t_qaoa(all_wave_funcs[ k ], all_costop_vecs[ k ], all_clause_mixers[ k ], all_reds_to_acts[ k ], pdepth, 
											curr_alphas, curr_betas, all_sol_vecs[ k ], num_bits, num_states)
			avg_fin_eng 		+= fin_eng/kinsts
			avg_fin_sup 		+= fin_sup/kinsts
		end	 
		return avg_fin_eng, avg_fin_sup
	end
	
	# sparse([ i for i = 1 : length(costop_vec) ], [ i for i = 1 : length(costop_vec) ], costop_vec, length(costop_vec), length(costop_vec))

	### 	RUN 		###
	SHIFT_SIZE 		= 0.05
	GRAD_COEF 		= 0.05
	epoch_engs 		= Array{Float64, 1}()
	epoch_sups 		= Array{Float64, 1}()
	best_alphas		= [ 0.0 for i = 1 : pdepth ]
	best_betas 		= [ 0.0 for i = 1 : pdepth ]
	best_avg_fin_eng= 2^(unred_num_bits)  
	best_avg_fin_sup= 0
	best_num_alpha	= 0
	best_num_beta 	= 0
	num_alphas		= 1
	num_betas		= 1
	alpha_cos 		= [ 0.20 * i/num_alphas for i = 1 : num_alphas ]
	beta_cos 		= [ 0.05 * i/num_betas 	for i = 1 : num_betas  ]
	BEST_CHOICE		= 0
	NUM_INIT_CHOICES= 1
	for i = 1 : num_alphas
		for j = 1 : num_betas
			println("SWEEPING: ", i, " ", j)
			for INIT_CHOICE = 1 : NUM_INIT_CHOICES
				curr_alphas, curr_betas = init_alphas_n_betas(pdepth, alpha_cos[ i ], beta_cos[ j ], INIT_CHOICE)
				## OPT ##
				grads 		= Array{Float64, 1}()
				start_time 	= Dates.now()
				avg_fin_eng = 0
				avg_fin_sup	= 0
				for run_id = 1 : pre_runs # num_runs
					rand_id 	= rand(1:kinsts)
					num_bits 	= all_num_bits[ rand_id ]
					num_states	= 2^(num_bits)
					appro_grad 			= simple_subspace_param_shift(all_wave_funcs[ rand_id ], [ all_costop_vecs[ rand_id ]  ], all_clause_mixers[ rand_id ], all_reds_to_acts[ rand_id ], pdepth, 
														curr_alphas, curr_betas, [ all_sol_vecs[ rand_id ] ], num_bits, num_states, SHIFT_SIZE, GRAD_COEF)
					push!(grads, appro_grad)
				end
				avg_fin_eng, avg_fin_sup = apply_epoch(curr_alphas, curr_betas)
				if avg_fin_eng < best_avg_fin_eng
					best_avg_fin_eng 	= avg_fin_eng 
					best_avg_fin_sup 	= avg_fin_sup
					best_num_alpha		= i 
					best_num_beta 		= j
					BEST_CHOICE			= INIT_CHOICE
					copyto!(best_alphas, curr_alphas)
					copyto!(best_betas,  curr_betas)
				end
			end
		end
	end
	curr_alphas, curr_betas = [ best_alphas, best_betas ]
	## OPT ##
	grads 		= Array{Float64, 1}()
	start_time 	= Dates.now()
	avg_fin_eng = 0
	avg_fin_sup	= 0
	for run_id = 1 : num_runs # num_runs
		if num_runs > 1000 && run_id%Int(num_runs/10) == 0
			print()
			println(Int(run_id/Int(num_runs/10)), "0% ", Dates.now() - start_time)
			avg_fin_eng, avg_fin_sup = apply_epoch(curr_alphas, curr_betas)
			push!(epoch_engs, avg_fin_eng)
			push!(epoch_sups, avg_fin_sup)
			println(avg_fin_eng)
			println(avg_fin_sup)
		end
		rand_id 	= rand(1:kinsts)
		num_bits 	= all_num_bits[ rand_id ]
		num_states	= 2^(num_bits)
		appro_grad 			= simple_subspace_param_shift(all_wave_funcs[ rand_id ], [ all_costop_vecs[ rand_id ]  ], all_clause_mixers[ rand_id ], all_reds_to_acts[ rand_id ], pdepth, 
											curr_alphas, curr_betas, [ all_sol_vecs[ rand_id ] ], num_bits, num_states, SHIFT_SIZE, GRAD_COEF)
		push!(grads, appro_grad)
	end
	avg_fin_eng, avg_fin_sup = apply_epoch(curr_alphas, curr_betas)
	if avg_fin_eng < best_avg_fin_eng
		best_avg_fin_eng 	= avg_fin_eng 
		best_avg_fin_sup 	= avg_fin_sup
		copyto!(best_alphas, curr_alphas)
		copyto!(best_betas,  curr_betas)
	end
	println("BEST ALPHAS: ", best_alphas)
	println("BEST BETAS: ", best_betas)
	println("BEST FIN ENG: ", best_avg_fin_eng)
	println("BEST FIN SUP: ", best_avg_fin_sup)
	###		SAVE		###
	res_params		= AngleParams(pdepth, best_avg_fin_eng, best_avg_fin_sup, num_epochs, num_runs, epoch_engs, epoch_sups, best_alphas, best_betas)
	upp_dir_str		= string("./train_angles_t_clen=", clen, "/")
	if !isdir(upp_dir_str)
		mkdir(upp_dir_str)
	end
	out_path		= string(upp_dir_str, "rand_insts", "_nbits=", unred_num_bits, "_mclauses=", num_clauses, "_kinsts=", kinsts,"_pdepth=", pdepth, ".json")
	open(out_path, "w") do f
		JSON.print(f, res_params, 4)
	end
	open(out_path, "r") do f
		json_string = JSON.read(f, String)
		read_params = Unmarshal.unmarshal(AngleParams, JSON.parse(json_string))
		println(read_params)
	end
	return 0 
end





#=
struct AngleParams
	pdepth::Int64
	avg_fin_eng::Float64
	avg_fin_sup::Float64
	epoch_num::Float64
	num_runs::Int64 
	epoch_engs::Array{Float64, 1}
	epoch_sups::Array{Float64, 1}
	alphas::Array{Float64, 1}
	betas::Array{Float64, 1}
end
=#


























