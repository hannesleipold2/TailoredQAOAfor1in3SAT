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

module SatGenerator

	include("bitvec_conversions.jl")

	export Literal, SatType, SatProblem

	@enum SatType begin
		Anyink 		= 1
		Oneink 		= 2
		Notalleq 	= 3
	end
	
	const SATPROBLEMTYPE = Oneink 

	struct Literal
		variable::Int64
		expected_value::Int64
	end

	struct SatProblem 
		num_variables::Int64 
		num_clauses::Int64
		vars_per_clause::Int64
		num_red_variables::Int64
		num_red_uncov_variables::Int64 
		clauses::Array{Array{Literal, 1}, 1}
		max_disjoint_clauses::Array{Array{Literal, 1}, 1}
		remaining_clauses::Array{Array{Literal, 1}, 1}
		red_clauses::Array{Array{Literal, 1}, 1}
		red_max_disjoint_clauses::Array{Array{Literal, 1}, 1}
		red_remaining_clauses::Array{Array{Literal, 1}, 1}
		red_solutions::Array{Array{Int64, 1}, 1}
		variables_unused::Array{Int64, 1}
		red_variables_uncovered::Array{Int64, 1}
		var_to_reduced::Dict{Int64, Int64}
		reduced_to_var::Dict{Int64, Int64}
	end

	struct ZatProblem 
		num_variables::Int64 
		num_clauses::Int64
		vars_per_clause::Int64
		num_red_variables::Int64
		num_red_uncov_variables::Int64 
		clauses::Array{Array{Tuple{Int64,Int64}, 1}, 1}
		max_disjoint_clauses::Array{Array{Tuple{Int64,Int64}, 1}, 1}
		remaining_clauses::Array{Array{Tuple{Int64,Int64}, 1}, 1}
		red_clauses::Array{Array{Tuple{Int64,Int64}, 1}, 1}
		red_max_disjoint_clauses::Array{Array{Tuple{Int64,Int64}, 1}, 1}
		red_remaining_clauses::Array{Array{Tuple{Int64,Int64}, 1}, 1}
		red_solutions::Array{Array{Int64, 1}, 1}
		variables_unused::Array{Int64, 1}
		red_variables_uncovered::Array{Int64, 1}
		var_to_reduced::Dict{Int64, Int64}
		reduced_to_var::Dict{Int64, Int64}
	end

	# CLAUSE GENERATIONS

	function gen_sat_clause(nbits, mclauses, clen)
		clauses = Array{Array{Literal, 1}, 1}()
		for cl_id = 1 : mclauses
			clause = Array{Literal, 1}()
			for var_id = 1 : clen
				id_r_ng = rand(0:1)
				bit_loc = rand(1:nbits)
				push!(clause, Literal(bit_loc, id_r_ng))
			end
			# println(clause)
			# println(clauses)
			push!(clauses, clause)
		end
		# println()
		# breakhere!()
		# println(clauses)
		return clauses 
	end

	# TODO: fix for other sat problems.

	function vec_sats_clause(state_vec, clause)
		# println(SATPROBLEMTYPE)
		if 1 == 1 
			num_sat = 0
			for lit_id = 1 : length(clause)
				if state_vec[ clause[lit_id].variable ] == clause[lit_id].expected_value
					num_sat += 1
				end
			end
			if num_sat == 1
				return 1
			else 
				return 0
			end
		else
			println("PROBLEM")
			breakhere!()
		end
		return 1 
	end

	# SOLUTION FINDING

	function app_all_clauses(potent_sols, nbits, clauses)
		sols = Array{Int64, 1}()
		for i = 1 : length(potent_sols)
			# println(potent_sols[ i ])
			pot_vec = int_to_bit_vec(potent_sols[ i ], nbits)
			add_vec = 1
			for j = 1 : length(clauses)
				if vec_sats_clause(pot_vec, clauses[ j ]) == 0
					add_vec = 0
					break 
				end
			end 
			if add_vec == 1
				push!(sols, bit_vec_to_int(pot_vec))
			end
		end
		return sols 
	end

	function all_sat_sols(nbits, clauses)
		pot_sols = zeros(Int64, 2^(nbits))
		for i = 1 : 2^(nbits) 
			pot_sols[ i ] = i 
		end 
		sols = app_all_clauses(pot_sols, nbits, clauses)
		return sols 
	end

	# MAX DISJOINT SET FINDING

	function rec_max_disjoint_clauses(nbits, clauses, curr_clause_id, candidate_clause_ids, curr_var_cover, best_clause_ids)
		# println(curr_clause_id)
		# println(best_clause_ids)
		if 	curr_clause_id == length(clauses) + 1
			if length(candidate_clause_ids) > length(best_clause_ids[1])
				best_clause_ids[1] = deepcopy(candidate_clause_ids)
			end
			return 
		end
		rec_max_disjoint_clauses(nbits, clauses, curr_clause_id + 1, candidate_clause_ids, curr_var_cover, best_clause_ids)
		clause_conflict	= 0
		for lit_id = 1 : length(clauses[curr_clause_id])
			var_id = clauses[curr_clause_id][lit_id].variable
			if var_id in curr_var_cover
				 clause_conflict = 1
			end
		end
		if clause_conflict == 0
			push!(candidate_clause_ids, curr_clause_id)
			for lit_id = 1 : length(clauses[curr_clause_id])
				var_id = clauses[curr_clause_id][lit_id].variable
				push!(curr_var_cover, var_id)
			end
			rec_max_disjoint_clauses(nbits, clauses, curr_clause_id + 1, candidate_clause_ids, curr_var_cover, best_clause_ids)
			pop!(candidate_clause_ids)
			for lit_id = 1 : length(clauses[curr_clause_id])
				var_id = clauses[curr_clause_id][lit_id].variable
				delete!(curr_var_cover, var_id)
			end
		end
	end

	function find_max_disjoint_clauses(nbits, clauses)
		start_clause_id 		= 1
		candidate_clause_ids 	= Array{Int64, 1}()
		var_cover 				= Set{Int64}()
		best_clause_ids 		= [  deepcopy(candidate_clause_ids) ]
		rec_max_disjoint_clauses(nbits, clauses, start_clause_id, candidate_clause_ids, var_cover, best_clause_ids)
		# println(best_clause_ids)
		return best_clause_ids[1]
	end


	# SPLIT INTO MAX DISJOINT AND REMAINING CLAUSES, FILTER 

	function find_unused_variable(nbits, clauses)
		var_used = Set{Int64}()
		for clause in clauses
			for lit in clause
				# println(lit)
				push!(var_used, lit.variable)
			end
		end
		vars_unused = Array{Int64, 1}()
		for var_id = 1 : nbits
			if !(var_id in var_used)
				push!(vars_unused, var_id)
			end
		end
		return vars_unused
	end

	function map_to_reduced_sat(nbits, clauses, dis_clauses, rem_clauses)
		vars_unused = find_unused_variable(nbits, clauses)
		vars_to_red = Dict{Int64, Int64}()
		red_to_vars = Dict{Int64, Int64}()
		var_cnt = 0
		for i = 1 : nbits
			if !(i in vars_unused) 
				var_cnt 				+= 1
				vars_to_red[i] 			= var_cnt
				red_to_vars[var_cnt] 	= i   
			end
		end
		red_bits		= nbits - length(vars_unused)
		red_clauses 	= typeof(clauses)()  
		red_dis_clauses = typeof(clauses)()
		red_rem_clauses = typeof(clauses)()
		for cl_id = 1 : length(clauses)
			red_clause = Array{Literal, 1}()
			for lit_id = 1 : length(clauses[cl_id])
				curr_lit = clauses[cl_id][lit_id]
				push!(red_clause, Literal(vars_to_red[curr_lit.variable], curr_lit.expected_value))
			end
			push!(red_clauses, red_clause)
		end
		for cl_id = 1 : length(dis_clauses)
			red_dis_clause = Array{Literal, 1}()
			for lit_id = 1 : length(dis_clauses[cl_id])
				curr_lit = dis_clauses[cl_id][lit_id]
				push!(red_dis_clause, Literal(vars_to_red[curr_lit.variable], curr_lit.expected_value))
			end
			push!(red_dis_clauses, red_dis_clause)
		end
		for cl_id = 1 : length(rem_clauses)
			red_rem_clause = Array{Literal, 1}()
			for lit_id = 1 : length(rem_clauses[cl_id])
				curr_lit = rem_clauses[cl_id][lit_id]
				push!(red_rem_clause, Literal(vars_to_red[curr_lit.variable], curr_lit.expected_value))
			end
			push!(red_rem_clauses, red_rem_clause)
		end
		return red_clauses, red_dis_clauses, red_rem_clauses, vars_to_red, red_to_vars, red_bits
	end

	function gen_full_sat_problem(nbits, mclauses, clen)
		# Generate INST
		sat_inst 	= SatGenerator.gen_sat_clause(nbits, mclauses, clen)
		unused_vars = SatGenerator.find_unused_variable(nbits, sat_inst)
		
		# Find max disjoint clauses and remaining clauses
		dis_cl_ids 	= SatGenerator.find_max_disjoint_clauses(nbits, sat_inst)
		to_rem_ids 	= sort(dis_cl_ids, rev=true)
		rem_cl_ids	= [ i for i = 1 : nbits ] 
		for i = 1 : length(to_rem_ids)
			delete!(rem_cl_ids, to_rem_ids[i])
		end
		dis_clauses = [ sat_inst[dis_cl_ids[i]] for i = 1 : length(dis_cl_ids) ]
		rem_clauses = [ sat_inst[rem_cl_ids[i]] for i = 1 : length(rem_cl_ids) ]
		
		# Find reduced representations
		red_clauses, red_dis_clauses, red_rem_clauses, vars_to_red, red_to_vars, red_bits = SatGenerator.map_to_reduced_sat(nbits, clauses, dis_clauses, rem_clauses)
		# println(dis_cl)
		# println(sat_inst)
		red_solutions 		= SatGenerator.all_sat_sols(red_bits, red_clauses)
		red_uncoved_vars	= SatGenerator.find_unused_variable(red_bits, red_dis_clauses)
		num_red_uncoved_bits= length(red_uncoved_vars)
		#=
		struct SatProblem
			num_variables::Int64 
			num_clauses::Int64
			vars_per_clause::Int64
			num_red_variables::Int64
			num_red_uncov_variables::Int64 
			clauses::Array{Array{Literal, 1}, 1}
			max_disjoint_clauses::Array{Array{Literal, 1}, 1}
			remaining_clauses::Array{Array{Literal, 1}, 1}
			red_clauses::Array{Array{Literal, 1}, 1}
			red_max_disjoint_clauses::Array{Array{Literal, 1}, 1}
			red_remaining_clauses::Array{Array{Literal, 1}, 1}
			red_solutions::Array{Array{Int64, 1}, 1}
			variables_unused::Array{Int64, 1}
			red_variables_uncovered::Array{Int64, 1}
			var_to_reduced::Dict{Int64, Int64}
			reduced_to_var::Dict{Int64, Int64}
		end		
		=#
		function lit_to_tuple(lit)
			return (lit.variable, lit.expected_value)
		end
		function arr_lit_to_tuple(arr_lit)
			return [ lit_to_tuple(arr_lit[i]) for i = 1 : length(arr_lit) ]
		end
		function arr_arr_lit_to_tuple(arr_arr_lit)
			return [ arr_lit_to_tuple(arr_arr_lit[i]) for i = 1 : length(arr_arr_lit) ] 
		end
		sat_prob 		= SatProblem(	nbits
									,	mclauses
									,	clen 
									,	num_red_uncoved_bits 
									,	red_bits
									,	sat_inst
									,	dis_clauses
									,	rem_clauses
									,	red_clauses
									,	red_dis_clauses
									,	red_rem_clauses
									,	red_solutions
									,	unused_vars
									,	red_uncoved_vars
									,	vars_to_red
									,	red_to_vars)
		breakhere!()
		push!(all_unused, length(unused_vars))
		push!(all_uncov, length(uncoved_vars) - length(unused_vars))
		push!(sat_insts, sat_inst) 
		push!(all_sols, SatGenerator.all_sat_sols(nbits, sat_insts[ use_cnt ]))
	end
end

using .SatGenerator
# StructTypes.StructType(::Type{Literal}) = StructTypes.Struct()
# StructTypes.StructType(::Type{SatProblem}) = StructTypes.Struct()
include("bitvec_conversions.jl")

function run_sat(out_dir, nbits, clen, mclauses, kinsts)
	println(out_dir)
	println("NUM VARS: \t\t", nbits)
	println("VARS PER CLAUSE: \t", clen)
	println("NUM CLAUSES: \t\t", mclauses)
	println("NUM INSTANCES: \t\t", kinsts)
	println()

	all_sols 	= Array{Array{Int64, 1}, 1}()
	sat_insts 	= Array{typeof(SatGenerator.gen_sat_clause(nbits, mclauses, clen)), 1}()
	sat_probs	= Array{SatProblem, 1}()
	gen_cnt 	= 0
	use_cnt 	= 0
	all_unused	= Array{Int64, 1}()
	all_uncov	= Array{Int64, 1}()
	while use_cnt < kinsts
		sat_inst = SatGenerator.gen_sat_clause(nbits, mclauses, clen)
		sat_sols = SatGenerator.all_sat_sols(nbits, sat_inst) 
		if length(sat_sols) > 0
			use_cnt 	+= 1
			unused_vars = SatGenerator.find_unused_variable(nbits, sat_inst)
			dis_cl_ids 	= SatGenerator.find_max_disjoint_clauses(nbits, sat_inst)
			to_rem_ids 	= sort(dis_cl_ids, rev=true)
			rem_cl_ids	= [ i for i = 1 : mclauses ] 
			println(typeof(rem_cl_ids))
			println(to_rem_ids)
			for i = 1 : length(to_rem_ids)
				println(to_rem_ids[i])
				deleteat!(rem_cl_ids, to_rem_ids[i])
			end
			println(rem_cl_ids)
			println(length(sat_inst))
			dis_clauses = [ sat_inst[dis_cl_ids[i]] for i = 1 : length(dis_cl_ids) ]
			rem_clauses = [ sat_inst[rem_cl_ids[i]] for i = 1 : length(rem_cl_ids) ]
			red_clauses, red_dis_clauses, red_rem_clauses, vars_to_red, red_to_vars, red_bits = SatGenerator.map_to_reduced_sat(nbits, sat_inst, dis_clauses, rem_clauses)
			# println(dis_cl)
			# println(sat_inst)
			red_int_solutions 	= SatGenerator.all_sat_sols(red_bits, red_clauses) 
			red_solutions		= [ int_to_bit_vec(red_int_solutions[i], red_bits) for i = 1 : length(red_int_solutions) ]
			red_uncoved_vars	= SatGenerator.find_unused_variable(red_bits, red_dis_clauses)
			num_red_uncoved_bits= length(red_uncoved_vars)
			#=
			struct SatProblem
				num_variables::Int64 
				num_clauses::Int64
				vars_per_clause::Int64
				num_red_variables::Int64
				num_red_uncov_variables::Int64 
				clauses::Array{Array{Literal, 1}, 1}
				max_disjoint_clauses::Array{Array{Literal, 1}, 1}
				remaining_clauses::Array{Array{Literal, 1}, 1}
				red_clauses::Array{Array{Literal, 1}, 1}
				red_max_disjoint_clauses::Array{Array{Literal, 1}, 1}
				red_remaining_clauses::Array{Array{Literal, 1}, 1}
				red_solutions::Array{Array{Int64, 1}, 1}
				variables_unused::Array{Int64, 1}
				red_variables_uncovered::Array{Int64, 1}
				var_to_reduced::Dict{Int64, Int64}
				reduced_to_var::Dict{Int64, Int64}
			end		
			=#
			println(typeof(num_red_uncoved_bits))
			println(typeof(red_solutions))
			sat_prob 		= SatProblem(	nbits
										,	mclauses
										,	clen 
										,	red_bits 
										,	num_red_uncoved_bits
										,	sat_inst
										,	dis_clauses
										,	rem_clauses
										,	red_clauses
										,	red_dis_clauses
										,	red_rem_clauses
										,	red_solutions
										,	unused_vars
										,	red_uncoved_vars
										,	vars_to_red
										,	red_to_vars)
			#
			# new_sat_prob = SatProblem()
			dir_str = string("./sat_1in", clen, "_nbits=", nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "/")
			if !isdir(dir_str)
				mkdir(dir_str)
			end
			open(string(dir_str,"inst_", use_cnt, ".json"),"w") do f
				JSON.print(f, sat_prob, 4)
			end			
		end 
		gen_cnt += 1
	end

	dir_str = string("./sat_1in", clen, "_nbits=", nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "/")

	open(string(dir_str,"inst_", 1,".json"), "r") do f 
		json_string = JSON.read(f, String)
		new_sat_prob = Unmarshal.unmarshal(SatProblem, JSON.parse(json_string))
		# println(new_sat_prob.clauses)
		# println(new_sat_prob.red_clauses)
		# println(new_sat_prob.red_solutions)
	end

	return 0 

	breakhere!()

	avg_unused= sum(all_unused)/length(all_unused)
	avg_uncov = sum(all_uncov)/length(all_uncov)


	println(string("GENCNT, USECNT: \t\t", gen_cnt, "\t", use_cnt))
	println(string("AVG UNUSED VARS: \t\t", string(avg_unused/nbits)[1:5]), 
					"\t", string(sum(all_unused)/length(all_unused) + 0.00001)[1:5])
	println(string("AVG UNCOVED VARS: \t\t", string(avg_uncov/nbits)[1:5]),
					"\t", string(avg_uncov + 0.00001)[1:5],
					"\t", 2^(avg_uncov) * (1.44)^(nbits-avg_unused-avg_uncov) )
	avg_sol_size = sum([ length(all_sols[i]) for i = 1 : kinsts ])/kinsts
	println(string("SOL, OVR SIZE: \t\t\t", string(avg_sol_size/(2^(nbits)) + 0.000001)[1:5], "\t", avg_sol_size, "\t", 2^(nbits)))
	all_sol_lsts	=   [ [ SatGenerator.int_to_bit_vec(all_sols[i][j], nbits)  
							for j = 1 : length(all_sols[i]) ] for i = 1 : kinsts ]
	# println(all_sol_lsts)
	all_dis_clauses = 	[  SatGenerator.max_disjoint_clauses(nbits, sat_insts[i]) for i = 1 : kinsts ]
	ratio = (1.0 * sum([ length(all_dis_clauses[i]) for i = 1 : kinsts ]))/(kinsts * mclauses)
	println(string("dis_cluases/all_clauses: \t",string(ratio + 0.000001)[1:5])) 
	
	

	breakhere!()

	function satinsts_to_JSON(nbits, mclauses, sat_insts, all_sol_lsts)
		k_insts 	= length(sat_insts)
		sat_dict 	= [  Dict("sat_id"=>string(i),
				"num_clauses"=>string(mclauses),
				"num_vars"=>string(nbits),
				"sols"=>[ Dict([ "sol_id"=>j, "sol_vec"=>all_sol_lsts[i][j] ]) for j = 1 : length(all_sol_lsts[i]) ],
				"clauses"=>[ Dict([ "clause_id"=>j, "clause"=>sat_insts[i][j] ]) for j=1:mclauses ]
				)
		for i=1:kinsts ]
	end
	sat_dict = [  Dict("sat_id"=>string(i),
				"num_clauses"=>string(mclauses),
				"num_vars"=>string(nbits),
				"sols"=>[ Dict([ "sol_id"=>j, "sol_vec"=>all_sol_lsts[i][j] ]) for j = 1 : length(all_sol_lsts[i]) ],
				"clauses"=>[ Dict([ "clause_id"=>j, "clause"=>sat_insts[i][j] ]) for j=1:mclauses ]
				)
		for i=1:kinsts ]

	data = Dict("sat_insts"=>sat_dict)
	# json_string = JSON.json(data)

	open("insts.json","w") do f
		JSON.print(f, data, 4)
	end

end

function clause_num_round(nbits, cl_to_nb)
	remainder = nbits % cl_to_nb
	if rand() < remainder/cl_to_nb
		return 1
	else
		return 0
	end
end

if length(ARGS) < 4
	run_code	= ARGS[1] 
	println("PROGRAM: out_dir nbits mclauses clen kinsts")
	println("defaulting...")
	println()
	out_dir     = "./sat1in3_insts/"
	nbits 		= 12
	clen        = 3
	cl_to_nb	= binomial(clen, 2)
	# mclauses    = Int(floor(nbits/cl_to_nb) + clause_num_round(nbits, cl_to_nb))
	mclauses	= Int(ceil(nbits/cl_to_nb))
	kinsts      = 100
else 
	println()
	out_dir     = ARGS[1]
	nbits       = ARGS[2]
	mclauses    = ARGS[3]
	clen        = ARGS[4]
	kinsts      = ARGS[5]
end 

if run_code == "1"
	run_sat(out_dir, nbits, clen, mclauses, kinsts)
end 


