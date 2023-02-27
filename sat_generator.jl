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

include("sat_problem.jl")

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
			# println(typeof(rem_cl_ids))
			# println(to_rem_ids)
			for i = 1 : length(to_rem_ids)
				# println(to_rem_ids[i])
				deleteat!(rem_cl_ids, to_rem_ids[i])
			end
			# println(rem_cl_ids)
			# println(length(sat_inst))
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
			# println(typeof(num_red_uncoved_bits))
			# println(typeof(red_solutions))
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
			upp_dir_str = string("./sat_1in", clen, "/")
			dir_str = string(upp_dir_str, "rand_insts", "_nbits=", nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "/")		
			if !isdir(upp_dir_str)
				mkdir(upp_dir_str)
			end
			if !isdir(dir_str)
				mkdir(dir_str)
			end
			open(string(dir_str,"inst_", use_cnt, ".json"),"w") do f
				JSON.print(f, sat_prob, 4)
			end			
		end 
		gen_cnt += 1
	end

	upp_dir_str = string("./sat_1in", clen, "/")
	dir_str = string(upp_dir_str, "rand_insts", "_nbits=", nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "/")

	open(string(dir_str,"inst_", 1,".json"), "r") do f 
		json_string = JSON.read(f, String)
		new_sat_prob = Unmarshal.unmarshal(SatProblem, JSON.parse(json_string))
		# println(new_sat_prob.clauses)
		# println(new_sat_prob.red_clauses)
		# println(new_sat_prob.red_solutions)
	end

	return 0 
	#=
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
	=# 
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
	nbits 		= 15
	clen        = 3
	cl_to_nb	= binomial(clen, 2)
	# mclauses    = Int(floor(nbits/cl_to_nb) + clause_num_round(nbits, cl_to_nb))
	mclauses	= Int(ceil(nbits/cl_to_nb))
	kinsts      = 100
else 
	println()
	run_code 	= "1"
	out_dir     = ARGS[1]
	nbits       = ARGS[2]
	mclauses    = ARGS[3]
	clen        = ARGS[4]
	kinsts      = ARGS[5]
end 

if run_code == "1"
	run_sat(out_dir, nbits, clen, mclauses, kinsts)
end 


