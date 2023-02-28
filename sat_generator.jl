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
			for i = 1 : length(to_rem_ids)
				deleteat!(rem_cl_ids, to_rem_ids[i])
			end
			dis_clauses = [ sat_inst[dis_cl_ids[i]] for i = 1 : length(dis_cl_ids) ]
			rem_clauses = [ sat_inst[rem_cl_ids[i]] for i = 1 : length(rem_cl_ids) ]
			red_clauses, red_dis_clauses, red_rem_clauses, vars_to_red, red_to_vars, red_bits = SatGenerator.map_to_reduced_sat(nbits, sat_inst, dis_clauses, rem_clauses)
			red_int_solutions 	= SatGenerator.all_sat_sols(red_bits, red_clauses) 
			red_solutions		= [ int_to_bit_vec(red_int_solutions[i], red_bits) for i = 1 : length(red_int_solutions) ]
			red_uncoved_vars	= SatGenerator.find_unused_variable(red_bits, red_dis_clauses)
			num_red_uncoved_bits= length(red_uncoved_vars)
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
	end

	return 0 
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
	out_dir     = "./sat1in3/"
	nbits 		= 12
	if length(ARGS) > 1
		nbits 	= parse(Int64, ARGS[2])
	end
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


