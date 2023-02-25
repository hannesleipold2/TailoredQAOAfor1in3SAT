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

include("bitvec_conversions.jl")
# include("sat_generator.jl")
# include("measures_n_reps.jl")
include("qaoa_circ.jl")


#=
	# This file is always evolving, title is misleading
=#


function run_single_inst(sat_prob::SatProblem)
	p_rounds    = 20
	pi_co		= pi / 5
	alphas      = [ pi_co * (p/p_rounds)       for p = 1 : p_rounds ]
	betas       = [ pi_co * (1 - (p/p_rounds)) for p = 1 : p_rounds ]
	println(alphas)
	println(betas)

	num_bits    = sat_prob.num_red_variables
	num_states  = 2^(num_bits)
	#=
		SOLUTION VECTORS
	=#
	red_sols = sat_prob.red_solutions
	sol_vecs = Array{Array{Complex{Float64}, 1}, 1}()
	for i = 1 : length(red_sols)
		sol_vec = zeros(Complex{Float64}, num_states)
		sol_vec[ bit_vec_to_int(red_sols[ i ]) ] = Float64(1)
		push!(sol_vecs, sol_vec)
	end	
	# println(sat_prob)
	wave_func 	= zeros(Complex{Float64}, num_states)
	for i = 1 : num_states
		wave_func[i] = (1/sqrt(num_states))
	end
	costop_vec  	= make_costop_vec(sat_prob)
	hollow_unitary	= SparseMatrixCSC{Complex{Float64}, Int64}(spzeros(num_states, num_states))
	for p = 1 : p_rounds
		fin_sup = calc_sol_support(sol_vecs, wave_func)
		println(string("SOLUTION SUPPORT: \t", string(fin_sup+0.00001)[1:5]))
		println(string("NORM: \t\t", string(norm(wave_func)+0.00001)[1:5]))
		# print_supstates()
		phase_energy!(wave_func, costop_vec, alphas[p])
		wave_func = apply_xmixer(wave_func, hollow_unitary, num_bits, betas[p])
	end
	fin_sup = calc_sol_support(sol_vecs, wave_func)
	println(fin_sup)
	return 0 
end

function run_tqaoa(nbits, clen, mclauses, kinsts)
	upp_dir_str = string("./sat_1in", clen, "/")
	dir_str = string(upp_dir_str, "rand_insts", "_nbits=", nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "/")
	open(string(dir_str,"inst_", 1,".json"), "r") do f 
		json_string = JSON.read(f, String)
		new_sat_prob = Unmarshal.unmarshal(SatProblem, JSON.parse(json_string))
		run_single_inst(new_sat_prob)
	end
	return 0 
end


run_tqaoa(12, 3, 4, 100)
