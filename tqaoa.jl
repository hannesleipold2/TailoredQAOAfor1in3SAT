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
include("sat_generator.jl")

using .SatGenerator
#=
	# This file is always evolving, title is misleading
=#


function calc_sol_support(sol_vecs::Array{Array{Complex{Float64},1},1}, wave_func::Array{Complex{Float64},1})
	sol_support = 0.0
	for i = 1 : length(sol_vecs)
		sol_support += abs(dot(sol_vecs[ i ], wave_func / norm(wave_func))) ^ 2
	end
	return sol_support
end


function make_costop_vec(sat_prob::SatProblem)
	num_bits    = sat_prob.num_red_variables
	num_states  = 2^(num_bits)
	clauses     = sat_prob.red_clauses
	clen        = sat_prob.vars_per_clause
	costop_vec  = zeros(Complex{Float64}, num_states)
	for i = 1 : num_states
		i_bitvec    = int_to_bit_vec(i, num_bits)
		i_cost      = 0 
		for clause in clauses
			c_cost  = 1
			for j = 1 : length(clause)
				do_red_cost = 1
				for k = 1 : length(clause)
					if j == k 
						if i_bitvec[clause[k].variable] != clause[k].expected_value
							do_red_cost = 0
						end
					else
						if i_bitvec[clause[k].variable] == clause[k].expected_value
							do_red_cost = 0
						end
					end
				end
				if do_red_cost == 1
					c_cost = 0
				end
			end
			i_cost += c_cost 
		end
		costop_vec[i] = i_cost
	end
	num_sols = 0
	for i = 1 : num_states
		if costop_vec[i] == 0
			num_sols += 1
		end
	end
	red_sols = sat_prob.red_solutions
	if num_sols != length(red_sols)
		println("Cost-Op num sols don't match known num sols")
		breakhere!()
	else 
		for sol in red_sols
			sol_int = bit_vec_to_int(sol)
			if costop_vec[sol_int] != 0
				println("PROBLEM ", sol_int, " ", costop_vec[sol_int])
				breakhere!()
			end
		end
	end
	return costop_vec
end

function phase_energy(wave_func, costop_vec, alpha)
	if length(wave_func) != length(costop_vec)
		println("unmatch cost and wave funcs ", length(wave_func), " ", length(costop_vec))
		breakhere!()
	end
	phase_vec = [ exp(1.0im * alpha * costop_vec[i]) for i = 1 : length(costop_vec) ]
	return [ phase_vec[i] * wave_func[i] for i = 1 : length(costop_vec) ]
end

function apply_xmixer(wave_func, num_bits, beta)
	num_states = 2^(num_bits)
	for i = 1 : num_bits
		U_trans = SparseMatrixCSC{Complex{Float64}, Int64}(spzeros(num_states, num_states))
		for j = 1 : num_states
			j_bitvec 		= int_to_bit_vec(j, num_bits)
			j_bitvec[i] 	= (j_bitvec[i] + 1) % 2
			j2				= bit_vec_to_int(j_bitvec) 
			U_trans[j , j ] = Complex{Float64}(1/2)
			U_trans[j , j2] = Complex{Float64}(1/2)
			U_trans[j2, j ] = Complex{Float64}(1/2)
			U_trans[j2, j2] = Complex{Float64}(1/2)
		end
		tmp_func = copy(wave_func)
		wave_func = Complex{Float64}(exp(-1.0im * pi * beta)) * U_trans * wave_func
		# println(wave_func)s
		wave_func = tmp_func - U_trans * tmp_func + wave_func
	end
	return wave_func
end

function run_single_inst(sat_prob::SatProblem)
	p_rounds    = 10
	alphas      = [ pi * (p/p_rounds)       for p = 1 : p_rounds ]
	betas       = [ pi * (1 - (p/p_rounds)) for p = 1 : p_rounds ]
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
		sol_vec 	= zeros(Complex{Float64}, num_states)
		sol_vec[ bit_vec_to_int(red_sols[ i ]) ] = Float64(1)
		push!(sol_vecs, sol_vec)
	end	
	# println(sat_prob)
	wave_func 	= zeros(Complex{Float64}, num_states)
	for i = 1 : num_states
		wave_func[i] = (1/sqrt(num_states))
	end
	costop_vec  = make_costop_vec(sat_prob)
	for p = 1 : p_rounds
		fin_sup = calc_sol_support(sol_vecs, wave_func)
		println(fin_sup)
		wave_func = phase_energy(wave_func, costop_vec, alphas[p])
		wave_func = apply_xmixer(wave_func, num_bits, betas[p])
	end
	fin_sup = calc_sol_support(sol_vecs, wave_func)
	println(fin_sup)
	return 0 
end


function run_tqaoa(nbits, clen, mclauses, kinsts)
	dir_str = string("./sat_1in", clen, "_nbits=", nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "/")
	open(string(dir_str,"inst_", 1,".json"), "r") do f 
		json_string = JSON.read(f, String)
		new_sat_prob = Unmarshal.unmarshal(SatProblem, JSON.parse(json_string))
		run_single_inst(new_sat_prob)
	end
	return 0 
end


run_tqaoa(12, 3, 4, 100)
