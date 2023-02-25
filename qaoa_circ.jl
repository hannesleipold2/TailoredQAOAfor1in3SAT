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
include("measures_n_reps.jl")

using .SatGenerator


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
		throw(DomainError)
	else 
		for sol in red_sols
			sol_int = bit_vec_to_int(sol)
			if costop_vec[sol_int] != 0
				println("PROBLEM ", sol_int, " ", costop_vec[sol_int])
				throw(DomainError)
			end
		end
	end
	return costop_vec
end

function phase_energy!(wave_func, costop_vec, alpha)
	if length(wave_func) != length(costop_vec)
		println("unmatch cost and wave funcs ", length(wave_func), " ", length(costop_vec))
		throw(DimensionMismatch)
	end
	for i = 1 : length(costop_vec)
		wave_func[i] *= exp(1.0im * alpha * costop_vec[i]) 
	end
	nothing 
end


function apply_xmixer(wave_func::Array{Complex{Float64}, 1}, U_trans::SparseMatrixCSC{Complex{Float64}, Int64}, num_bits, beta)
	num_states = 2^(num_bits)
	for i = 1 : num_bits
		### BUILD MATRIX ###
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
		# println(wave_func)
		wave_func = tmp_func - U_trans * tmp_func + wave_func
		### CLEAR MATRIX ###
		for j = 1 : num_states
			j_bitvec 		= int_to_bit_vec(j, num_bits)
			j_bitvec[i] 	= (j_bitvec[i] + 1) % 2
			j2				= bit_vec_to_int(j_bitvec) 
			U_trans[j , j ] = Complex{Float64}(0)
			U_trans[j , j2] = Complex{Float64}(0)
			U_trans[j2, j ] = Complex{Float64}(0)
			U_trans[j2, j2] = Complex{Float64}(0)
		end
	end
	return wave_func
end

function run_qaoa_circ(	wave_func::Array{Complex{Float64}}, costop::Array{Complex{Float64}}, U_mixer::SparseMatrixCSC{Complex{Float64}, Int64}, 
						alphas, betas, pdepth, num_bits)
	for i = 1 : pdepth
		phase_energy!(wave_func, costop)
		wave_func = apply_xmixer(wave_func::Array{Complex{Float64}, 1}, U_trans::SparseMatrixCSC{Complex{Float64}, Int64}, num_bits, beta)
	end
	fin_eng = 0 
end


function initialize_wave_func(sat_prob::SatProblem)
    
    return 0 
end