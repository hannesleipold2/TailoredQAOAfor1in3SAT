using Random 
include("qaoa_circ.jl")

function gen_costop!(costop_vec, clauses, num_bits)
	num_states 	= 2^(num_bits)
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
	return costop_vec
end

function rand_parameter_shift(	wave_func::Array{Complex{Float64}, 1}, H_phase::Array{Complex{Float64}, 1}, U_mix::SparseMatrixCSC{Complex{Float64}, Int64}, 
								batch_clauses, alphas, betas, pdepth, shift_size, grad_coef)
	rand_id = rand(1:pdepth)
	rand_bit = rand(1:0)
	if rand_bit == 1
		alphas[rand_id] += shift_size
	else 
		betas[rand_id] 	+= shift_size
	end
	for batch_clause in batch_clauses
		gen_costop!(H_phase, clauses, num_bits)
		fin_eng1, fin_sup1 = [ 0, 0 ]
	end
	if rand_id == 1
		alphas[rand_id] -= 2 * shift_size
	else 
		betas[rand_id] 	-= 2 * shift_size
	end
	fin_eng2, fin_sup2 = [ 0, 0 ]
	appro_grad = (fin_eng1 - fin_eng2)
	if rand_id == 1
		alphas[rand_id] += shift_size + grad_coef * appro_grad
	else 
		betas[rand_id] 	+= shift_size + grad_coef * appro_grad
	end
end



function train(sat_probs::Array{SatProblem, 1}, alphas, betas, pdepth=10, batch_size=10)
	## filter batches to have the same overall reduced size
	first_sat_id = rand(1:length(sat_probs))
	
	return 0 
end





























