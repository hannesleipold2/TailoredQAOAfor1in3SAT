function find_sols_per_clause(num_bits, clauses)
	sols = Array{Array{Array{Array{Int64, 1}, 1}, 1}, 1}()
	for cl_id = 1 : length(clauses) 
		clause = clauses[cl_id]
		clause_sol = Array{Array{Array{Int64, 1}, 1}, 1}()
		vars = Array{Int64, 1}() 
		for lit_id = 1 : length(clause)
			push!(vars, clause[lit_id].variable) 
		end
		for i = 1 : 2^length(clause)
			pot_sol = int_to_bit_vec(i, length(clause)) 
			num_sat = 0
			for lit_id = 1 : length(clause)
				if pot_sol[lit_id] == clause[lit_id].expected_value
					num_sat += 1
				end
			end
			if num_sat == 1 
				push!(clause_sol, [ [ vars[ i ], pot_sol[ i ] ] for i = 1 : length(clause) ])
			end
		end
		push!(sols, clause_sol)
	end
	return sols
end 

function rec_tailor_subspace(curr_config, reds_to_acts, state_cntp, sols_per_cl, uncov_vars, num_bits, cl_id, uncov_id)
	if cl_id == length(sols_per_cl) + 1
		if uncov_id == length(uncov_vars) + 1
			ret_vec = [ -1 for i = 1 : num_bits ]
			for con_id = 1 : length(curr_config)
				var_id, bit_val = curr_config[con_id]
				ret_vec[var_id] = bit_val 
			end
			for i = 1 : num_bits
				if ret_vec[i] < 0
					println(ret_vec)
					throw(DomainError)
				end
			end
			# println(ret_vec)
			ret_int                         = bit_vec_to_int(ret_vec)
			state_cntp[ 1 ]                 = state_cntp[ 1 ] + 1
			reds_to_acts[ state_cntp[ 1 ] ] = ret_int 
		else
			for var in uncov_vars
				push!(curr_config, [ var, 0 ])
				rec_tailor_subspace(curr_config, reds_to_acts, state_cntp, sols_per_cl, uncov_vars, num_bits, cl_id, uncov_id+1)
				pop!(curr_config)
				push!(curr_config, [ var, 1 ])
				rec_tailor_subspace(curr_config, reds_to_acts, state_cntp, sols_per_cl, uncov_vars, num_bits, cl_id, uncov_id+1)
				pop!(curr_config)		
			end
		end
	else 
		for sol_id = 1 : length(sols_per_cl[cl_id])
			for var_id = 1 : length(sols_per_cl[cl_id][sol_id])
				# println(curr_config)
				# println(sols_per_cl[cl_id][sol_id][var_id])
				push!(curr_config, sols_per_cl[cl_id][sol_id][var_id])
			end
			rec_tailor_subspace(curr_config, reds_to_acts, state_cntp, sols_per_cl, uncov_vars, num_bits, cl_id+1, uncov_id)
			for var_id = 1 : length(sols_per_cl[cl_id][sol_id])
				pop!(curr_config)
			end
		end
	end
end

function find_num_tailor_states(dis_clauses, uncov_vars)
	return (3^(length(dis_clauses))) * 2^(length(uncov_vars))
end

function cost_oper(num_bits, clauses, red_sols=[])
	num_states  = 2^(num_bits)
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
	if length(red_sols) > 0 && num_sols != length(red_sols)
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


function find_tailor_subspace(sat_prob::SatProblem)
	num_bits    = sat_prob.num_red_variables
	num_states  = 2^(num_bits)
	clauses     = sat_prob.red_clauses
	dis_clauses = sat_prob.red_max_disjoint_clauses
	uncov_vars	= sat_prob.red_variables_uncovered
	clen        = sat_prob.vars_per_clause
	sols_per_cl = find_sols_per_clause(num_bits, dis_clauses)
	reds_to_acts= [ 0 for i = 1 : find_num_tailor_states(dis_clauses, uncov_vars) ]
	wave_func  	= spzeros(Complex{Float64}, num_states)   
	curr_config	= Array{Array{Int64, 1}, 1}()
	state_cntp  = [ 0 ]
	rec_tailor_subspace(curr_config, reds_to_acts, state_cntp, sols_per_cl, uncov_vars, num_bits, 1, 1)
	# println(uncov_vars)
	# println(sols_per_cl)
	# println(clauses)
	# println(wave_func)
	# println(num_states)
	# println(nnz(wave_func))
	# costop_vec = cost_oper(num_bits, dis_clauses)
	#=
	for i = 1 : length(reds_to_acts)
		println(costop_vec[ reds_to_acts[ i ] ])
	end
	breakhere!()
	=# 
	return reds_to_acts
end