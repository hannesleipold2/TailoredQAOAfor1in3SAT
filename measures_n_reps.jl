
function calc_sol_support(sol_vecs::Array{Array{Complex{Float64},1},1}, wave_func::Array{Complex{Float64},1})
	sol_support = 0.0
	for i = 1 : length(sol_vecs)
		sol_support += abs(dot(sol_vecs[ i ], wave_func / norm(wave_func))) ^ 2
	end
	return sol_support
end


function print_supstates(wave_func, sol_vecs, H_final, state_len)
	full_support = 0.0 
	println()
	for state_val = 1 : 2^(state_len)
		state_sup 		= abs(wave_func[ state_val ])^2 
		full_support 	+= state_sup
		full_state 		= int_to_bit_vec(state_val, state_len)
		print(string(state_sup)[1:5], "   ", H_final[ state_val, state_val ], "   ")
		println(full_state)
	end
	println()
	for sol_vec in sol_vecs 
		state_sup 		= abs(dot(wave_func, sol_vec))^2 
		println(string(state_sup)[1:5], "   ", abs(dot(sol_vec, H_final * sol_vec)), "   ")
	end
	println()
	println(full_support)
	println(abs(dot(wave_func, H_final * wave_func)))
	println(norm(wave_func))
end