using LoopVectorization 


copytontt!(B, A) = vmapntt!(identity, B, A)

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

function phase_subspace_energy!(wave_func, costop_vec, reds_to_acts, alpha)
	if length(wave_func) != length(costop_vec)
		println("unmatch cost and wave funcs ", length(wave_func), " ", length(costop_vec))
		throw(DimensionMismatch)
	end
	for i = 1 : length(reds_to_acts)
		wave_func[ reds_to_acts[ i ] ] *= exp(1.0im * alpha * costop_vec[ reds_to_acts[ i ] ]) 
	end
	nothing 
end



function apply_xmixer(wave_func::Array{Complex{Float64}, 1}, U_xmixer::SparseMatrixCSC{Complex{Float64}, Int64}, num_bits, beta)
	num_states = 2^(num_bits)
	for i = 1 : num_bits
		### BUILD MATRIX ###
		for j = 1 : num_states
			j_bitvec 		= int_to_bit_vec(j, num_bits)
			j_bitvec[i] 	= (j_bitvec[i] + 1) % 2
			j2				= bit_vec_to_int(j_bitvec) 
			U_xmixer[j , j ] = Complex{Float64}(1/2)
			U_xmixer[j , j2] = Complex{Float64}(1/2)
			U_xmixer[j2, j ] = Complex{Float64}(1/2)
			U_xmixer[j2, j2] = Complex{Float64}(1/2)
		end
		tmp_func = copy(wave_func)
		wave_func = Complex{Float64}(exp(-1.0im * pi * beta)) * U_xmixer * wave_func
		# println(wave_func)
		wave_func = tmp_func - U_xmixer * tmp_func + wave_func
		### CLEAR MATRIX ###
		for j = 1 : num_states
			j_bitvec 		= int_to_bit_vec(j, num_bits)
			j_bitvec[i] 	= (j_bitvec[i] + 1) % 2
			j2				= bit_vec_to_int(j_bitvec) 
			U_xmixer[j , j ] = Complex{Float64}(0)
			U_xmixer[j , j2] = Complex{Float64}(0)
			U_xmixer[j2, j ] = Complex{Float64}(0)
			U_xmixer[j2, j2] = Complex{Float64}(0)
		end
	end
	return wave_func
end


function apply_all_xmixers(wave_func::Array{Complex{Float64}, 1}, copy_wave_func::Array{Complex{Float64}, 1}, U_xmixers::Array{SparseMatrixCSC{Complex{Float64}, Int64}, 1}, num_bits, beta)
	num_states = 2^(num_bits)
	for i = 1 : num_bits
		wave_func = wave_func + (Complex{Float64}(exp(-1.0im * pi * beta) - 1) * U_xmixers[ i ] * wave_func)
	end
	return wave_func 
end


function apply_all_clause_mixers(wave_func::SparseVector{Complex{Float64}, Int64}, copy_wave_func::SparseVector{Complex{Float64}, Int64}, U_clause_mixers::Array{SparseMatrixCSC{Complex{Float64}, Int64}, 1}, num_bits, beta)
	num_states = 2^(num_bits)
	for i = 1 : length(U_clause_mixers)
		wave_func 	= wave_func + (Complex{Float64}(exp(-1.0im * pi * beta) - 1) * U_clause_mixers[ i ] * wave_func)
	end
	return wave_func 
end



