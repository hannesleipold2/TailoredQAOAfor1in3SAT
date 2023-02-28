using Dates
start_compile_time = Dates.now()
using SparseArrays
using BenchmarkTools
# using CuArrays
using LinearAlgebra 
end_compile_time = Dates.now()
println("COMPILE TIME: ", end_compile_time - start_compile_time)

include("bitvec_conversions.jl")

#=
	0.5       [0,0]
	0.25      [0,1]
	0.125     [1,0]
	0.125     [1,1]
	([0, 0] + [0, 1])   -> 0.5 ([0, 0] + [0, 1]) + 0.5 ([1,0] + [1,1])
	
	[0, x...x]          -> 0.5 [0, x...x] + 0.5 [ 1, x....x ]
	[x, 0, x,..., x]    -> 0.5 [x, 0, x, ... x] + 0.5 [  ]
=#

function apply_new_xmixer!(wave_func::Array{Complex{Float64}, 1}, num_bits, beta)
	num_states = 2^(num_bits)
	before_ket0_at_i = zeros(Complex{Float64}, num_states)
	before_ket1_at_i = zeros(Complex{Float64}, num_states)
	for i = 1 : num_bits
		for j = 1 : num_states
			if int_to_bit_vec(j, num_bits)[i] == 0
				before_ket0_at_i[j] = wave_func[j]
			else
				before_ket1_at_i[j] = wave_func[j]
			end
		end
		ket0_norm = norm(before_ket0_at_i)
		before_ket0_at_i /= ket0_norm
		ket1_norm = norm(before_ket1_at_i)
		before_ket1_at_i /= ket1_norm

		a = dot(before_ket0_at_i, wave_func)
		b = dot(before_ket1_at_i, wave_func)
		copyto!(wave_func, ((a * cos(beta)) + (b * 1.0im * sin(beta))) * (before_ket0_at_i/norm(before_ket0_at_i)))
		wave_func += ((a * 1.0im * sin(beta)) + (b * cos(beta))) * (before_ket1_at_i/norm(before_ket1_at_i))
		for j = 1 : num_states
			if int_to_bit_vec(j, num_bits)[i] == 0
				before_ket0_at_i[j] = 0
			else
				before_ket1_at_i[j] = 0
			end
		end
	end
	return nothing 
end

function apply_xmixer(wave_func::Array{Complex{Float64}, 1}, U_trans::SparseMatrixCSC{Complex{Float64}, Int64}, num_bits, beta)
	num_states = 2^(num_bits)
	wave_func2 = copy(wave_func)
	for i = 1 : num_bits
		pow_i = 2^(i-1)
		### BUILD MATRIX ###
		for j = 1 : num_states
			# j_bitvec 		= int_to_bit_vec(j, num_bits)
			# j_bitvec[i] 	= (j_bitvec[i] + 1) % 2
			j2				= xor(j-1, pow_i)+1 
			# println(j, " ", j2)
			U_trans[j , j ] = Complex{Float64}(cos(beta))
			U_trans[j , j2] = Complex{Float64}(-1.0im * sin(beta))
			U_trans[j2, j ] = Complex{Float64}(-1.0im * sin(beta))
			U_trans[j2, j2] = Complex{Float64}(cos(beta))
		end
		wave_func2 = U_trans * wave_func
		### CLEAR MATRIX ###
		for j = 1 : num_states
			j2				= xor(j-1, pow_i)+1
			# j_bitvec 		= int_to_bit_vec(j, num_bits)
			# j_bitvec[i] 	= (j_bitvec[i] + 1) % 2
			# j2				= bit_vec_to_int(j_bitvec) 
			U_trans[j , j ] = Complex{Float64}(0)
			U_trans[j , j2] = Complex{Float64}(0)
			U_trans[j2, j ] = Complex{Float64}(0)
			U_trans[j2, j2] = Complex{Float64}(0)
		end
		dropzeros!(U_trans)
	end
	return wave_func
end


#=
	2^(2) bit_vector
=#
function unit_tester(num_bits)
	num_states  = 2^(num_bits)
	wave_time   = Dates.now()
	wave_func   = [ (1.0 + 0.0im)/sqrt(num_states) for i = 1 : num_states ]
	println("WFUNC BUILD TIME: ", Dates.now() - wave_time)
	mat_time    = Dates.now()
	U_mixer     = spzeros(Complex{Float64}, num_states, num_states)
	println("UMAT BUILD TIME: ", Dates.now() - mat_time)
	NUM_RUNS    = 1
	start_time  = Dates.now()
	for i = 1 : NUM_RUNS
		println("NORM B: ", norm(wave_func))
		iter_time = Dates.now()
		wave_func2 = apply_xmixer(wave_func, U_mixer, num_bits, 0.01)
		println(Dates.now() - iter_time)
		println("NORM 1: ", norm(wave_func2))
		apply_new_xmixer!(wave_func, num_bits, 0.01)
		println(Dates.now() - iter_time)
		println("NORM 2: ", norm(wave_func))
		println(dot(wave_func, wave_func2))
		end_time  = Dates.now()
		delta_time= end_time - iter_time
		if (i-1)%(round(NUM_RUNS/10)) == 0
			println(round((i-1)/(round(NUM_RUNS/10))), "0% done: ", delta_time)
		end
	end
	end_time  = Dates.now()
	println("TOTAL TIME: ", end_time - start_time)
end


unit_tester(16)



