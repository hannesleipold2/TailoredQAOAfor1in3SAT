using Dates
start_compile_time = Dates.now()
using SparseArrays
using LinearAlgebra
end_compile_time = Dates.now()
println("COMPILE TIME: ", end_compile_time - start_compile_time)

include("bitvec_conversions.jl")

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
    NUM_RUNS    = 100
    start_time  = Dates.now()
    for i = 1 : NUM_RUNS
        iter_time = Dates.now()
        wave_func = apply_xmixer(wave_func, U_mixer, num_bits, 0.01)
        end_time  = Dates.now()
        delta_time= end_time - iter_time
        if (i-1)%(round(NUM_RUNS/10)) == 0
            println(delta_time)
        end
    end      
    end_time  = Dates.now()
    println("TOTAL TIME: ", end_time - start_time)
end


unit_tester(14)



