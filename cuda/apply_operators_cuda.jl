using LoopVectorization 
using CUDA
#using ExponentialUtilities

copytontt!(B, A) = vmapntt!(identity, B, A)

#Diagonal only, otherwise transpose would be required
function cuda_exp(incoming_matrix,k=20)
    res_mat = CuArray(Diagonal(ones(size(incoming_matrix)[1]))) + incoming_matrix
    curr_mat = incoming_matrix 
    fact_term = 1
    for i = 2 : k
        curr_mat = incoming_matrix * curr_mat
        res_mat = res_mat + (curr_mat / fact_term)
        fact_term *= i 
    end
    return res_mat
end


function phase_energy!(wave_func, costop_vec, alpha)
    wave_func = cuda_exp( 1.0im * costop_vec) * wave_func
    nothing 
end

function apply_xmixer(wave_func::Array{Complex{Float64}, 1}, U_xmixer::CUSPARSE.CuSparseMatrixCSC{Complex{Float64}, Int32}, num_bits, beta)
    num_states = 2^(num_bits)
    for i = 1 : num_bits
        ### BUILD MATRIX ###
        for j = 1 : num_states
            j_bitvec        = int_to_bit_vec(j, num_bits)
            j_bitvec[i]     = (j_bitvec[i] + 1) % 2
            j2              = bit_vec_to_int(j_bitvec) 
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
            j_bitvec        = int_to_bit_vec(j, num_bits)
            j_bitvec[i]     = (j_bitvec[i] + 1) % 2
            j2              = bit_vec_to_int(j_bitvec) 
            U_xmixer[j , j ] = Complex{Float64}(0)
            U_xmixer[j , j2] = Complex{Float64}(0)
            U_xmixer[j2, j ] = Complex{Float64}(0)
            U_xmixer[j2, j2] = Complex{Float64}(0)
        end
    end
    return wave_func
end


function apply_all_xmixers(wave_func::CuArray{Complex{Float64}, 1}, copy_wave_func::CuArray{Complex{Float64}, 1}, U_xmixers::Array{CUSPARSE.CuSparseMatrixCSC{Complex{Float64}, Int32}, 1}, num_bits, beta)
    num_states = 2^(num_bits)
    for i = 1 : num_bits
        wave_func = wave_func + (Complex{Float64}(exp(-1.0im * pi * beta) - 1) * U_xmixers[ i ] * wave_func)
    end
    return wave_func 
end


function apply_all_clause_mixers(wave_func::CuVector{Complex{Float64}, Int32}, copy_wave_func::CuVector{Complex{Float64}, Int32}, U_clause_mixers::Array{CUSPARSE.CuSparseMatrixCSC{Complex{Float64}, Int32}, 1}, num_bits, beta)
    num_states = 2^(num_bits)
    for i = 1 : length(U_clause_mixers)
        before_nnz = nnz(wave_func)
        wave_func = wave_func + (Complex{Float64}(exp(-1.0im * pi * beta) - 1) * U_clause_mixers[ i ] * wave_func)
        after_nnz = nnz(wave_func)
        if after_nnz > before_nnz
            println(after_nnz, " ", before_nnz)
            throw(DimensionMismatch)
        end
    
    end
    return wave_func 
end



