using SparseArrays
using LinearAlgebra
using KrylovKit
using MAT
using Glob
using Random
using Optim 
using JSON 
# using StructTypes
using CUDA

include("bitvec_conversions_cuda.jl")

function init_cost_oper(num_bits, clauses, red_sols=[])
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
    tmp = sparse([ i for i = 1 : length(costop_vec) ], [ i for i = 1 : length(costop_vec) ], costop_vec, length(costop_vec), length(costop_vec))
    return CUSPARSE.CuSparseMatrixCSC(tmp)
end

function init_xmixers(num_bits)
    num_states  = 2^(num_bits)
    all_U_xmixer = [ spzeros(Complex{Float64}, num_states, num_states) for i = 1 : num_bits ]
    #all_U_xmixer = [ CUDA.zeros() ]
    for i = 1 : num_bits
        ### BUILD MATRIX ###
        for j = 1 : num_states
            j_bitvec        = int_to_bit_vec(j, num_bits)
            j_bitvec[i]     = (j_bitvec[i] + 1) % 2
            j2              = bit_vec_to_int(j_bitvec) 
            all_U_xmixer[i][j , j ] = Complex{Float64}(1/2)
            all_U_xmixer[i][j , j2] = Complex{Float64}(1/2)
            all_U_xmixer[i][j2, j ] = Complex{Float64}(1/2)
            all_U_xmixer[i][j2, j2] = Complex{Float64}(1/2)
        end
    end
    return [ CUSPARSE.CuSparseMatrixCSC(all_U_xmixer[i]) for i = 1 : length(all_U_xmixer) ]
end

function check_cmixer(U_clause_mixers, cl_id, reds_to_acts)
    num_cnt = 0
    for i = 1 : length(reds_to_acts)
        for j = 1 : length(reds_to_acts)
            if abs(U_clause_mixers[ cl_id ][ i , j ]) > 0.5
                num_cnt += 1
            end
        end
    end
    if nnz(U_clause_mixers[cl_id]) > num_cnt
        rows = rowvals(U_clause_mixers[cl_id])
        println(rows)
        println(vals) 
        println(nnz(U_clause_mixers[cl_id]))
        println(num_cnt)
        println(cl_id)
        breakhere!()
    end
end

function fill_clause_mixers!(all_U_clause_mixers, num_bits, reds_to_acts, sols_per_clause, uncov_vars)
    num_states = 2^(num_bits)
    a_set = Set{Int64}()
    for i = 1 : length(reds_to_acts)
        push!(a_set, reds_to_acts[ i ])
    end
    for cl_id = 1 : length(sols_per_clause)
        for i = 1 : length(reds_to_acts)
            i_val = reds_to_acts[ i ]
            i_vec = int_to_bit_vec(i_val, num_bits)
            for j = 1 : length(sols_per_clause[cl_id])
                skip_val = 0
                for l = 1 : length(sols_per_clause[ cl_id ][ j ])
                    if i_vec[ sols_per_clause[ cl_id ][ j ][ l ][ 1 ] ] != sols_per_clause[ cl_id ][ j ][ l ][ 2 ]
                        skip_val = 1
                        break
                    end
                end
                if skip_val == 1
                    println("SKIPPED")
                    continue
                end
                all_U_clause_mixers[ cl_id ][ i_val, i_val ] = 1.0/length(sols_per_clause[ cl_id ][ j ])
                for k = 1 : length(sols_per_clause[cl_id])
                    if j == k 
                        continue 
                    end
                    # println(sols_per_clause[cl_id][ j ])
                    println(sols_per_clause[cl_id][ j ])
                    println(sols_per_clause[cl_id][ k ])
                    println(length(sols_per_clause[cl_id]))
                    println()
                    for l = 1 : length(sols_per_clause[ cl_id ][ j ])
                        i_vec[ sols_per_clause[cl_id][ j ][ l ][ 1 ] ] = sols_per_clause[cl_id][ k ][ l ][ 2 ]
                    end
                    r_val = bit_vec_to_int(i_vec)
                    if !(r_val in a_set)
                        println(r_val)
                        println(i_val)
                        println(sols_per_clause[cl_id][ k ])
                        breakhere!()
                    end
                    all_U_clause_mixers[ cl_id ][ i_val, r_val ] = 1.0/length(sols_per_clause[cl_id][ j ])
                end
            end
        end 
        check_cmixer(all_U_clause_mixers, cl_id, reds_to_acts)
    end
    mat_id = length(sols_per_clause)
    for i in uncov_vars
        mat_id += 1
        ### BUILD MATRIX ###
        for j = 1 : length(reds_to_acts)
            j_val           = reds_to_acts[ j ]
            j_bitvec        = int_to_bit_vec(j_val, num_bits)
            j_bitvec[i]     = (j_bitvec[i] + 1) % 2
            j_val2          = bit_vec_to_int(j_bitvec) 
            all_U_clause_mixers[ mat_id ][ j_val , j_val  ] = Complex{Float64}(1/2)
            all_U_clause_mixers[ mat_id ][ j_val , j_val2 ] = Complex{Float64}(1/2)
            all_U_clause_mixers[ mat_id ][ j_val2, j_val  ] = Complex{Float64}(1/2)
            all_U_clause_mixers[ mat_id ][ j_val2, j_val2 ] = Complex{Float64}(1/2)
            if !(j_val2 in a_set)
                println(j_val)
                println(j_val2)
                println(sols_per_clause[cl_id][ k ])
                breakhere!()
            end
        end
        check_cmixer(all_U_clause_mixers, cl_id, reds_to_acts)
    end
    for mat_id = 1 : length(all_U_clause_mixers)
        if nnz(all_U_clause_mixers[ mat_id ]) != length(reds_to_acts)^2
            println()
            println(nnz(all_U_clause_mixers[ mat_id ]))
            println(length(reds_to_acts)^2)
            breakhere!()
        end
    end
    nothing 
end


function init_clause_mixers(num_bits, reds_to_acts, sols_per_clause, uncov_vars)
    num_states = 2^(num_bits)
    all_U_clause_mixers = [ spzeros(Complex{Float64}, num_states, num_states) for i = 1 : (length(sols_per_clause) + length(uncov_vars)) ]
    fill_clause_mixers!(all_U_clause_mixers, num_bits, reds_to_acts, sols_per_clause, uncov_vars)
    return [ CUSPARSE.CuSparseMatrixCSC(all_U_clause_mixers[i]) for i = 1 : length(all_U_clause_mixers) ]
end

function init_sol_space(red_sols, num_states)
    sol_vecs = Array{Array{Complex{Float64}, 1}, 1}()
    for i = 1 : length(red_sols)
        sol_vec = zeros(Complex{Float64}, num_states)
        sol_vec[ bit_vec_to_int(red_sols[ i ]) ] = Float64(1)
        push!(sol_vecs, sol_vec)
    end
    return [ CuVector{ComplexF64}(sol_vecs[i]) for i = 1 : length(sol_vecs) ]
end


function init_ut_wavefunc(num_states)
    wave_func = zeros(Complex{Float64}, num_states)
    for i = 1 : num_states
        wave_func[i] = (1/sqrt(num_states))
    end
    return CuVector{ComplexF64}(wave_func)
end

function init_t_wavefunc(num_states, reds_to_acts)
    wave_func = spzeros(Complex{Float64}, num_states)
    for i = 1 : length(reds_to_acts)
        wave_func[i] = (1/sqrt(length(reds_to_acts)))
    end
    return CuVector{ComplexF64}(wave_func)
end

