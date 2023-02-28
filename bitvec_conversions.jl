function bit_vec_to_int(vec::Array{Int64, 1})
    outval = Int64(0 * vec[ 1 ])
    for i = 1 : length(vec)
        outval += Int64(2^(i-1) * vec[ i ])
    end
    return Int64(outval + 1)
end


function bit_vec_to_int(vec::Array{Int, 1})
    outval = 0 * vec[ 1 ]
    for i = 1 : length(vec)
        outval += 2^(i-1) * vec[ i ]
    end
    return outval + 1
end

function int_to_bit_vec(int_val::Int, bit_length::Int)
    if int_val > 2^(bit_length)
        println("val larger than num bits allows")
        println(int_val)
        throw(DomainError)
    end
    outvec      = zeros(Int64, bit_length)
    curr_int    = int_val - 1
    for i = 1 : bit_length
        rem         = curr_int % 2
        curr_int    ÷= 2
        outvec[ i ] = rem
    end
    return outvec
end
