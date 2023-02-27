include("cuda/tqaoa_cuda.jl")

@time train_general_qaoa(12, 3, Int(ceil(12/3)), 100)