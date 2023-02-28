struct ExpResults
	pdepth::Int64
	end_eng::Float64
	end_sup::Float64
    num_states::Int64
end

struct AngleParams
	pdepth::Int64
	avg_fin_eng::Float64
	avg_fin_sup::Float64
	num_epochs::Int64
	num_runs::Int64 
	epoch_engs::Array{Float64, 1}
	epoch_sups::Array{Float64, 1}
	alphas::Array{Float64, 1}
	betas::Array{Float64, 1}
end