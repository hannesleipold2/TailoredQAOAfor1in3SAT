@time using Glob
@time using MAT
using JSON
@time using Plots 
# @time using StatsPlots
using CurveFit

import Unmarshal

include("base_structs.jl")

function angle_to_color(degree, sv=1.0)
	color = convert(RGB, HSV(degree, 1.0, sv))
	return color 
end

function fix_globs(file_list)
    new_file_list = [ "" ]
    pop!(new_file_list)
    for i = 1 : length(file_list)
        file_str = file_list[ i ]
        push!(new_file_list, replace(file_str, "\\" => "/" ))
    end
    return new_file_list
end

function compute_avg(arr)
	return (1.0 * sum(arr)) / length(arr)
end

function compute_var(arr)
	avg = compute_avg(arr)
	var = 0.0
	for i = 1 : length(arr)
		var += (arr[ i ] - avg)^2
	end
	return var / length(arr) 
end 

function compute_std(arr)
	var = compute_var(arr)
	return sqrt(var)
end

function reverse_dict(some_dict)
	all_keys 	= keys(some_dict)
	(nkeys, first_key) 	= iterate(all_keys)
	println(typeof(nkeys))
	println(typeof(first_key))
	println(nkeys)
	println(first_key)
	println(some_dict[nkeys])
	new_dict = Dict([ (some_dict[ nkeys ], nkeys) ])
	for key in all_keys
		new_dict[  some_dict[ key ] ] =  key
	end
	return new_dict
end


function comp_avg(list_of_vals)
	sum_vals = 0.0
	if length(list_of_vals) == 0
		return -1 
	end
	for i = 1 : length(list_of_vals)
		sum_vals += list_of_vals[ i ]
	end
	return sum_vals / length(list_of_vals)
end
function comp_med(list_of_vals)
	med_ind = Int(ceil(length(list_of_vals) / 2))
	sl 		= sort(list_of_vals)
	return sl[ med_ind ]
end
function comp_lowquant(list_of_vals)
	low_ind = Int(floor(length(list_of_vals) / 4))
	sl 		= sort(list_of_vals)
	if low_ind == 0
		low_ind = 1
	end
	return sl[ low_ind ]
end
function comp_upquant(list_of_vals)
	up_ind = Int(ceil(3 * length(list_of_vals) / 4))
	sl 		= sort(list_of_vals)
	return sl[ up_ind ]
end



function gen_plots(train_nbits, clen, mclauses, kinsts, pdepth, DPI=100)
	train_ut_dir	= string("./train_angles", "_ut", "_clen=", clen, "/")
	train_str 		= string("rand_insts", "_nbits=", train_nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "_pdepth=", pdepth)
	train_path 		= string(train_ut_dir, train_str, ".json")

	nbit_range	= [ 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ]
	### SET UP LOAD/SAVE ###
	ut_dir = string("./results_qaoa_ut_clen=", clen ,"_pdepth=", pdepth, "/")
	if !isdir(ut_dir)
		mkdir(ut_dir)
	end
	succ_vs_nbit_per_inst 			= [ Array{Float64, 1}() for i = 1 : length(nbit_range) ]
	
	low_succ_vs_nbit_per_inst		= [ 0.0 for i = 1 : length(nbit_range) ]
	med_succ_vs_nbit_per_inst		= [ 0.0 for i = 1 : length(nbit_range) ]
	upp_succ_vs_nbit_per_inst		= [ 0.0 for i = 1 : length(nbit_range) ]

	ut_low_size_vs_nbit_per_inst		= [ 0.0 for i = 1 : length(nbit_range) ]
	ut_med_size_vs_nbit_per_inst		= [ 0.0 for i = 1 : length(nbit_range) ]
	ut_upp_size_vs_nbit_per_inst		= [ 0.0 for i = 1 : length(nbit_range) ]

	flatten_nbits_per_inst 			= Array{Float64, 1}()
	flatten_succ_per_inst 			= Array{Float64, 1}()
	for nbit_id = 1 : length(nbit_range)
		nbits 			= nbit_range[nbit_id]
		search_str1 	= string(ut_dir, "inst_nbits=", nbits, "/*.json")
		all_file_strs 	= fix_globs(glob(search_str1))
		# println(all_file_strs)
		size_per_inst 	= Array{Float64, 1}()
		succ_per_inst 	= Array{Float64, 1}()
		for i = 1 : length(all_file_strs)
			open(all_file_strs[ i ], "r") do f 
				json_string = JSON.read(f, String)
				new_res = Unmarshal.unmarshal(ExpResults, JSON.parse(json_string))
				push!(succ_per_inst, new_res.end_sup)
				push!(size_per_inst, new_res.num_states)
				push!(flatten_succ_per_inst, new_res.end_sup)
				push!(flatten_nbits_per_inst, nbits)
			end
		end
		low_succ_vs_nbit_per_inst[ nbit_id ] = comp_lowquant(succ_per_inst)
		med_succ_vs_nbit_per_inst[ nbit_id ] = comp_med(succ_per_inst)
		upp_succ_vs_nbit_per_inst[ nbit_id ] = comp_upquant(succ_per_inst)

		ut_low_size_vs_nbit_per_inst[ nbit_id ] = comp_lowquant(size_per_inst)
		ut_med_size_vs_nbit_per_inst[ nbit_id ] = comp_med(size_per_inst)
		ut_upp_size_vs_nbit_per_inst[ nbit_id ] = comp_upquant(size_per_inst)
	end

	plot_succ_vs_nbits = Plots.plot(legend=:bottomright, title="QAOA vs. tQAOA on 1-in-3 SAT", 
							xlabel="Number of Variables", ylabel="Log of Success Prob")


	Plots.scatter!(	flatten_nbits_per_inst, log.(flatten_succ_per_inst), label=string("QAOA"), 
	seriescolor=angle_to_color(360*(1/3)), markersize=3, markeralpha=0.1, markershape=:cross)

	lsig = [ abs(med_succ_vs_nbit_per_inst[i] - low_succ_vs_nbit_per_inst[i]) + 0.001 for i = 1 : length(med_succ_vs_nbit_per_inst) ]
	usig = [ abs(upp_succ_vs_nbit_per_inst[i] - med_succ_vs_nbit_per_inst[i]) + 0.001 for i = 1 : length(med_succ_vs_nbit_per_inst) ]

	Plots.plot!(nbit_range, log.(med_succ_vs_nbit_per_inst), label="", linewidth=4, linestyle=:solid, 
	seriescolor=angle_to_color(360*(1/3)), ribbon=(lsig, usig), fillalpha=0.1)


	exp_fit = curve_fit(ExpFit, flatten_nbits_per_inst, flatten_succ_per_inst)
	println(exp_fit)

	train_t_dir		= string("./train_angles", "_t", "_clen=", clen, "/")
	train_str 		= string("rand_insts", "_nbits=", train_nbits, "_mclauses=", mclauses, "_kinsts=", kinsts, "_pdepth=", pdepth)
	train_path 		= string(train_ut_dir, train_str, ".json")

	nbit_range	= [ 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ]
	### SET UP LOAD/SAVE ###
	t_dir = string("./results_qaoa_t_clen=", clen ,"_pdepth=", pdepth, "/")
	if !isdir(t_dir)
		mkdir(t_dir)
	end
	succ_vs_nbit_per_inst 			= [ Array{Float64, 1}() for i = 1 : length(nbit_range) ]
	
	t_low_size_vs_nbit_per_inst		= [ 0.0 for i = 1 : length(nbit_range) ]
	t_med_size_vs_nbit_per_inst		= [ 0.0 for i = 1 : length(nbit_range) ]
	t_upp_size_vs_nbit_per_inst		= [ 0.0 for i = 1 : length(nbit_range) ]

	flatten_nbits_per_inst 			= Array{Float64, 1}()
	flatten_succ_per_inst 			= Array{Float64, 1}()
	for nbit_id = 1 : length(nbit_range)
		nbits 			= nbit_range[nbit_id]
		search_str1 	= string(t_dir, "inst_nbits=", nbits, "/*.json")
		all_file_strs 	= fix_globs(glob(search_str1))
		# println(all_file_strs)
		size_per_inst 	= Array{Float64, 1}()
		succ_per_inst 	= Array{Float64, 1}()
		for i = 1 : length(all_file_strs)
			open(all_file_strs[ i ], "r") do f 
				json_string = JSON.read(f, String)
				new_res = Unmarshal.unmarshal(ExpResults, JSON.parse(json_string))
				push!(succ_per_inst, new_res.end_sup)
				push!(size_per_inst, new_res.num_states)
				push!(flatten_succ_per_inst, new_res.end_sup)
				push!(flatten_nbits_per_inst, nbits)
			end
		end
		low_succ_vs_nbit_per_inst[ nbit_id ] = comp_lowquant(succ_per_inst)
		med_succ_vs_nbit_per_inst[ nbit_id ] = comp_med(succ_per_inst)
		upp_succ_vs_nbit_per_inst[ nbit_id ] = comp_upquant(succ_per_inst)

		t_low_size_vs_nbit_per_inst[ nbit_id ] = comp_lowquant(size_per_inst)
		t_med_size_vs_nbit_per_inst[ nbit_id ] = comp_med(size_per_inst)
		t_upp_size_vs_nbit_per_inst[ nbit_id ] = comp_upquant(size_per_inst)
	end


	Plots.scatter!(	flatten_nbits_per_inst, log.(flatten_succ_per_inst), label=string("tQAOA"), 
	seriescolor=angle_to_color(360*(2/3)), markersize=3, markeralpha=0.1, markershape=:cross)

	lsig = [ abs(med_succ_vs_nbit_per_inst[i] - low_succ_vs_nbit_per_inst[i]) + 0.001 for i = 1 : length(med_succ_vs_nbit_per_inst) ]
	usig = [ abs(upp_succ_vs_nbit_per_inst[i] - med_succ_vs_nbit_per_inst[i]) + 0.001 for i = 1 : length(med_succ_vs_nbit_per_inst) ]

	Plots.plot!(nbit_range, log.(med_succ_vs_nbit_per_inst), label="", linewidth=4, linestyle=:solid, 
	seriescolor=angle_to_color(360*(2/3)), ribbon=(lsig, usig), fillalpha=0.1)

	exp_fit2 = curve_fit(ExpFit, flatten_nbits_per_inst, flatten_succ_per_inst)
	println(exp_fit2)


	# Plots.plot!(xlim=0, ylim=1.0, legend=:bottomleft)
	Plots.plot!(dpi=DPI)
	DO_SAVE = 1
	if DO_SAVE == 1
		savefig(plot_succ_vs_nbits, string("./plots/", "plot_succ_vs_nbits_pdepth=", pdepth, ".png"))
	end

	#=
	plot_size_vs_nbits = Plots.plot(legend=:bottomright, title="QAOA vs. tQAOA on 1-in-3 SAT", 
	xlabel="Number of Variables", ylabel="Log of Search Space Dim")

	utlsig = [ abs(ut_med_size_vs_nbit_per_inst[i] - ut_low_size_vs_nbit_per_inst[i]) + 0.001 for i = 1 : length(ut_med_size_vs_nbit_per_inst) ]
	utusig = [ abs(ut_upp_size_vs_nbit_per_inst[i] - ut_med_size_vs_nbit_per_inst[i]) + 0.001 for i = 1 : length(ut_med_size_vs_nbit_per_inst) ]

	Plots.plot!(nbit_range, ut_med_size_vs_nbit_per_inst, label="QAOA", linewidth=4, linestyle=:solid, 
	seriescolor=angle_to_color(360*(1/3)), ribbon=(utlsig, utusig), fillalpha=0.1)

	tlsig = [ abs(t_med_size_vs_nbit_per_inst[i] - t_low_size_vs_nbit_per_inst[i]) + 0.001 for i = 1 : length(t_med_size_vs_nbit_per_inst) ]
	tusig = [ abs(t_upp_size_vs_nbit_per_inst[i] - t_med_size_vs_nbit_per_inst[i]) + 0.001 for i = 1 : length(t_med_size_vs_nbit_per_inst) ]

	Plots.plot!(nbit_range, log.(t_med_size_vs_nbit_per_inst), label="tQAOA", linewidth=4, linestyle=:solid, 
	seriescolor=angle_to_color(360*(2/3)), ribbon=(tlsig, tusig), fillalpha=0.1)

	Plots.plot!(dpi=DPI)
	DO_SAVE = 1
	if DO_SAVE == 1
		savefig(plot_size_vs_nbits, string("./plots/", "plot_size_vs_nbits.png"))
	end
	=#
end 

# gen_plots(12, 3, 4, 100, 14, 300)
# gen_plots(12, 3, 4, 100, 60, 300)

gen_plots(12, 3, 4, 100, 14, 100)
gen_plots(12, 3, 4, 100, 60, 100)