def getText(sparse_grid_level, objective, dim, numMCPoints, gridResolution, MC_distribution):
    if objective == 'maxOkushiri':
        response_functions = 1
        analysis_drivers = 'maxOkushiri_for_dakota.py'
    elif objective == 'Okushiri':
        response_functions = 451
        if gridResolution == 16:
            analysis_drivers = 'okushiri_for_dakota_16.py'
        elif gridResolution == 64:
            analysis_drivers = 'okushiri_for_dakota_64.py'
        else:
            print(f'gridResolution {gridResolution} not supported')

    if MC_distribution == 'uniform':
        MC_points_file = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/evalPoints{dim}D_R{gridResolution}_{numMCPoints}.dat'
        MC_evals_file = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/evaluations_{objective}_{dim}D_R{gridResolution}_level{sparse_grid_level}.dat'
        hdf5_output_file = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/results_{objective}_{dim}D_R{gridResolution}_level{sparse_grid_level}'
    elif MC_distribution == 'normal':
        MC_points_file = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/evalPoints{dim}D_R{gridResolution}_{numMCPoints}_normal.dat'
        MC_evals_file = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/evaluations_{objective}_{dim}D_R{gridResolution}_level{sparse_grid_level}_normal.dat'
        hdf5_output_file = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/results_{objective}_{dim}D_R{gridResolution}_level{sparse_grid_level}_normal'

    text = f"""
# Call with 'dakota -i dakota_pce_okushiri.in'

# perform pce on python-given function
environment
    output_precision = 16
    # Write output to file instead of printing it in console
    # output_file = '/home/rehmemk/git/anugasgpp/Okushiri/dakota/output.txt'
    results_output 		# write the results to a file
        hdf5				# write results in hdf5 format instead of plain txt.
            model_selection
                top_method
                #  all_methods
                # all
                #  none
            interface_selection
                #  simulation
                # all
                none
            # name of the results file                
            results_output_file = '{hdf5_output_file}'

method
    id_method = 'Polynomial Chaos'
    polynomial_chaos
    # stoch_collocation
    # quadrature_order = 100		# full Gauss quadrature
    sparse_grid_level = {sparse_grid_level}
    
    # variance_based_decomp   # calculate Sobol indices
    output silent			#do not print the individual function evaluations
    convergence_tolerance = 1.e-16

# export coefficients of PCE to file
# export_expansion_file = '/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/result_pce_coeff.dat'

# evaluate the PCE at points given through file
import_approx_points_file = '{MC_points_file}'
export_approx_points_file = '{MC_evals_file}'


variables
    uniform_uncertain = 6
         lower_bounds = 0.5		0.5		0.5     0.5		0.5		0.5
         upper_bounds = 1.5		1.5		1.5     1.5		1.5     1.5
         descriptors =	'x1'	'x2'	'x3'    'x4'	'x5'	'x6'
	#normal_uncertain = 6
	#	means 			= 1.0       1.0     1.0	    1.0       1.0     1.0	     #1.0     1.0
	#	std_deviations	= 0.125      0.125    0.125    0.125      0.125 0.125    #0.125    0.125
	#	lower_bounds 	= 0.5       0.5      0.5    0.5       0.5      0.5       #0.5      0.5
	#	upper_bounds 	= 1.5       1.5     1.5     1.5       1.5     1.5        #1.5     1.5
	#	descriptors		= 'x1'      'x2'    'x3'    'x4'      'x5'    'x6'       #'x7'    'x8'
# 	lognormal_uncertain = 1
# 		lambdas			= 7.71
# 		zetas		 	=  1.0056 
# 		lower_bounds	= 100.0
# 		upper_bounds 	= 50000.0
# 		descriptors	 	= 'x2'

interface
    fork
        analysis_drivers = '{analysis_drivers}'

responses
# dimensionality of the output
response_functions = {response_functions}
no_gradients
no_hessians
    """
    return text
