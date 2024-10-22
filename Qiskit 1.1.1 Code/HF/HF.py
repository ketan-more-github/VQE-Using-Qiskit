import numpy as np
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import Aer
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms import MinimumEigensolverResult
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver , VQE
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
from qiskit_algorithms.optimizers import SLSQP , SPSA , ADAM
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import Aer
from qiskit_algorithms import MinimumEigensolverResult
from qiskit_aer.primitives import Estimator as est
from qiskit.circuit.library import EfficientSU2



    #    molecule_dict = {"H2":f"H .0 .0 .0; H .0 .0 {dist}",
    #                      "HF":f"H .0 .0 .0; F .0 .0 {dist}",
    #                      "LiH":f"H .0 .0 .0; Li .0 .0 {dist}",
    #                      "H2O":"O .0 .0 .0; H 0.757 0.586 0.0; H -0.757 0.586 0.0"}

try:
    backend = Aer.get_backend('aer_simulator_statevector_gpu')
    backend.set_options(device='GPU')

except Exception as e:
    print(f"Failed to set GPU options: {e}")

molecule = MoleculeInfo(
        # Coordinates in Angstrom
        symbols=["H", "F"],
        coords=([0.0, 0.0, 0.0], [0.0, 0.0 ,0.9555]),
        multiplicity=1,  # = 2*spin + 1
        charge=0,
    )


driver = PySCFDriver.from_molecule(molecule)
properties = driver.run()

problem = FreezeCoreTransformer(
    freeze_core=True, remove_orbitals=[6]
).transform(properties)

num_particles = problem.num_particles
num_spatial_orbitals = problem.num_spatial_orbitals

mapper = ParityMapper(num_particles=num_particles)
hamiltonian = mapper.map(problem.second_q_ops()[0])    


var_form = EfficientSU2(hamiltonian.num_qubits)


num_params1 = var_form.num_parameters
print(f"using usscd = {num_params1}")



target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)

ansatz_isa = pm.run(var_form)

hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

def cost_func(params, ansatz, hamiltonian, estimator):
 
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")
 

    cost_history_dict["cost_history"].clear()

    return energy


cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}

x0 =  np.pi * np.random.random(num_params1)

import time

classical_time = 0
device_time = 0
ttl_device_time = 0

def timed_cost_func(x, *args):
    global device_time , ttl_device_time
    ansatz_isa, hamiltonian_isa, estimator = args
    start_device_time = time.time()
    
    cost = cost_func(x, ansatz_isa, hamiltonian_isa, estimator )

    end_device_time = time.time()
    device_time = end_device_time - start_device_time
    ttl_device_time += device_time
    
    return cost



def callback(xk):
    global classical_time
    
    print(f"Current parameters: {xk}")
    print(f"Device (GPU) time: {device_time:.4f} seconds")

# optimizer = SLSQP(maxiter=40)
# aer_estimator = est(approximation=True)
# vqe = VQE(aer_estimator, var_form, optimizer)
# vqe.initial_point = [0] * var_form.num_parameters
# algorithm = GroundStateEigensolver(mapper, vqe)
# electronic_structure_result = algorithm.solve(problem)
# electronic_structure_result.formatting_precision = 6
# print(f"ground state energy = {electronic_structure_result.groundenergy}")
# print(electronic_structure_result)

print()
with Session(backend=backend) as session:
    estimator = Estimator(mode=session)
    estimator.options.default_shots = 2046  #4096

    start_classical_time = time.time()
    res = minimize(
        timed_cost_func,
        x0,
        args=(ansatz_isa, hamiltonian_isa, estimator),
        method="COBYLA",
        callback=callback,
    
        
    )
    end_classical_time = time.time()
    classical_time += end_classical_time - start_classical_time
    print("")
    print(f"Total time: {classical_time:.10f} seconds")
    print(f"Total device time: {ttl_device_time:.10f} seconds")
    print(f"Total classical optimizer time { classical_time - ttl_device_time:.10f} " )




 
print(res)
print()
result = MinimumEigensolverResult()
result.eigenvalue = res.fun
interpreted_result = problem.interpret(result)
print(interpreted_result)

print("calculting using numpy")

sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
result = problem.interpret(sol)
print(result)


