import warnings
import numpy as np
from numpy import cos, sin, sqrt
import numba
from numba import njit
from elastica.joint import FreeJoint
from elastica.utils import Tolerance
from free_simulator import FreeAssembly

# Join the two rods
from elastica._linalg import _batch_norm, _batch_cross, _batch_matvec, _batch_dot, _batch_matmul, _batch_matrix_transpose
from elastica.interaction import (
    elements_to_nodes_inplace,
    node_to_element_position,
    node_to_element_velocity,
)

from elastica._rotations import _inv_skew_symmetrize

@njit(cache=True)#用numba编译，加快速度
def _single_get_rotation_matrix(theta:float, unit_axis):#反转数组
    rot_mat = np.empty((3, 3))

    v0 = unit_axis[0]
    v1 = unit_axis[1]
    v2 = unit_axis[2]

    u_prefix = sin(theta)
    u_sq_prefix = 1.0 - cos(theta)

    rot_mat[0, 0] = 1.0 - u_sq_prefix * (v1 * v1 + v2 * v2)
    rot_mat[1, 1] = 1.0 - u_sq_prefix * (v0 * v0 + v2 * v2)
    rot_mat[2, 2] = 1.0 - u_sq_prefix * (v0 * v0 + v1 * v1)

    rot_mat[0, 1] = u_prefix * v2 + u_sq_prefix * v0 * v1
    rot_mat[1, 0] = -u_prefix * v2 + u_sq_prefix * v0 * v1
    rot_mat[0, 2] = -u_prefix * v1 + u_sq_prefix * v0 * v2
    rot_mat[2, 0] = u_prefix * v1 + u_sq_prefix * v0 * v2
    rot_mat[1, 2] = u_prefix * v0 + u_sq_prefix * v1 * v2
    rot_mat[2, 1] = -u_prefix * v0 + u_sq_prefix * v1 * v2

    return rot_mat

@njit(cache=True)
def _single_inv_rotate(director):#单独反向旋转？
    vector = np.empty((3))

    vector[0] = director[2,1]-director[1,2]
    vector[1] = director[0,2]-director[2,0]
    vector[2] = director[1,0]-director[0,1]
    trace = director[0,0] + director[1,1] + director[2,2]

    rtol = 1e-5
    atol = 1e-8
    if np.abs(trace - 3) <= (atol+rtol*3):
    #if np.isclose(trace, 3):
        multiplier = (0.5-(trace-3.0)/12.0)
        vector *= multiplier
        #warnings.warn("Misalignment trace close to 3", RuntimeWarning)
    elif np.abs(trace + 1) <= (atol+rtol):
    #elif np.isclose(trace, -1):
        a = np.argmax(np.diag(director))
        b = (a+1) % 3
        c = (a+2) % 3
        s = np.sqrt(director[a,a] - director[b,b] - director[c,c] + 1)
        v = np.array([
                s/2,
                (1/(2*s))*(director[b,a]+director[a,b]),
                (1/(2*s))*(director[c,a]+director[a,c]),
            ])
        norm_v = np.sqrt(np.sum(v*v))
        vector = np.pi * v / norm_v
    else:
        theta = np.arccos(0.5 * trace - 0.5)
        multiplier = -0.5 * theta / np.sin(theta+1e-14)
        vector *= multiplier

    return vector

@njit(cache=True)
def _inv_rotate(director_collection):#反向旋转
    blocksize = director_collection.shape[2]
    vector_collection = np.empty((3, blocksize))

    for k in range(blocksize):
        vector_collection[0, k] = director_collection[2,1,k]-director_collection[1,2,k]
        vector_collection[1, k] = director_collection[0,2,k]-director_collection[2,0,k]
        vector_collection[2, k] = director_collection[1,0,k]-director_collection[0,1,k]
        trace = director_collection[0,0,k] + director_collection[1,1,k] + director_collection[2,2,k]

        rtol = 1e-5
        atol = 1e-8
        if np.abs(trace - 3) <= (atol+rtol*3):
        #if np.isclose(trace, 3):
            multiplier = (0.5-(trace-3.0)/12.0)
            vector_collection[:, k] *= multiplier
            #warnings.warn("Misalignment trace close to 3", RuntimeWarning)
        elif np.abs(trace + 1) <= (atol+rtol):
        #elif np.isclose(trace, -1):
            a = np.argmax(np.diag(director_collection[:,:,k]))
            b = (a+1) % 3
            c = (a+2) % 3
            s = np.sqrt(director_collection[a,a,k] - director_collection[b,b,k] - director_collection[c,c,k] + 1)
            v = np.array([
                    s/2,
                    (1/(2*s))*(director_collection[b,a,k]+director_collection[a,b,k]),
                    (1/(2*s))*(director_collection[c,a,k]+director_collection[a,c,k]),
                ])
            norm_v = np.sqrt(np.sum(v*v))
            vector_collection[:, k] = np.pi * v / norm_v
        else:
            theta = np.arccos(0.5 * trace - 0.5)
            multiplier = -0.5 * theta / np.sin(theta+1e-14)
            vector_collection[:, k] *= multiplier

    return vector_collection

class SurfaceJointSideBySide(FreeJoint):
    """"""

    def __init__(self, k, nu, kt, rd1_local, rd2_local, stability_check=False):
        super().__init__(k, nu)
        # additional in-plane constraint through restoring torque 通过恢复扭矩附加平面内约束
        # stiffness of the restoring constraint -- tuned empirically 恢复约束的刚度——根据经验调整
        # self.kr = 1e2
        self.kt = kt
        self.rd1_local = rd1_local
        self.rd2_local = rd2_local
        self._flag_initialize_To = True

        self.stability_check = stability_check

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        # TODO: documentation

        self.rod_one_rd2, self.rod_two_rd2, self.spring_force = self._apply_forces(
            self.k,
            self.nu,
            rod_one.mass,
            rod_two.mass,
            self.rd1_local,
            self.rd2_local,
            rod_one.position_collection,
            rod_two.position_collection,
            rod_one.radius[None,:],
            rod_two.radius[None,:],
            rod_one.velocity_collection,
            rod_two.velocity_collection,
			rod_one.director_collection,
			rod_two.director_collection,
			rod_one.tangents,
			rod_two.tangents,
            rod_one.external_forces,
            rod_two.external_forces,
        )

    @staticmethod#不需要创建class就可以直接使用
    @njit(cache=True)
    def _apply_forces(
        k,
        nu,
        rod_one_mass,
        rod_two_mass,
		rod_one_rd2_local,
		rod_two_rd2_local,
        rod_one_position_collection,
        rod_two_position_collection,
        rod_one_radius,
        rod_two_radius,
        rod_one_velocity_collection,
        rod_two_velocity_collection,
        rod_one_director_collection,
        rod_two_director_collection,
        rod_one_tangents,
        rod_two_tangents,
        rod_one_external_forces,
        rod_two_external_forces,
    ):
        # Compute element positions
        rod_one_element_position = node_to_element_position(
            rod_one_position_collection
        )
        rod_two_element_position = node_to_element_position(
            rod_two_position_collection
			)
        
        # Compute vector r*d2 (radius * normal vector) for each rod and element
        rod_one_rd2 = _batch_matvec(
				_batch_matrix_transpose(rod_one_director_collection),
                rod_one_rd2_local * rod_one_radius
			)
        rod_one_surface_position = rod_one_element_position + rod_one_rd2
        rod_two_rd2 = _batch_matvec(
				_batch_matrix_transpose(rod_two_director_collection),
                rod_two_rd2_local * rod_two_radius
			)
        rod_two_surface_position = rod_two_element_position + rod_two_rd2


        # Compute spring force between two rods using Fc=k*epsilon
        distance_vector = rod_two_surface_position - rod_one_surface_position
        distance = _batch_norm(distance_vector)
        spring_force = k * (distance_vector)

        # Damping force
        rod_one_element_velocity = node_to_element_velocity(
            rod_one_mass,
            rod_one_velocity_collection
        )
        rod_two_element_velocity = node_to_element_velocity(
            rod_two_mass,
            rod_two_velocity_collection
        )

        relative_velocity = rod_two_element_velocity - rod_one_element_velocity

        normalized_distance_vector = np.zeros((relative_velocity.shape))

        idx_nonzero_distance = np.where(distance >= 1e-12)[0]

        normalized_distance_vector[..., idx_nonzero_distance] = (
            distance_vector[..., idx_nonzero_distance] / distance[idx_nonzero_distance]
        )

        normal_relative_velocity_vector = (
            _batch_dot(relative_velocity, normalized_distance_vector)
            * normalized_distance_vector
        )

        damping_force = -nu * normal_relative_velocity_vector

        # Compute the total force
    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        if self._flag_initialize_To:
            self.BAt = _batch_matmul(_batch_matrix_transpose(rod_two.director_collection),
                                     rod_one.director_collection)
            self._flag_initialize_To = False

        omega = self._apply_torques(
            self.kt,
            self.spring_force,
            self.rod_one_rd2,
            self.rod_two_rd2,
            index_one,
            index_two,
            rod_one.director_collection,
            rod_two.director_collection,
            rod_one.external_torques,
            rod_two.external_torques,
            self.BAt,
        )

        # Safety Check
        if self.stability_check and np.abs(omega).max() > np.pi/4:
            warnings.warn("Parallel connection angle exceeded 45 degrees: Larger kt might be needed", RuntimeWarning)

    @staticmethod
    @njit(cache=True)
    def _apply_torques(
        kt,
        spring_force,
        rod_one_rd2,
        rod_two_rd2,
        index_one,
        index_two,
        rod_one_director_collection,
        rod_two_director_collection,
        rod_one_external_torques,
        rod_two_external_torques,
        BAt,
    ):
        # Compute torques due to the connection forces
        #spring_force *= kt * 1e-3
        torque_on_rod_one = _batch_cross(rod_one_rd2, spring_force)
        torque_on_rod_two = _batch_cross(rod_two_rd2, -spring_force)

        # Alignment Torque "Alignment Torque" 是一个物理术语，通常在描述力矩、旋转和对齐等方面的动态过程时使用。
        Tp = _batch_matmul(
                _batch_matmul(rod_two_director_collection, BAt),
                _batch_matrix_transpose(rod_one_director_collection)
            )
        omega = _inv_rotate(Tp) / 2.0
        #omega_mag = _batch_norm(omega)
        tau = omega * kt
        torque_on_rod_one += tau
        torque_on_rod_two -= tau

        # Change coordinate
        torque_on_rod_one_material_frame = _batch_matvec(
            rod_one_director_collection, torque_on_rod_one
        )
        torque_on_rod_two_material_frame = _batch_matvec(
            rod_two_director_collection, torque_on_rod_two
        )
        #rod_one_external_torques[:] += torque_on_rod_one_material_frame
        #rod_two_external_torques[:] += torque_on_rod_two_material_frame
        for k in range(torque_on_rod_one_material_frame.shape[-1]):
            rod_one_external_torques[0,k] += torque_on_rod_one_material_frame[0,k]
            rod_one_external_torques[1,k] += torque_on_rod_one_material_frame[1,k]
            rod_one_external_torques[2,k] += torque_on_rod_one_material_frame[2,k]

            rod_two_external_torques[0,k] += torque_on_rod_two_material_frame[0,k]
            rod_two_external_torques[1,k] += torque_on_rod_two_material_frame[1,k]
            rod_two_external_torques[2,k] += torque_on_rod_two_material_frame[2,k]
        return omega


class TipToTipStraightJoint(FreeJoint):
    """"""

    def __init__(self, k, nu, kt, rod1_rd2_local, rod2_rd2_local, stability_check=False):
        super().__init__(k, nu)
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned empirically
        # self.kr = 1e2
        self.kt = kt
        self.rod1_rd2_local = rod1_rd2_local
        self.rod2_rd2_local = rod2_rd2_local
        self._flag_initialize_To = True


        self.stability_check = stability_check

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        # TODO: documentation

        self.rod_one_rd2, self.rod_two_rd2, self.spring_force = self._apply_forces(
            self.k,
            self.nu,
            self.rod1_rd2_local,
            self.rod2_rd2_local,
            rod_one.position_collection,
            rod_two.position_collection,
            rod_one.radius[None,:],
            rod_two.radius[None,:],
            rod_one.velocity_collection,
            rod_two.velocity_collection,
			rod_one.director_collection,
			rod_two.director_collection,
			rod_one.tangents,
			rod_two.tangents,
            rod_one.external_forces,
            rod_two.external_forces,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_forces(
        k,
        nu,
		rod_one_rd2_local,
		rod_two_rd2_local,
        rod_one_position_collection,
        rod_two_position_collection,
        rod_one_radius,
        rod_two_radius,
        rod_one_velocity_collection,
        rod_two_velocity_collection,
        rod_one_director_collection,
        rod_two_director_collection,
        rod_one_tangents,
        rod_two_tangents,
        rod_one_external_forces,
        rod_two_external_forces,
    ):
        # Compute element positions
        rod_one_element_position = 0.5 * (rod_one_position_collection[...,-1] + rod_one_position_collection[...,-2])
        rod_two_element_position = 0.5 * (rod_two_position_collection[..., 0] + rod_two_position_collection[..., 1])

        # Compute vector r*d2 (radius * normal vector) for each rod and element
        rod_one_rd2 = rod_one_director_collection[...,-1].T @ (rod_one_rd2_local * rod_one_radius[...,-1])
        rod_one_surface_position = rod_one_element_position + rod_one_rd2
        rod_two_rd2 = rod_two_director_collection[...,0].T @ (rod_two_rd2_local * rod_two_radius[...,0])
        rod_two_surface_position = rod_two_element_position + rod_two_rd2

        # Compute spring force between two rods using Fc=k*epsilon
        distance_vector = rod_two_surface_position - rod_one_surface_position
        distance = np.linalg.norm(distance_vector)
        spring_force = k * (distance_vector)

        # Damping force
        rod_one_element_velocity = 0.5 * (rod_one_velocity_collection[...,-1] + rod_one_velocity_collection[...,-2])
        rod_two_element_velocity = 0.5 * (rod_two_velocity_collection[..., 0] + rod_two_velocity_collection[..., 1])
        relative_velocity = rod_two_element_velocity - rod_one_element_velocity

        '''
        normalized_distance_vector = np.zeros((relative_velocity.shape))

        idx_nonzero_distance = np.where(distance >= 1e-12)[0]

        normalized_distance_vector[..., idx_nonzero_distance] = (
            distance_vector[..., idx_nonzero_distance] / distance[idx_nonzero_distance]
        )

        normal_relative_velocity_vector = (
            _batch_dot(relative_velocity, normalized_distance_vector)
            * normalized_distance_vector
        )
        '''

        damping_force = -nu * relative_velocity

        # Compute the total force
        total_force = spring_force + damping_force

        # Re-distribute forces from elements to nodes.
        rod_one_external_forces[...,-1] += 0.5*total_force
        rod_one_external_forces[...,-2] += 0.5*total_force
        rod_two_external_forces[..., 0] -= 0.5*total_force
        rod_two_external_forces[..., 1] -= 0.5*total_force

        return rod_one_rd2, rod_two_rd2, total_force

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        if self._flag_initialize_To:
            self.BAt = rod_two.director_collection[...,0].T @ rod_one.director_collection[...,-1]
            self._flag_initialize_To = False

        # DEBUG
        rod_two.director_collection[..., 0] = rod_one.director_collection[..., -1]
        rod_two.omega_collection[..., 0] = rod_one.omega_collection[..., -1]
        rod_two.alpha_collection[..., 0] = rod_one.alpha_collection[..., -1]
        return

        omega = self._apply_torques(
            self.kt,
            self.spring_force,
            self.rod_one_rd2,
            self.rod_two_rd2,
            rod_one.director_collection[..., -1],
            rod_two.director_collection[...,  0],
            rod_one.external_torques,
            rod_two.external_torques,
            self.BAt,
        )

        # Safety Check
        if self.stability_check and np.abs(omega).max() > np.pi/4:
            warnings.warn("Parallel connection angle exceeded 45 degrees: Larger kt might be needed", RuntimeWarning)

    @staticmethod
    @njit(cache=True)
    def _apply_torques(
        kt,
        spring_force,
        rod_one_rd2,
        rod_two_rd2,
        rod_one_director,
        rod_two_director,
        rod_one_external_torques,
        rod_two_external_torques,
        BAt,
    ):
        # Compute torques due to the connection forces
        #spring_force *= kt * 1e-3
        torque_on_rod_one = np.cross(rod_one_rd2,  spring_force)
        torque_on_rod_two = np.cross(rod_two_rd2, -spring_force)

        # Alignment Torque
        Tp = (rod_two_director @ BAt) @ rod_one_director.T
            
        omega = _single_inv_rotate(Tp) / 2.0
        #omega_mag = _batch_norm(omega)
        tau = omega * kt
        torque_on_rod_one += tau
        torque_on_rod_two -= tau

        # Change coordinate
        torque_on_rod_one_material_frame = rod_one_director @ torque_on_rod_one
        torque_on_rod_two_material_frame = rod_two_director @ torque_on_rod_two

        # Add torque
        rod_one_external_torques[...,-1] += torque_on_rod_one_material_frame
        rod_two_external_torques[..., 0] += torque_on_rod_two_material_frame

        return omega


#10.24GJM
#try to add a streching force to an SurfaceJointSideBySide soft arm
#not sure which value should I put
k = 1e4
nu = 1e-4 #粘性阻尼系数
kt = 1
rd1_local = 5
rd2_local = 5
SurfaceJointSideBySide_try_1 = SurfaceJointSideBySide(k,nu,kt,rd1_local,rd2_local)

rod_one_spec = {
    'n_elements' : 100,
    'start' : np.zeros((3,)),
    'direction' : np.array([0.0,0.0,1.0]), #Q matrix
    'normal' : np.array([0.0,1.0,0.0]), #法向量
    'base_length' : 1,
    'base_radius' : 0.1,
    'density' : 1000,
    'nu' : None,
    'youngs_modulus' : 1e6,
    'outer_radius' : 0.005,
    'inner_radius' : 0.002,
    'damping_constant' : 1e-4
           }

rod_two_spec = {
    'n_elements' : 100,
    'start' : np.array([0.01,0,0]),
    'direction' : np.array([0.0,0.0,1.0]),
    'normal' : np.array([0.0,1.0,0.0]),
    'base_length' : 1,
    'base_radius' : 0.1,
    'density' : 1000,
    'nu' : None,
    'youngs_modulus' : 1e6,
    'outer_radius' : 0.005,#现在暂时没用
    'inner_radius' : 0.002,
    'damping_constant' : 1e-4
           }

time_step = 1e-4
class Environment:
    def __init__(self,time_step):
        self.time_step = time_step

env = Environment(time_step)
assembly = FreeAssembly(env)
simulator = assembly.simulator

rod_one = assembly.create_rod(name='A_01_rodone',is_first_segment=True, verbose=False, **rod_one_spec)
rod_two = assembly.create_rod(name='A_02_rodtwo',is_first_segment=True, verbose=False, **rod_two_spec)

assembly.glue_rods_surface_connection(rod_one,rod_two,k,nu,kt)#这两根rods依旧是分开的，所以仍旧需要对两根都施加力，同时用callback



#how to add external forces?????

# position_collection = np.array(position)  # 杆的位置集合
# velocity_collection = np.array(velocity)  # 杆的速度集合
# director_collection = np.array(director)  #杆的方向集合
# tangents = tangents                       #杆的切线
# external_forces = external_forces
# radius = radius  # 杆的半径（示例属性）
# radius = np.array([0.005])[np.newaxis, :]
# radius = radius = np.array[0.005]
# rod_one = Rod(mass=1,position=[0.,0.,0.],velocity=[0.,0.,0.],radius= radius, director=[1.,0.,0.],tangents=[0.,0.,1.],external_forces=[1.,0.,0.])
# rod_two = Rod(mass=1,position=[0.,0.,0.],velocity=[0.,0.,0.],radius= radius, director=[1.,0.,0.],tangents=[0.,0.,1.],external_forces=[1.,0.,0.])

# SurfaceJointSideBySide_try_1.apply_forces(rod_one,None,rod_two,None)
simulator.add_forcing_to(rod_one).using()
simulator.add_forcing_to(rod_two).using()

#callback 参考butterfly 两个rods都要写 def make_callback(self, system, time, current_step: int):
#  固定用函数（也可以模仿timoshenko）给两个rods固定  def tip_to_base_connection(self, rod1, rod2, k, nu, kt):

#def plot_video_2D
#making vedio





