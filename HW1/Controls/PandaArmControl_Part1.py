import mujoco as mj
from mujoco import viewer
import numpy as np
import math


# Set the XML filepath
xml_filepath = "../franka_emika_panda/panda_nohand_torque_fixed_board.xml"

################################# Control Callback Definitions #############################

# Control callback for gravity compensation
def gravity_comp(model, data):
    # data.ctrl exposes the member that sets the actuator control inputs that participate in the
    # physics, data.qfrc_bias exposes the gravity forces expressed in generalized coordinates, i.e.
    # as torques about the joints

    data.ctrl[:7] = data.qfrc_bias[:7]

# Force control callback
def force_control(model, data):
    # Implement a force control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.

    # Instantiate a handle to the desired body on the robot
    body = data.body("hand")
    # Get the Jacobian for the desired location on the robot (The end-effector)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mj.mj_jacBody(model, data, jacp, jacr, body.id)
    # This function works by taking in return parameters!!! Make sure you supply it with placeholder
    # variables

    # Specify the desired force in global coordinates
    desired_force = np.array([15.0, 0.0, 0.0])
    # Compute the required control input using desied force values
    torques = data.qfrc_bias + jacp.T @ desired_force
    # Set the control inputs
    data.ctrl[:7] = torques[:7]
    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Force readings updated here
    force[:] = np.roll(force, -1)[:]
    force[-1] = data.sensordata[2]

# Control callback for an impedance controller
def impedance_control(model, data):
    # Implement an impedance control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.

    # Instantiate a handle to the desired body on the robot
    body = data.body("hand")
    # Set the desired position
    desired_position = target_position
    # Set the desired velocities
    v_d = 0
    # Set the desired orientation (Use numpy quaternion manipulation functions)
    desired_orientation = initial_orientation
    # Get the current orientation
    current_orientation = body.xmat.reshape(3, 3)
    # Get orientation error
    orientation_error =  0.5 * (
    np.cross(current_orientation[:, 0], desired_orientation[:, 0])
    + np.cross(current_orientation[:, 1], desired_orientation[:, 1])
    + np.cross(current_orientation[:, 2], desired_orientation[:, 2])
    )
    # Get the position error
    position_error = desired_position - body.xpos
    # Get the Jacobian at the desired location on the robot
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mj.mj_jacBody(model, data, jacp, jacr, body.id)

    # This function works by taking in return parameters!!! Make sure you supply it with placeholder
    # variables

    # Compute the impedance control input torques
    Kp = 1000.0
    Kd = 1000.0
    linear_velocity = jacp @ data.qvel
    force_cmd = Kp * position_error + Kd * (v_d - linear_velocity)
    angular_velocity = jacr @ data.qvel
    K_R = 50.0
    D_R = 10.0
    moment_cmd = K_R * orientation_error - D_R * angular_velocity
    torque = data.qfrc_bias + jacp.T @ force_cmd + jacr.T @ moment_cmd
    # Set the control inputs
    data.ctrl[:7] = torque[:7]
    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Update force sensor readings
    force[:] = np.roll(force, -1)[:]
    force[-1] = data.sensordata[2]


def position_control(model, data):
    # Instantiate a handle to the desired body on the robot
    body = data.body("hand")

    # Set the desired joint angle positions
    desired_joint_positions = np.array(
        [0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

    # Set the desired joint velocities
    desired_joint_velocities = np.array([0, 0, 0, 0, 0, 0, 0])

    # Desired gain on position error (K_p)
    Kp = 1000

    # Desired gain on velocity error (K_d)
    Kd = 1000

    # Set the actuator control torques
    data.ctrl[:7] = data.qfrc_bias[:7] + Kp * \
        (desired_joint_positions-data.qpos[:7]) + Kd * \
        (np.array([0, 0, 0, 0, 0, 0, 0])-data.qvel[:7])


####################################### MAIN #####################################
if __name__ == "__main__":
    # Load the xml file here
    model = mj.MjModel.from_xml_path(xml_filepath)
    data = mj.MjData(model)

    # Set the simulation scene to the home configuration
    mj.mj_resetDataKeyframe(model, data, 0)

    ################################# Swap Callback Below This Line #################################
    # This is where you can set the control callback. Take a look at the Mujoco documentation for more
    # details. Very briefly, at every timestep, a user-defined callback function can be provided to
    # mujoco that sets the control inputs to the actuator elements in the model. The gravity
    # compensation callback has been implemented for you. Run the file and play with the model as
    # explained in the PDF

    mj.mj_forward(model, data)
    initial_position = data.body("hand").xpos.copy()
    initial_orientation = data.body("hand").xmat.reshape(3, 3).copy()
    target_position = initial_position.copy()
    target_position[0] += 0.0457174

    mj.set_mjcb_control(gravity_comp)  # set Position, Force and Impedance control and test them

    ################################# Swap Callback Above This Line #################################

    # Initialize variables to store force and time data points
    force_sensor_max_time = 10
    force = np.zeros(int(force_sensor_max_time/model.opt.timestep))
    time = np.linspace(0, force_sensor_max_time, int(
        force_sensor_max_time/model.opt.timestep))

    # Launch the simulate viewer
    # viewer.launch(model, data) # this will indefinitely run the simulation until the viewer is closed
    
    with viewer.launch_passive(model, data) as v:
        for _ in range(int(force_sensor_max_time/model.opt.timestep)):
            mj.mj_step(model, data)
            v.sync()
        print(data.qpos[:7])
        mj.mj_forward(model, data)
        input("Press anything to continue...") # omit this to autoclose the viewer when time is up



    # Save recorded force and time points as a csv file
    force = np.reshape(force, (5000, 1))
    time = np.reshape(time, (5000, 1))
    plot = np.concatenate((time, force), axis=1)
    np.savetxt('force_vs_time.csv', plot, delimiter=',')
