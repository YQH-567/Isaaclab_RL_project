import lebai_sdk

lebai_sdk.init()
robot = lebai_sdk.connect("机器人IP", False)

kin_data = robot.get_kin_data()

print(kin_data["actual_joint_pose"])