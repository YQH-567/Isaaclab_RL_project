import lebai_sdk

lebai_sdk.init()
robot = lebai_sdk.connect("192.168.0.50", False)

kin_data = robot.get_kin_data()

print(kin_data["actual_joint_pose"])