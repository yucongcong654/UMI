import math
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

def print_joints_rad(arm: OpenArmFollower, label: str):
    obs = arm.get_observation()
    rad_positions = {}
    
    # OpenArm has 7 joints + 1 gripper
    for i in range(1, 8):
        name = f"joint_{i}.pos"
        if name in obs:
            deg_val = obs[name]
            rad_positions[f"joint_{i}"] = math.radians(deg_val)
            
    if "gripper.pos" in obs:
        rad_positions["gripper"] = math.radians(obs["gripper.pos"])
        
    print(f"\n--- {label} ---")
    for k, v in rad_positions.items():
        print(f"{k}: {v:.6f} rad")
    print("-------------------\n")

def main():
    config = OpenArmFollowerConfig(
        id="my_openarm_umi_left2",
        port="can1",
        side="left",
        can_interface="socketcan",
        max_relative_target=8.0,
    )
    arm = OpenArmFollower(config)
    
    print("Connecting to OpenArm (left)...")
    arm.connect(calibrate=True)
    
    print("Disabling torque (entering zero-torque/freedrive mode)...")
    if hasattr(arm, "bus") and hasattr(arm.bus, "disable_torque"):
        arm.bus.disable_torque()
    else:
        print("Warning: Could not find disable_torque method on arm.bus")
    
    print_joints_rad(arm, "Initial Joint Positions")
    
    while True:
        obs = arm.get_observation()
        print(math.radians(obs["gripper.pos"])
        )
    input("Press ENTER to read final positions and close connection...")
    
    print_joints_rad(arm, "Final Joint Positions")
    
    print("Disconnecting...")
    arm.disconnect()

if __name__ == "__main__":
    main()
