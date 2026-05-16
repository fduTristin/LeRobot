import time
import json
from pathlib import Path

from lerobot.robots.xlerobot import XLerobot, XLerobotConfig


def main() -> None:
    cfg = XLerobotConfig(id="my_xlerobot")
    robot = XLerobot(cfg)

    try:
        robot.connect()
        print("[MAIN] Connected to robot.\n")
    except Exception as e:
        print(f"[MAIN] Failed to connect: {e}")
        return

    saved_path = Path(__file__).resolve().parent / "head_home_position.json"

    try:
        print("=" * 50)
        print("Head Position Measurer")
        print("=" * 50)
        print(f"Saving to: {saved_path}")
        print()
        print("当前头部电机位置（持续刷新，按 Enter 记录当前值并退出）：")
        print()

        while True:
            obs = robot.get_observation()
            h1 = obs.get("head_motor_1.pos", None)
            h2 = obs.get("head_motor_2.pos", None)

            if h1 is not None and h2 is not None:
                print(f"\r  head_motor_1 = {h1:>8.2f}    head_motor_2 = {h2:>8.2f}   ", end="", flush=True)
            else:
                print(f"\r  无法读取头部电机位置", end="", flush=True)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n[MAIN] 退出（未保存）.")
    else:
        # 按 Enter 触发 EOFError 跳出循环，正常记录
        pass

    # 读取最终位置
    obs = robot.get_observation()
    h1 = obs.get("head_motor_1.pos", 0.0)
    h2 = obs.get("head_motor_2.pos", 0.0)

    data = {
        "head_motor_1": round(h1, 4),
        "head_motor_2": round(h2, 4),
    }

    with open(saved_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[MAIN] 已保存: {saved_path}")
    print(f"  head_motor_1 = {h1}")
    print(f"  head_motor_2 = {h2}")
    print("\n请将以下内容更新到 7_xlerobot_teleop_joycon.py 中的 zero_pos：")
    print(f'  self.zero_pos = {data}')
    print()

    robot.disconnect()
    print("[MAIN] Done.")


if __name__ == "__main__":
    main()
