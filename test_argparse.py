import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="这是一个示例程序。")

# 添加参数
parser.add_argument("--tag", type=str,help="标签描述", required=True)

# 解析命令行参数
args = parser.parse_args()

print(f"接收到的标签为: {args.tag}")
